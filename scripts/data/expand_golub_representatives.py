#!/usr/bin/env python3
"""Learn several training-only subcentroids for each Golub diagnosis."""

from __future__ import annotations

import argparse
import csv
import errno
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from expand_tcga_pancancer_representatives import (
    kmeans,
    load_labels,
    load_matrix,
    sha256,
    top_variance_features,
)

try:
    import numpy as np
except ModuleNotFoundError as error:
    print(
        "expand_golub_representatives.py requires NumPy.",
        file=sys.stderr,
    )
    raise SystemExit(2) from error


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "processed" / "golub_v1"
MANAGED_NAMES = {
    "representatives.npy",
    "representative_labels.npy",
    "representatives.csv",
    "queries.npy",
    "query_labels.npy",
    "features.csv",
    "samples.csv",
    "normalization_mean.npy",
    "normalization_scale.npy",
    "source_manifest.json",
    "source_dataset.json",
    "dataset.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cluster the ALL and AML training samples separately and replace "
            "the two class means with an even total number of subcentroids."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"base transformed dataset (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="destination (default: data/processed/golub_rTOTAL_v1)",
    )
    parser.add_argument(
        "--total-representatives",
        type=int,
        default=4,
        help="even total split equally between ALL and AML (default: 4)",
    )
    parser.add_argument(
        "--clustering-features",
        type=int,
        default=256,
        help=(
            "highest-variance training probes per class used for assignments "
            "(default: 256); final centroids use all dimensions"
        ),
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=50,
        help="maximum Lloyd iterations (default: 50)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="deterministic k-means++ seed (default: 42)",
    )
    parser.add_argument(
        "--feature-block-size",
        type=int,
        default=2048,
        help="probes processed together for full centroids (default: 2048)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace managed files from an earlier expansion",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.total_representatives < 2 or args.total_representatives % 2 != 0:
        raise ValueError("--total-representatives must be a positive even number")
    if args.clustering_features < 1:
        raise ValueError("--clustering-features must be positive")
    if args.max_iterations < 1:
        raise ValueError("--max-iterations must be positive")
    if args.feature_block_size < 1:
        raise ValueError("--feature-block-size must be positive")


def read_class_metadata(path: Path) -> dict[int, str]:
    if not path.is_file():
        raise RuntimeError(f"missing representative metadata: {path}")
    result: dict[int, str] = {}
    with path.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            try:
                label = int(row["label"])
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError(
                    f"{path.name} has invalid representative labels"
                ) from error
            if label in result:
                raise RuntimeError(f"duplicate class label in {path.name}: {label}")
            result[label] = (
                row.get("diagnosis") or row.get("outcome") or f"class_{label}"
            )
    if not result:
        raise RuntimeError(f"{path.name} contains no representatives")
    return result


def hardlink_required(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise RuntimeError(f"missing required source file: {source}")
    try:
        os.link(source, destination)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        shutil.copyfile(source, destination)


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
        input_dir = args.input_dir.expanduser().resolve()
        if not input_dir.is_dir():
            raise RuntimeError(f"input directory does not exist: {input_dir}")
        if args.output_dir is None:
            output_dir = (
                PROJECT_ROOT
                / "data"
                / "processed"
                / f"golub_r{args.total_representatives}_v1"
            )
        else:
            requested_output = args.output_dir.expanduser()
            if requested_output.is_symlink():
                raise RuntimeError(
                    f"refusing to write through symbolic-link output: "
                    f"{requested_output}"
                )
            output_dir = requested_output.resolve()

        pool = load_matrix(input_dir / "representative_pool.npy")
        pool_labels = load_labels(
            input_dir / "representative_pool_labels.npy", pool.shape[0]
        )
        queries = load_matrix(input_dir / "queries.npy")
        query_labels = load_labels(
            input_dir / "query_labels.npy", queries.shape[0]
        )
        if pool.shape[1] != queries.shape[1]:
            raise RuntimeError("pool and query feature dimensions differ")

        class_names = read_class_metadata(input_dir / "representatives.csv")
        labels = tuple(sorted(class_names))
        if len(labels) != 2:
            raise RuntimeError(
                f"Golub expansion requires two classes; found {len(labels)}"
            )
        if set(int(value) for value in np.unique(pool_labels)) != set(labels):
            raise RuntimeError("pool labels do not match representatives.csv")
        if not set(int(value) for value in np.unique(query_labels)).issubset(labels):
            raise RuntimeError("query labels do not match representatives.csv")

        representatives_per_class = args.total_representatives // len(labels)
        counts = {
            label: int(np.count_nonzero(pool_labels == label)) for label in labels
        }
        smallest = min(counts.values())
        if representatives_per_class > smallest:
            raise RuntimeError(
                f"requested {representatives_per_class} representatives per class, "
                f"but the smallest training class has only {smallest} samples"
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(
            name for name in MANAGED_NAMES if (output_dir / name).exists()
        )
        if existing and not args.force:
            print(
                f"Refusing to overwrite existing files in {output_dir}: "
                f"{', '.join(existing)}\nRun again with --force to replace them.",
                file=sys.stderr,
            )
            return 1

        print(
            f"Learning {representatives_per_class} subcentroids for each "
            f"diagnosis ({args.total_representatives} total).",
            flush=True,
        )
        with tempfile.TemporaryDirectory(
            prefix=".golub-subcentroids-", dir=output_dir
        ) as temporary_directory:
            staging = Path(temporary_directory)
            representatives = np.lib.format.open_memmap(
                staging / "representatives.npy",
                mode="w+",
                dtype=np.dtype("<f4"),
                shape=(args.total_representatives, pool.shape[1]),
                fortran_order=False,
            )
            representative_labels = np.empty(
                args.total_representatives, dtype=np.dtype("<u2")
            )
            records: list[dict[str, object]] = []
            output_row = 0
            for class_index, label in enumerate(labels, start=1):
                rows = np.flatnonzero(pool_labels == label).astype(np.intp)
                features = top_variance_features(
                    pool,
                    rows,
                    args.clustering_features,
                    args.feature_block_size,
                )
                clustering_values = np.asarray(pool[np.ix_(rows, features)])
                assignments, iterations, inertia = kmeans(
                    clustering_values,
                    representatives_per_class,
                    args.seed + label * 1_000_003,
                    args.max_iterations,
                )
                for cluster in range(representatives_per_class):
                    member_rows = rows[assignments == cluster]
                    for start in range(0, pool.shape[1], args.feature_block_size):
                        end = min(start + args.feature_block_size, pool.shape[1])
                        values = np.asarray(
                            pool[member_rows, start:end], dtype=np.float64
                        )
                        representatives[output_row, start:end] = np.mean(
                            values, axis=0
                        )
                    representative_labels[output_row] = label
                    records.append(
                        {
                            "representative_row": output_row,
                            "label": label,
                            "diagnosis": class_names[label],
                            "cluster": cluster,
                            "construction": "training_pool_kmeans_subcentroid",
                            "cluster_sample_count": member_rows.size,
                            "class_pool_sample_count": rows.size,
                            "clustering_feature_count": features.size,
                            "kmeans_iterations": iterations,
                            "reduced_space_inertia": inertia,
                        }
                    )
                    output_row += 1
                print(
                    f"  [{class_index}/{len(labels)}] {class_names[label]}: "
                    f"cluster sizes {np.bincount(assignments).tolist()}",
                    flush=True,
                )

            representatives.flush()
            del representatives
            np.save(
                staging / "representative_labels.npy",
                representative_labels,
                allow_pickle=False,
            )
            with (staging / "representatives.csv").open(
                "w", encoding="utf-8", newline=""
            ) as output:
                writer = csv.DictWriter(output, fieldnames=tuple(records[0]))
                writer.writeheader()
                writer.writerows(records)

            for name in (
                "queries.npy",
                "query_labels.npy",
                "features.csv",
                "samples.csv",
                "normalization_mean.npy",
                "normalization_scale.npy",
                "source_manifest.json",
            ):
                hardlink_required(input_dir / name, staging / name)
            hardlink_required(
                input_dir / "dataset.json", staging / "source_dataset.json"
            )

            metadata = {
                "dataset_name": "Golub training-pool subcentroids",
                "format_version": 1,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_dataset": str(input_dir),
                "representatives": {
                    "file": "representatives.npy",
                    "shape": [args.total_representatives, pool.shape[1]],
                    "dtype": "float32",
                    "labels_file": "representative_labels.npy",
                    "labels_dtype": "uint16",
                    "classes": len(labels),
                    "total_count": args.total_representatives,
                    "count_per_class": representatives_per_class,
                    "counts_by_class": {
                        class_names[label]: representatives_per_class
                        for label in labels
                    },
                    "construction": "per-class k-means subcentroids",
                    "sha256": sha256(staging / "representatives.npy"),
                },
                "clustering": {
                    "assignment_space": (
                        "highest within-class variance training probes"
                    ),
                    "requested_feature_count": args.clustering_features,
                    "final_centroid_space": "all source probes",
                    "initialization": "k-means++",
                    "seed": args.seed,
                    "maximum_iterations": args.max_iterations,
                    "metadata_file": "representatives.csv",
                },
                "queries": {
                    "file": "queries.npy",
                    "shape": list(queries.shape),
                    "labels_file": "query_labels.npy",
                    "storage": "hardlink_to_source_dataset_or_copy_across_filesystems",
                },
                "leakage_control": (
                    "only published training rows were used for variance "
                    "ranking, clustering, and centroid construction; published "
                    "test queries are carried over unchanged"
                ),
                "numpy_version": np.__version__,
            }
            with (staging / "dataset.json").open(
                "w", encoding="utf-8"
            ) as output:
                json.dump(metadata, output, indent=2, sort_keys=True)
                output.write("\n")

            staged_names = {
                path.name for path in staging.iterdir() if path.is_file()
            }
            if args.force:
                for stale in MANAGED_NAMES - staged_names:
                    path = output_dir / stale
                    if path.is_file():
                        path.unlink()
            for name in sorted(staged_names):
                (staging / name).replace(output_dir / name)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(
        f"Saved {args.total_representatives} representatives and "
        f"{queries.shape[0]} queries to {output_dir}.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
