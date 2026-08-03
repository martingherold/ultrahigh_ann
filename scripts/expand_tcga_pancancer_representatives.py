#!/usr/bin/env python3
"""Learn multiple training-only subcentroids per TCGA cancer project."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError as error:
    print(
        "expand_tcga_pancancer_representatives.py requires NumPy.",
        file=sys.stderr,
    )
    raise SystemExit(2) from error


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "processed" / "tcga_pancancer_v1"
HASH_BLOCK_SIZE = 1024 * 1024
MANAGED_NAMES = {
    "representatives.npy",
    "representative_labels.npy",
    "representatives.csv",
    "queries.npy",
    "query_labels.npy",
    "features.csv",
    "samples.csv",
    "source_manifest.json",
    "source_dataset.json",
    "dataset.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cluster each Pan-Cancer representative-pool project separately "
            "and replace its single centroid with K training-only subcentroids."
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
        help="destination (default: data/processed/tcga_pancancer_kK_v1)",
    )
    parser.add_argument(
        "--representatives-per-class",
        type=int,
        default=4,
        help="subcentroids learned per cancer project (default: 4)",
    )
    parser.add_argument(
        "--clustering-features",
        type=int,
        default=256,
        help=(
            "highest-variance genes per project used for assignments "
            "(default: 256); full dimensions are used for final centroids"
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
        help="genes processed together when constructing centroids (default: 2048)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace managed files from an earlier expansion",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.representatives_per_class < 1:
        raise ValueError("--representatives-per-class must be positive")
    if args.clustering_features < 1:
        raise ValueError("--clustering-features must be positive")
    if args.max_iterations < 1:
        raise ValueError("--max-iterations must be positive")
    if args.feature_block_size < 1:
        raise ValueError("--feature-block-size must be positive")


def load_matrix(path: Path) -> np.ndarray:
    if not path.is_file():
        raise RuntimeError(f"missing required matrix: {path}")
    matrix = np.load(path, mmap_mode="r", allow_pickle=False)
    if matrix.ndim != 2 or matrix.dtype != np.dtype("<f4"):
        raise RuntimeError(f"{path.name} must be a two-dimensional float32 NPY")
    return matrix


def load_labels(path: Path, expected_rows: int) -> np.ndarray:
    if not path.is_file():
        raise RuntimeError(f"missing required labels: {path}")
    labels = np.load(path, mmap_mode="r", allow_pickle=False)
    if labels.ndim != 1 or labels.dtype != np.dtype("<u2"):
        raise RuntimeError(f"{path.name} must be a one-dimensional uint16 NPY")
    if labels.size != expected_rows:
        raise RuntimeError(f"{path.name} row count does not match its matrix")
    return labels


def read_class_metadata(path: Path) -> dict[int, dict[str, str]]:
    if not path.is_file():
        raise RuntimeError(f"missing representative metadata: {path}")
    result: dict[int, dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            label = int(row["label"])
            if label in result:
                raise RuntimeError(f"duplicate class label in {path.name}: {label}")
            result[label] = row
    if not result:
        raise RuntimeError(f"{path.name} contains no representatives")
    return result


def top_variance_features(
    pool: np.ndarray,
    rows: np.ndarray,
    count: int,
    block_size: int,
) -> np.ndarray:
    dimensions = pool.shape[1]
    count = min(count, dimensions)
    variances = np.empty(dimensions, dtype=np.float64)
    for start in range(0, dimensions, block_size):
        end = min(start + block_size, dimensions)
        values = np.asarray(pool[rows, start:end], dtype=np.float64)
        variances[start:end] = np.var(values, axis=0)
    if count == dimensions:
        return np.arange(dimensions, dtype=np.intp)
    selected = np.argpartition(variances, -count)[-count:]
    order = np.lexsort((selected, -variances[selected]))
    return selected[order].astype(np.intp, copy=False)


def initialize_kmeans_plus_plus(
    values: np.ndarray,
    cluster_count: int,
    random_engine: np.random.Generator,
) -> np.ndarray:
    centers = np.empty((cluster_count, values.shape[1]), dtype=np.float64)
    first = int(random_engine.integers(values.shape[0]))
    centers[0] = values[first]
    minimum_distances = np.sum((values - centers[0]) ** 2, axis=1)
    chosen = {first}
    for cluster in range(1, cluster_count):
        total = float(np.sum(minimum_distances))
        if total == 0.0:
            index = next(i for i in range(values.shape[0]) if i not in chosen)
        else:
            threshold = random_engine.random() * total
            index = int(np.searchsorted(np.cumsum(minimum_distances), threshold))
            index = min(index, values.shape[0] - 1)
            if index in chosen:
                index = max(
                    (i for i in range(values.shape[0]) if i not in chosen),
                    key=lambda i: minimum_distances[i],
                )
        chosen.add(index)
        centers[cluster] = values[index]
        distances = np.sum((values - centers[cluster]) ** 2, axis=1)
        minimum_distances = np.minimum(minimum_distances, distances)
    return centers


def assign_clusters(values: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distances = (
        np.sum(values * values, axis=1)[:, None]
        + np.sum(centers * centers, axis=1)[None, :]
        - 2.0 * values @ centers.T
    )
    np.maximum(distances, 0.0, out=distances)
    assignments = np.argmin(distances, axis=1).astype(np.intp, copy=False)
    minimum_distances = distances[np.arange(values.shape[0]), assignments]
    return assignments, minimum_distances


def repair_empty_clusters(
    assignments: np.ndarray,
    minimum_distances: np.ndarray,
    cluster_count: int,
) -> None:
    counts = np.bincount(assignments, minlength=cluster_count)
    for empty in np.flatnonzero(counts == 0):
        candidates = np.argsort(minimum_distances)[::-1]
        for index in candidates:
            source = assignments[index]
            if counts[source] > 1:
                assignments[index] = empty
                counts[source] -= 1
                counts[empty] += 1
                minimum_distances[index] = 0.0
                break
        else:
            raise RuntimeError("cannot repair an empty k-means cluster")


def kmeans(
    values: np.ndarray,
    cluster_count: int,
    seed: int,
    max_iterations: int,
) -> tuple[np.ndarray, int, float]:
    values64 = np.asarray(values, dtype=np.float64)
    random_engine = np.random.default_rng(seed)
    centers = initialize_kmeans_plus_plus(values64, cluster_count, random_engine)
    previous: np.ndarray | None = None
    for iteration in range(1, max_iterations + 1):
        assignments, minimum_distances = assign_clusters(values64, centers)
        repair_empty_clusters(assignments, minimum_distances, cluster_count)
        centers = np.vstack(
            [values64[assignments == cluster].mean(axis=0) for cluster in range(cluster_count)]
        )
        if previous is not None and np.array_equal(assignments, previous):
            final_assignments, final_distances = assign_clusters(values64, centers)
            repair_empty_clusters(final_assignments, final_distances, cluster_count)
            return final_assignments, iteration, float(np.sum(final_distances))
        previous = assignments.copy()
    final_assignments, final_distances = assign_clusters(values64, centers)
    repair_empty_clusters(final_assignments, final_distances, cluster_count)
    return final_assignments, max_iterations, float(np.sum(final_distances))


def hardlink(source: Path, destination: Path) -> None:
    if source.is_file():
        os.link(source, destination)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(HASH_BLOCK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
        input_dir = args.input_dir.expanduser().resolve()
        if not input_dir.is_dir():
            raise RuntimeError(f"input directory does not exist: {input_dir}")
        output_dir = (
            args.output_dir.expanduser().resolve()
            if args.output_dir is not None
            else PROJECT_ROOT
            / "data"
            / "processed"
            / f"tcga_pancancer_k{args.representatives_per_class}_v1"
        )

        pool_path = input_dir / "representative_pool.npy"
        pool_labels_path = input_dir / "representative_pool_labels.npy"
        if not pool_path.is_file() or not pool_labels_path.is_file():
            raise RuntimeError(
                "the representative pool is not materialized; run "
                "'python3 scripts/transform_tcga_pancancer.py "
                "--write-representative-pool --force' first"
            )
        pool = load_matrix(pool_path)
        pool_labels = load_labels(pool_labels_path, pool.shape[0])
        queries = load_matrix(input_dir / "queries.npy")
        query_labels = load_labels(input_dir / "query_labels.npy", queries.shape[0])
        if pool.shape[1] != queries.shape[1]:
            raise RuntimeError("pool and query feature dimensions differ")
        class_metadata = read_class_metadata(input_dir / "representatives.csv")
        labels = tuple(sorted(class_metadata))
        if set(int(value) for value in np.unique(pool_labels)) != set(labels):
            raise RuntimeError("pool labels do not match representatives.csv")
        if not set(int(value) for value in np.unique(query_labels)).issubset(labels):
            raise RuntimeError("query labels do not match representatives.csv")

        counts = {
            label: int(np.count_nonzero(pool_labels == label)) for label in labels
        }
        smallest = min(counts.values())
        if args.representatives_per_class > smallest:
            raise RuntimeError(
                f"requested {args.representatives_per_class} representatives per "
                f"class, but the smallest pool has only {smallest} samples"
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

        representative_count = len(labels) * args.representatives_per_class
        print(
            f"Learning {args.representatives_per_class} subcentroids for each "
            f"of {len(labels)} projects ({representative_count} representatives).",
            flush=True,
        )
        with tempfile.TemporaryDirectory(
            prefix=".tcga-subcentroids-",
            dir=output_dir,
        ) as temporary_directory:
            staging = Path(temporary_directory)
            representatives = np.lib.format.open_memmap(
                staging / "representatives.npy",
                mode="w+",
                dtype=np.dtype("<f4"),
                shape=(representative_count, pool.shape[1]),
                fortran_order=False,
            )
            representative_labels = np.empty(representative_count, dtype=np.dtype("<u2"))
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
                    args.representatives_per_class,
                    args.seed + label * 1_000_003,
                    args.max_iterations,
                )
                for cluster in range(args.representatives_per_class):
                    member_rows = rows[assignments == cluster]
                    for start in range(0, pool.shape[1], args.feature_block_size):
                        end = min(start + args.feature_block_size, pool.shape[1])
                        values = np.asarray(pool[member_rows, start:end], dtype=np.float64)
                        representatives[output_row, start:end] = np.mean(values, axis=0)
                    representative_labels[output_row] = label
                    metadata = class_metadata[label]
                    records.append(
                        {
                            "row_index": output_row,
                            "label": label,
                            "project_id": metadata["project_id"],
                            "cancer_type": metadata["cancer_type"],
                            "cluster": cluster,
                            "construction": "training_pool_kmeans_subcentroid",
                            "cluster_sample_count": member_rows.size,
                            "project_pool_sample_count": rows.size,
                            "clustering_feature_count": features.size,
                            "kmeans_iterations": iterations,
                            "reduced_space_inertia": inertia,
                        }
                    )
                    output_row += 1
                print(
                    f"  [{class_index:02}/{len(labels)}] "
                    f"{class_metadata[label]['project_id']}: "
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
                "source_manifest.json",
            ):
                hardlink(input_dir / name, staging / name)
            hardlink(input_dir / "dataset.json", staging / "source_dataset.json")
            metadata = {
                "dataset_name": "TCGA Pan-Cancer training-pool subcentroids",
                "format_version": 2,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_dataset": str(input_dir),
                "representatives": {
                    "file": "representatives.npy",
                    "shape": [representative_count, pool.shape[1]],
                    "dtype": "float32",
                    "labels_file": "representative_labels.npy",
                    "labels_dtype": "uint16",
                    "classes": len(labels),
                    "count_per_class": args.representatives_per_class,
                    "construction": "per-class k-means subcentroids",
                    "sha256": sha256(staging / "representatives.npy"),
                },
                "clustering": {
                    "assignment_space": "highest within-project variance genes",
                    "requested_feature_count": args.clustering_features,
                    "final_centroid_space": "all source features",
                    "initialization": "k-means++",
                    "seed": args.seed,
                    "maximum_iterations": args.max_iterations,
                    "metadata_file": "representatives.csv",
                },
                "queries": {
                    "file": "queries.npy",
                    "shape": list(queries.shape),
                    "labels_file": "query_labels.npy",
                    "storage": "hardlink_to_source_dataset",
                },
                "leakage_control": (
                    "only representative_pool.npy rows were used for feature "
                    "selection, clustering, and centroid construction"
                ),
                "numpy_version": np.__version__,
            }
            with (staging / "dataset.json").open("w", encoding="utf-8") as output:
                json.dump(metadata, output, indent=2, sort_keys=True)
                output.write("\n")

            staged_names = {path.name for path in staging.iterdir() if path.is_file()}
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
        f"Saved {representative_count} representatives and {queries.shape[0]} "
        f"queries to {output_dir}.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
