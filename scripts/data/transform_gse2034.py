#!/usr/bin/env python3
"""Transform GSE2034 into a leakage-safe binary centroid benchmark."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import random
import shutil
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from download_gse2034 import (
    CLINICAL_NAME,
    EXPECTED_FEATURES,
    EXPECTED_RELAPSE_COUNTS,
    EXPECTED_SAMPLES,
    MATRIX_NAME,
    ClinicalSample,
    read_clinical_samples,
    validate_clinical_table,
)

try:
    import numpy as np
except ModuleNotFoundError as error:
    print(
        "transform_gse2034.py requires NumPy; install it with "
        "'python3 -m pip install numpy'.",
        file=sys.stderr,
    )
    raise SystemExit(2) from error


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "gse2034"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "gse2034_v1"
HASH_BLOCK_SIZE = 1024 * 1024
LABEL_NAMES = {0: "relapse_free", 1: "distant_metastasis"}
MANAGED_OUTPUT_NAMES = {
    "representatives.npy",
    "representative_labels.npy",
    "representative_pool.npy",
    "representative_pool_labels.npy",
    "queries.npy",
    "query_labels.npy",
    "normalization_mean.npy",
    "normalization_scale.npy",
    "representatives.csv",
    "samples.csv",
    "features.csv",
    "dataset.json",
    "source_manifest.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Join the official GSE2034 expression and clinical tables, make "
            "a stratified held-out split, fit log2/z-score preprocessing only "
            "on the representative pool, and construct one centroid per class."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"validated raw directory (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"destination directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--representative-fraction",
        type=float,
        default=0.70,
        help=(
            "fraction of each outcome class used to construct centroids "
            "(default: 0.70)"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="stratified split random seed (default: 42)",
    )
    parser.add_argument(
        "--expected-samples",
        type=int,
        default=EXPECTED_SAMPLES,
        help=f"expected sample count (default: {EXPECTED_SAMPLES})",
    )
    parser.add_argument(
        "--expected-features",
        type=int,
        default=EXPECTED_FEATURES,
        help=f"expected probe-set count (default: {EXPECTED_FEATURES})",
    )
    parser.add_argument(
        "--expected-relapse-free",
        type=int,
        default=EXPECTED_RELAPSE_COUNTS[0],
        help=(
            "expected number of label-0 clinical rows "
            f"(default: {EXPECTED_RELAPSE_COUNTS[0]})"
        ),
    )
    parser.add_argument(
        "--expected-relapse",
        type=int,
        default=EXPECTED_RELAPSE_COUNTS[1],
        help=(
            "expected number of label-1 clinical rows "
            f"(default: {EXPECTED_RELAPSE_COUNTS[1]})"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace managed files from an earlier transformation",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.representative_fraction < 1.0:
        raise ValueError(
            "--representative-fraction must be strictly between zero and one"
        )
    if args.expected_samples < 2:
        raise ValueError("--expected-samples must be at least two")
    if args.expected_features < 1:
        raise ValueError("--expected-features must be positive")
    if args.expected_relapse_free < 2 or args.expected_relapse < 2:
        raise ValueError("each expected label count must be at least two")
    if (
        args.expected_relapse_free + args.expected_relapse
        != args.expected_samples
    ):
        raise ValueError("expected label counts must sum to --expected-samples")


def load_expression_matrix(
    path: Path,
    expected_samples: int,
    expected_features: int,
) -> tuple[tuple[str, ...], tuple[str, ...], np.ndarray]:
    """Load the GEO feature-major table directly into sample-major storage."""
    sample_accessions: tuple[str, ...] | None = None
    feature_ids: list[str] = []
    matrix: np.ndarray | None = None
    table_started = False
    table_ended = False

    try:
        with gzip.open(
            path,
            mode="rt",
            encoding="utf-8",
            newline="",
        ) as source:
            rows = csv.reader(source, delimiter="\t")
            for row_number, row in enumerate(rows, start=1):
                if not row:
                    continue
                if not table_started:
                    if row[0] != "!series_matrix_table_begin":
                        continue
                    table_started = True
                    try:
                        header = next(rows)
                    except StopIteration as error:
                        raise RuntimeError(
                            "series matrix ends before its table header"
                        ) from error
                    if not header or header[0] != "ID_REF":
                        raise RuntimeError(
                            "series matrix has an unexpected table header"
                        )
                    sample_accessions = tuple(header[1:])
                    if len(sample_accessions) != expected_samples:
                        raise RuntimeError(
                            f"series matrix has {len(sample_accessions)} samples; "
                            f"expected {expected_samples}"
                        )
                    if len(set(sample_accessions)) != len(sample_accessions):
                        raise RuntimeError(
                            "series matrix contains duplicate sample accessions"
                        )
                    matrix = np.empty(
                        (expected_samples, expected_features),
                        dtype="<f4",
                    )
                    continue

                if row[0] == "!series_matrix_table_end":
                    table_ended = True
                    break
                if sample_accessions is None or matrix is None:
                    raise RuntimeError("series matrix is missing its table header")
                if len(feature_ids) >= expected_features:
                    raise RuntimeError(
                        f"series matrix has more than {expected_features} features"
                    )
                if len(row) != expected_samples + 1:
                    raise RuntimeError(
                        f"matrix row {row_number} has {len(row) - 1} values; "
                        f"expected {expected_samples}"
                    )
                feature_id = row[0]
                if not feature_id:
                    raise RuntimeError(f"matrix row {row_number} has no feature ID")
                try:
                    values = np.asarray(row[1:], dtype="<f4")
                except ValueError as error:
                    raise RuntimeError(
                        f"matrix row {row_number} contains a non-numeric value"
                    ) from error
                if not np.isfinite(values).all():
                    raise RuntimeError(
                        f"matrix row {row_number} contains a non-finite value"
                    )
                matrix[:, len(feature_ids)] = values
                feature_ids.append(feature_id)
    except (gzip.BadGzipFile, UnicodeDecodeError, csv.Error) as error:
        raise RuntimeError(f"cannot parse {path.name}: {error}") from error

    if not table_started or not table_ended:
        raise RuntimeError("series matrix does not contain a complete data table")
    if sample_accessions is None or matrix is None:
        raise RuntimeError("series matrix contains no expression data")
    if len(feature_ids) != expected_features:
        raise RuntimeError(
            f"series matrix has {len(feature_ids)} features; "
            f"expected {expected_features}"
        )
    if len(set(feature_ids)) != len(feature_ids):
        raise RuntimeError("series matrix contains duplicate feature IDs")
    if np.min(matrix) < 0.0:
        raise RuntimeError("series matrix contains negative values before log2(x+1)")
    return sample_accessions, tuple(feature_ids), matrix


def stratified_split(
    labels: np.ndarray,
    representative_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    pool: list[int] = []
    queries: list[int] = []
    for label in sorted(int(value) for value in np.unique(labels)):
        indices = [
            int(index)
            for index in np.flatnonzero(labels == label)
        ]
        rng.shuffle(indices)
        pool_count = round(len(indices) * representative_fraction)
        pool_count = max(1, min(len(indices) - 1, pool_count))
        pool.extend(sorted(indices[:pool_count]))
        queries.extend(sorted(indices[pool_count:]))
    return (
        np.asarray(pool, dtype=np.int64),
        np.asarray(queries, dtype=np.int64),
    )


def normalize_from_pool(
    matrix: np.ndarray,
    pool_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Apply log2(x+1), then training-pool z-scores without leakage."""
    np.add(matrix, np.float32(1.0), out=matrix)
    np.log2(matrix, out=matrix)
    pool_values = matrix[pool_indices]
    mean = np.mean(pool_values, axis=0, dtype=np.float64).astype("<f4")
    scale = np.std(pool_values, axis=0, dtype=np.float64).astype("<f4")
    constant = scale == 0.0
    constant_count = int(np.count_nonzero(constant))
    scale[constant] = np.float32(1.0)
    np.subtract(matrix, mean, out=matrix)
    np.divide(matrix, scale, out=matrix)
    if not np.isfinite(matrix).all():
        raise RuntimeError("normalization produced non-finite values")
    return mean, scale, constant_count


def class_counts(labels: np.ndarray, indices: np.ndarray) -> dict[str, int]:
    counts = Counter(int(labels[index]) for index in indices)
    return {LABEL_NAMES[label]: counts[label] for label in sorted(LABEL_NAMES)}


def save_array(path: Path, values: np.ndarray, dtype: str) -> None:
    array = np.ascontiguousarray(values, dtype=dtype)
    np.save(path, array, allow_pickle=False)


def write_samples(
    path: Path,
    sample_accessions: tuple[str, ...],
    clinical_by_accession: dict[str, ClinicalSample],
    pool_indices: np.ndarray,
    query_indices: np.ndarray,
) -> None:
    pool_rows = {
        int(source_index): row
        for row, source_index in enumerate(pool_indices)
    }
    query_rows = {
        int(source_index): row
        for row, source_index in enumerate(query_indices)
    }
    fieldnames = (
        "source_column",
        "geo_accession",
        "patient_id",
        "role",
        "role_row",
        "label",
        "outcome",
        "follow_up_months",
        "lymph_node_status",
        "er_status",
        "brain_relapse",
    )
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for source_column, geo_accession in enumerate(sample_accessions):
            clinical = clinical_by_accession[geo_accession]
            if source_column in pool_rows:
                role = "representative_pool"
                role_row = pool_rows[source_column]
            else:
                role = "query"
                role_row = query_rows[source_column]
            writer.writerow(
                {
                    "source_column": source_column,
                    "geo_accession": geo_accession,
                    "patient_id": clinical.patient_id,
                    "role": role,
                    "role_row": role_row,
                    "label": clinical.relapse,
                    "outcome": LABEL_NAMES[clinical.relapse],
                    "follow_up_months": clinical.follow_up_months,
                    "lymph_node_status": clinical.lymph_node_status,
                    "er_status": clinical.er_status,
                    "brain_relapse": clinical.brain_relapse,
                }
            )


def write_features(path: Path, feature_ids: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("feature_index", "probe_set_id"))
        writer.writerows(enumerate(feature_ids))


def write_representatives(
    path: Path,
    labels: np.ndarray,
    pool_indices: np.ndarray,
) -> None:
    pool_labels = labels[pool_indices]
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(
            (
                "representative_row",
                "label",
                "outcome",
                "construction",
                "source_sample_count",
            )
        )
        for row, label in enumerate(sorted(LABEL_NAMES)):
            writer.writerow(
                (
                    row,
                    label,
                    LABEL_NAMES[label],
                    "arithmetic_mean_centroid",
                    int(np.count_nonzero(pool_labels == label)),
                )
            )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(HASH_BLOCK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, object]:
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def read_source_manifest(input_dir: Path) -> dict[str, object] | None:
    path = input_dir / "manifest.json"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as source:
        manifest = json.load(source)
    if not isinstance(manifest, dict):
        raise RuntimeError("source manifest must contain a JSON object")
    return manifest


def build_dataset_metadata(
    args: argparse.Namespace,
    staging_dir: Path,
    labels: np.ndarray,
    pool_indices: np.ndarray,
    query_indices: np.ndarray,
    constant_feature_count: int,
    source_manifest_available: bool,
) -> dict[str, object]:
    dimensions = args.expected_features
    generated_files = {
        path.name: file_record(path)
        for path in sorted(staging_dir.iterdir())
        if path.is_file() and path.name != "dataset.json"
    }
    return {
        "dataset_name": "GSE2034 stratified relapse benchmark",
        "format_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "matrices": {
            "representatives": {
                "file": "representatives.npy",
                "shape": [2, dimensions],
                "purpose": "one training-only centroid per outcome class",
            },
            "representative_pool": {
                "file": "representative_pool.npy",
                "shape": [len(pool_indices), dimensions],
                "purpose": "samples used to fit preprocessing and centroids",
            },
            "queries": {
                "file": "queries.npy",
                "shape": [len(query_indices), dimensions],
                "purpose": "held-out samples used only for evaluation",
            },
            "common_layout": {
                "orientation": "rows_by_features",
                "feature_order": "GPL96 probe-set order in the GEO matrix",
                "dtype": "float32",
                "byte_order": "little-endian",
                "memory_order": "C",
            },
        },
        "labels": {
            "representative_file": "representative_labels.npy",
            "representative_pool_file": "representative_pool_labels.npy",
            "query_file": "query_labels.npy",
            "dtype": "uint16",
            "mapping": {str(label): name for label, name in LABEL_NAMES.items()},
            "source_column": "relapse (1=True)",
        },
        "split": {
            "policy": "stratified random holdout by clinical outcome",
            "seed": args.seed,
            "requested_representative_fraction": args.representative_fraction,
            "representative_pool_count": len(pool_indices),
            "query_count": len(query_indices),
            "representative_pool_counts_by_class": class_counts(
                labels,
                pool_indices,
            ),
            "query_counts_by_class": class_counts(labels, query_indices),
            "leakage_control": (
                "split precedes preprocessing; normalization statistics and "
                "class centroids use only representative-pool rows"
            ),
        },
        "normalization": {
            "method": "log2(x+1), then per-probe population z-score",
            "fit_rows": "representative_pool only",
            "mean_file": "normalization_mean.npy",
            "scale_file": "normalization_scale.npy",
            "zero_variance_probe_sets": constant_feature_count,
            "zero_variance_policy": "retain the probe set and use scale 1",
        },
        "representative_construction": {
            "method": "arithmetic_mean_centroid",
            "count_per_class": 1,
            "metadata_file": "representatives.csv",
        },
        "features": {
            "file": "features.csv",
            "count": dimensions,
            "definition": "Affymetrix HG-U133A (GPL96) probe sets",
        },
        "samples": {
            "file": "samples.csv",
            "count": args.expected_samples,
            "alignment": "clinical rows joined by GEO accession",
        },
        "source_manifest_available": source_manifest_available,
        "transformations": [
            "join patient outcomes to expression columns by GSM accession",
            "make a seeded stratified 70/30-style holdout split",
            "apply log2(x+1) to processed GEO expression values",
            "fit per-probe means and scales on representative-pool rows only",
            "apply those training-only z-scores to pool and query rows",
            "construct one arithmetic-mean centroid per outcome class",
        ],
        "numpy_version": np.__version__,
        "generated_files": generated_files,
    }


def install_staged_outputs(
    staging_dir: Path,
    output_dir: Path,
    force: bool,
) -> None:
    staged_names = {
        path.name for path in staging_dir.iterdir() if path.is_file()
    }
    if force:
        for stale_name in MANAGED_OUTPUT_NAMES - staged_names:
            stale_path = output_dir / stale_name
            if stale_path.is_file():
                stale_path.unlink()
    for name in sorted(staged_names):
        (staging_dir / name).replace(output_dir / name)


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
        input_dir = args.input_dir.expanduser().resolve()
        requested_output_dir = args.output_dir.expanduser()
        if requested_output_dir.is_symlink():
            raise RuntimeError(
                f"refusing to write through symbolic-link output: "
                f"{requested_output_dir}"
            )
        output_dir = requested_output_dir.resolve()
        if not input_dir.is_dir():
            raise RuntimeError(f"input directory does not exist: {input_dir}")
        matrix_path = input_dir / MATRIX_NAME
        clinical_path = input_dir / CLINICAL_NAME
        if not matrix_path.is_file():
            raise RuntimeError(f"missing expression matrix: {matrix_path}")
        if not clinical_path.is_file():
            raise RuntimeError(f"missing clinical table: {clinical_path}")

        print("Loading the 22,283-dimensional GSE2034 matrix...", flush=True)
        sample_accessions, feature_ids, matrix = load_expression_matrix(
            matrix_path,
            args.expected_samples,
            args.expected_features,
        )
        expected_counts = {
            0: args.expected_relapse_free,
            1: args.expected_relapse,
        }
        validate_clinical_table(
            clinical_path,
            sample_accessions,
            expected_counts,
        )
        clinical_samples = read_clinical_samples(clinical_path)
        clinical_by_accession = {
            sample.geo_accession: sample for sample in clinical_samples
        }
        labels = np.asarray(
            [
                clinical_by_accession[accession].relapse
                for accession in sample_accessions
            ],
            dtype="<u2",
        )

        pool_indices, query_indices = stratified_split(
            labels,
            args.representative_fraction,
            args.seed,
        )
        print(
            f"Fitting preprocessing on {len(pool_indices)} representative-pool "
            f"samples; holding out {len(query_indices)} queries...",
            flush=True,
        )
        mean, scale, constant_feature_count = normalize_from_pool(
            matrix,
            pool_indices,
        )
        representative_pool = np.ascontiguousarray(
            matrix[pool_indices],
            dtype="<f4",
        )
        pool_labels = np.ascontiguousarray(labels[pool_indices], dtype="<u2")
        queries = np.ascontiguousarray(matrix[query_indices], dtype="<f4")
        query_labels = np.ascontiguousarray(
            labels[query_indices],
            dtype="<u2",
        )
        representatives = np.stack(
            [
                representative_pool[pool_labels == label].mean(
                    axis=0,
                    dtype=np.float64,
                )
                for label in sorted(LABEL_NAMES)
            ]
        ).astype("<f4")
        representative_labels = np.asarray(sorted(LABEL_NAMES), dtype="<u2")

        output_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(
            name
            for name in MANAGED_OUTPUT_NAMES
            if (output_dir / name).exists()
        )
        if existing and not args.force:
            print(
                f"Refusing to overwrite existing files in {output_dir}: "
                f"{', '.join(existing)}\nRun again with --force to replace them.",
                file=sys.stderr,
            )
            return 1

        with tempfile.TemporaryDirectory(
            prefix=".gse2034-transform-",
            dir=output_dir,
        ) as temporary_directory:
            staging_dir = Path(temporary_directory)
            save_array(
                staging_dir / "representatives.npy",
                representatives,
                "<f4",
            )
            save_array(
                staging_dir / "representative_labels.npy",
                representative_labels,
                "<u2",
            )
            save_array(
                staging_dir / "representative_pool.npy",
                representative_pool,
                "<f4",
            )
            save_array(
                staging_dir / "representative_pool_labels.npy",
                pool_labels,
                "<u2",
            )
            save_array(staging_dir / "queries.npy", queries, "<f4")
            save_array(
                staging_dir / "query_labels.npy",
                query_labels,
                "<u2",
            )
            save_array(staging_dir / "normalization_mean.npy", mean, "<f4")
            save_array(staging_dir / "normalization_scale.npy", scale, "<f4")
            write_samples(
                staging_dir / "samples.csv",
                sample_accessions,
                clinical_by_accession,
                pool_indices,
                query_indices,
            )
            write_features(staging_dir / "features.csv", feature_ids)
            write_representatives(
                staging_dir / "representatives.csv",
                labels,
                pool_indices,
            )

            source_manifest = read_source_manifest(input_dir)
            if source_manifest is None:
                with (staging_dir / "source_manifest.json").open(
                    "w",
                    encoding="utf-8",
                ) as output:
                    json.dump({"available": False}, output, indent=2)
                    output.write("\n")
            else:
                shutil.copyfile(
                    input_dir / "manifest.json",
                    staging_dir / "source_manifest.json",
                )

            metadata = build_dataset_metadata(
                args,
                staging_dir,
                labels,
                pool_indices,
                query_indices,
                constant_feature_count,
                source_manifest is not None,
            )
            with (staging_dir / "dataset.json").open(
                "w",
                encoding="utf-8",
            ) as output:
                json.dump(metadata, output, indent=2, sort_keys=True)
                output.write("\n")
            install_staged_outputs(staging_dir, output_dir, args.force)
    except (
        OSError,
        RuntimeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(
        f"Saved 2 representatives and {len(query_indices)} queries with "
        f"{args.expected_features:,} dimensions to {output_dir}.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
