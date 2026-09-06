#!/usr/bin/env python3
"""Transform TCGA Kidney expression matrices into sample-major NPY data."""

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
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path
from typing import TextIO

try:
    import numpy as np
except ModuleNotFoundError as error:
    print(
        "transform_tcga_kidney.py requires NumPy; install it with "
        "'python3 -m pip install numpy'.",
        file=sys.stderr,
    )
    raise SystemExit(2) from error


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "tcga"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "tcga_kidney_v1"
DEFAULT_FEATURE_COUNT = 60_660
HASH_BLOCK_SIZE = 1024 * 1024
TRANSPOSE_BLOCK_ROWS = 32
CENTROID_BLOCK_FEATURES = 4_096


@dataclass(frozen=True)
class Cohort:
    project_id: str
    subtype: str
    label: int

    @property
    def matrix_stem(self) -> str:
        return f"{self.project_id}.star_fpkm.tsv"


COHORTS = (
    Cohort("TCGA-KICH", "Kidney Chromophobe", 0),
    Cohort("TCGA-KIRC", "Kidney Renal Clear Cell Carcinoma", 1),
    Cohort("TCGA-KIRP", "Kidney Renal Papillary Cell Carcinoma", 2),
)

SAMPLE_TYPE_NAMES = {
    "01": "Primary Solid Tumor",
    "02": "Recurrent Solid Tumor",
    "03": "Primary Blood Derived Cancer - Peripheral Blood",
    "04": "Recurrent Blood Derived Cancer - Bone Marrow",
    "05": "Additional - New Primary",
    "06": "Metastatic",
    "07": "Additional Metastatic",
    "08": "Human Tumor Original Cells",
    "09": "Primary Blood Derived Cancer - Bone Marrow",
    "10": "Blood Derived Normal",
    "11": "Solid Tissue Normal",
    "12": "Buccal Cell Normal",
    "13": "EBV Immortalized Normal",
    "14": "Bone Marrow Normal",
    "20": "Control Analyte",
    "40": "Recurrent Blood Derived Cancer - Peripheral Blood",
    "50": "Cell Lines",
    "60": "Primary Xenograft Tissue",
    "61": "Cell Line Derived Xenograft Tissue",
}

CURRENT_OUTPUT_NAMES = {
    "representative_pool.npy",
    "representative_pool_labels.npy",
    "representatives.npy",
    "representative_labels.npy",
    "representatives.csv",
    "queries.npy",
    "query_labels.npy",
    "samples.csv",
    "features.csv",
    "dataset.json",
    "source_manifest.json",
}
LEGACY_OUTPUT_NAMES = {"X.npy", "y.npy"}
MANAGED_OUTPUT_NAMES = CURRENT_OUTPUT_NAMES | LEGACY_OUTPUT_NAMES


@dataclass(frozen=True)
class Sample:
    row_index: int
    sample_id: str
    participant_id: str
    project_id: str
    cancer_subtype: str
    label: int
    sample_type_code: str
    sample_type: str


@dataclass(frozen=True)
class CohortMatrix:
    cohort: Cohort
    path: Path
    sample_ids: tuple[str, ...]
    selected_columns: np.ndarray
    output_start: int
    output_end: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert the three TCGA Kidney STAR-FPKM matrices from "
            "genes-by-samples TSV into representative and query matrices."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"directory containing source matrices (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"destination directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--sample-types",
        nargs="+",
        default=("01",),
        metavar="CODE",
        help=(
            "TCGA sample-type codes to retain (default: 01, primary solid "
            "tumor). Labels always identify the source cancer project, so "
            "including normal samples changes the prediction task."
        ),
    )
    parser.add_argument(
        "--representative-fraction",
        type=float,
        default=0.70,
        help=(
            "fraction of participants used to construct the class "
            "representatives (default: 0.70); the remainder are queries"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="participant-role assignment random seed (default: 42)",
    )
    parser.add_argument(
        "--expected-features",
        type=int,
        default=DEFAULT_FEATURE_COUNT,
        help=(
            "expected number of genes; use 0 to infer it from the first "
            f"matrix (default: {DEFAULT_FEATURE_COUNT})"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace managed files from an earlier transformation",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[str, ...]:
    if not 0.0 < args.representative_fraction < 1.0:
        raise ValueError(
            "--representative-fraction must be strictly between zero and one"
        )
    if args.expected_features < 0:
        raise ValueError("--expected-features cannot be negative")

    sample_types = tuple(dict.fromkeys(args.sample_types))
    if not sample_types:
        raise ValueError("at least one sample type must be selected")
    for code in sample_types:
        if len(code) != 2 or not code.isdigit():
            raise ValueError(
                f"invalid TCGA sample-type code {code!r}; expected two digits"
            )

    return sample_types


def open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8", newline="")
    return path.open(mode="r", encoding="utf-8", newline="")


def resolve_matrix_path(input_dir: Path, cohort: Cohort) -> Path:
    candidates = (
        input_dir / cohort.matrix_stem,
        input_dir / f"{cohort.matrix_stem}.gz",
    )
    existing = [path for path in candidates if path.is_file()]
    if not existing:
        names = " or ".join(path.name for path in candidates)
        raise RuntimeError(f"missing {cohort.project_id} matrix: expected {names}")
    if len(existing) > 1:
        raise RuntimeError(
            f"both compressed and uncompressed matrices exist for "
            f"{cohort.project_id}; retain only one"
        )
    return existing[0]


def parse_sample_id(sample_id: str) -> tuple[str, str]:
    fields = sample_id.split("-")
    if (
        len(fields) < 4
        or fields[0] != "TCGA"
        or len(fields[1]) != 2
        or len(fields[2]) != 4
        or len(fields[3]) < 3
    ):
        raise RuntimeError(f"unexpected TCGA sample ID: {sample_id!r}")

    sample_type_code = fields[3][:2]
    if not sample_type_code.isdigit():
        raise RuntimeError(
            f"sample ID does not contain a numeric sample type: {sample_id!r}"
        )

    participant_id = "-".join(fields[:3])
    return participant_id, sample_type_code


def parse_header(path: Path) -> tuple[str, ...]:
    with open_text(path) as source:
        line = source.readline()
    if not line:
        raise RuntimeError(f"{path.name} is empty")

    fields = line.rstrip("\r\n").split("\t")
    if len(fields) < 2 or fields[0] != "Ensembl_ID":
        raise RuntimeError(f"{path.name} has an unexpected header")

    sample_ids = tuple(fields[1:])
    if len(set(sample_ids)) != len(sample_ids):
        raise RuntimeError(f"{path.name} contains duplicate sample IDs")
    return sample_ids


def collect_samples(
    input_dir: Path,
    retained_sample_types: tuple[str, ...],
) -> tuple[list[CohortMatrix], list[Sample]]:
    retained = set(retained_sample_types)
    seen_sample_ids: set[str] = set()
    cohort_matrices: list[CohortMatrix] = []
    samples: list[Sample] = []

    for cohort in COHORTS:
        path = resolve_matrix_path(input_dir, cohort)
        sample_ids = parse_header(path)

        duplicates = seen_sample_ids.intersection(sample_ids)
        if duplicates:
            raise RuntimeError(
                f"duplicate sample ID across cohorts: {min(duplicates)}"
            )
        seen_sample_ids.update(sample_ids)

        selected_columns: list[int] = []
        output_start = len(samples)

        for source_column, sample_id in enumerate(sample_ids):
            participant_id, sample_type_code = parse_sample_id(sample_id)
            if sample_type_code not in retained:
                continue

            selected_columns.append(source_column)
            samples.append(
                Sample(
                    row_index=len(samples),
                    sample_id=sample_id,
                    participant_id=participant_id,
                    project_id=cohort.project_id,
                    cancer_subtype=cohort.subtype,
                    label=cohort.label,
                    sample_type_code=sample_type_code,
                    sample_type=SAMPLE_TYPE_NAMES.get(
                        sample_type_code,
                        "Unknown TCGA sample type",
                    ),
                )
            )

        if not selected_columns:
            selected_text = ", ".join(retained_sample_types)
            raise RuntimeError(
                f"{path.name} has no samples with requested types: {selected_text}"
            )

        cohort_matrices.append(
            CohortMatrix(
                cohort=cohort,
                path=path,
                sample_ids=sample_ids,
                selected_columns=np.asarray(selected_columns, dtype=np.intp),
                output_start=output_start,
                output_end=len(samples),
            )
        )

    return cohort_matrices, samples


def assign_participant_roles(
    samples: list[Sample],
    representative_fraction: float,
    seed: int,
) -> list[str]:
    participant_labels: dict[str, int] = {}
    participants_by_label: dict[int, set[str]] = {
        cohort.label: set() for cohort in COHORTS
    }
    for sample in samples:
        previous_label = participant_labels.setdefault(
            sample.participant_id,
            sample.label,
        )
        if previous_label != sample.label:
            raise RuntimeError(
                f"participant {sample.participant_id} appears under multiple labels"
            )
        participants_by_label[sample.label].add(sample.participant_id)

    random_engine = random.Random(seed)
    participant_roles: dict[str, str] = {}

    for label in sorted(participants_by_label):
        participants = sorted(participants_by_label[label])
        if len(participants) < 2:
            raise RuntimeError(
                f"label {label} has fewer than two participants; cannot "
                "construct a representative and retain an independent query"
            )
        random_engine.shuffle(participants)
        representative_count = round(
            len(participants) * representative_fraction
        )
        representative_count = max(
            1,
            min(len(participants) - 1, representative_count),
        )

        for participant in participants[:representative_count]:
            participant_roles[participant] = "representative_pool"
        for participant in participants[representative_count:]:
            participant_roles[participant] = "query"

    return [participant_roles[sample.participant_id] for sample in samples]


def count_feature_rows(path: Path) -> int:
    with open_text(path) as source:
        if not source.readline():
            raise RuntimeError(f"{path.name} is empty")
        return sum(1 for _ in source)


def parse_feature_line(
    line: str,
    path: Path,
    expected_values: int,
) -> tuple[str, np.ndarray]:
    feature_id, separator, value_text = line.partition("\t")
    if not separator or not feature_id:
        raise RuntimeError(f"{path.name} contains a malformed feature row")

    values = np.fromstring(
        value_text.rstrip("\r\n"),
        dtype=np.dtype("<f4"),
        sep="\t",
    )
    if values.size != expected_values:
        raise RuntimeError(
            f"{path.name} feature {feature_id!r} contains {values.size} values; "
            f"expected {expected_values}"
        )
    if not np.isfinite(values).all():
        raise RuntimeError(
            f"{path.name} feature {feature_id!r} contains a non-finite value"
        )
    return feature_id, values


def indices_for_role(roles: list[str], role: str) -> np.ndarray:
    return np.asarray(
        [index for index, assigned_role in enumerate(roles) if assigned_role == role],
        dtype=np.intp,
    )


def write_sample_major_subset(
    feature_major: np.ndarray,
    sample_indices: np.ndarray,
    path: Path,
) -> None:
    output = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.dtype("<f4"),
        shape=(len(sample_indices), feature_major.shape[0]),
        fortran_order=False,
    )
    try:
        for start in range(0, len(sample_indices), TRANSPOSE_BLOCK_ROWS):
            end = min(start + TRANSPOSE_BLOCK_ROWS, len(sample_indices))
            selected = sample_indices[start:end]
            output[start:end, :] = feature_major[:, selected].T
        output.flush()
    finally:
        del output


def write_centroid_representatives(
    feature_major: np.ndarray,
    samples: list[Sample],
    representative_pool_indices: np.ndarray,
    path: Path,
) -> None:
    representatives = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.dtype("<f4"),
        shape=(len(COHORTS), feature_major.shape[0]),
        fortran_order=False,
    )
    try:
        for cohort in COHORTS:
            cohort_indices = np.asarray(
                [
                    index
                    for index in representative_pool_indices
                    if samples[int(index)].label == cohort.label
                ],
                dtype=np.intp,
            )
            if not len(cohort_indices):
                raise RuntimeError(
                    f"representative pool is empty for {cohort.project_id}"
                )
            for start in range(
                0, feature_major.shape[0], CENTROID_BLOCK_FEATURES
            ):
                end = min(
                    start + CENTROID_BLOCK_FEATURES,
                    feature_major.shape[0],
                )
                values = feature_major[start:end, cohort_indices]
                representatives[cohort.label, start:end] = np.mean(
                    values,
                    axis=1,
                    dtype=np.float64,
                )
        representatives.flush()
    finally:
        del representatives


def convert_expression_matrix(
    cohort_matrices: list[CohortMatrix],
    samples: list[Sample],
    roles: list[str],
    feature_count: int,
    staging_dir: Path,
) -> list[str]:
    sample_count = len(samples)
    temporary_path = staging_dir / "features_by_samples.npy"
    feature_major = np.lib.format.open_memmap(
        temporary_path,
        mode="w+",
        dtype=np.dtype("<f4"),
        shape=(feature_count, sample_count),
        fortran_order=False,
    )
    feature_ids: list[str] = []

    try:
        with ExitStack() as stack:
            streams = [
                stack.enter_context(open_text(matrix.path))
                for matrix in cohort_matrices
            ]
            for stream, matrix in zip(streams, cohort_matrices, strict=True):
                header = stream.readline()
                expected_header = "\t".join(("Ensembl_ID", *matrix.sample_ids))
                if header.rstrip("\r\n") != expected_header:
                    raise RuntimeError(
                        f"{matrix.path.name} changed after its header was read"
                    )

            for feature_index, lines in enumerate(zip_longest(*streams)):
                if feature_index >= feature_count:
                    raise RuntimeError(
                        f"source matrices contain more than {feature_count} features"
                    )
                if any(line is None for line in lines):
                    raise RuntimeError(
                        "cohort matrices have different feature counts"
                    )

                shared_feature_id: str | None = None
                for line, matrix in zip(lines, cohort_matrices, strict=True):
                    assert line is not None
                    feature_id, values = parse_feature_line(
                        line,
                        matrix.path,
                        len(matrix.sample_ids),
                    )
                    if shared_feature_id is None:
                        shared_feature_id = feature_id
                    elif feature_id != shared_feature_id:
                        raise RuntimeError(
                            "cohort matrices do not have matching Ensembl gene order: "
                            f"expected {shared_feature_id!r}, found {feature_id!r} "
                            f"in {matrix.path.name}"
                        )

                    feature_major[
                        feature_index,
                        matrix.output_start : matrix.output_end,
                    ] = values[matrix.selected_columns]

                assert shared_feature_id is not None
                feature_ids.append(shared_feature_id)
                if (feature_index + 1) % 5_000 == 0:
                    print(
                        f"  converted {feature_index + 1:,} / "
                        f"{feature_count:,} features",
                        flush=True,
                    )

        if len(feature_ids) != feature_count:
            raise RuntimeError(
                f"source matrices contain {len(feature_ids)} features; "
                f"expected {feature_count}"
            )

        feature_major.flush()
        representative_pool_indices = indices_for_role(
            roles, "representative_pool"
        )
        query_indices = indices_for_role(roles, "query")
        print("Writing sample-major representative and query data...", flush=True)
        write_sample_major_subset(
            feature_major,
            representative_pool_indices,
            staging_dir / "representative_pool.npy",
        )
        write_sample_major_subset(
            feature_major,
            query_indices,
            staging_dir / "queries.npy",
        )
        write_centroid_representatives(
            feature_major,
            samples,
            representative_pool_indices,
            staging_dir / "representatives.npy",
        )
    finally:
        del feature_major
        if temporary_path.exists():
            temporary_path.unlink()

    return feature_ids


def write_labels(
    path: Path,
    samples: list[Sample],
    sample_indices: np.ndarray,
) -> None:
    labels = np.asarray(
        [samples[int(index)].label for index in sample_indices],
        dtype=np.dtype("<u2"),
    )
    np.save(path, labels, allow_pickle=False)


def write_representative_labels(path: Path) -> None:
    labels = np.asarray(
        [cohort.label for cohort in COHORTS],
        dtype=np.dtype("<u2"),
    )
    np.save(path, labels, allow_pickle=False)


def write_samples(
    path: Path,
    samples: list[Sample],
    roles: list[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        fieldnames = (
            "sample_index",
            "role",
            "role_row_index",
            "sample_id",
            "participant_id",
            "project_id",
            "cancer_subtype",
            "label",
            "sample_type_code",
            "sample_type",
        )
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        next_role_row = {"representative_pool": 0, "query": 0}
        for sample, role in zip(samples, roles, strict=True):
            role_row_index = next_role_row[role]
            next_role_row[role] += 1
            writer.writerow(
                {
                    "sample_index": sample.row_index,
                    "role": role,
                    "role_row_index": role_row_index,
                    "sample_id": sample.sample_id,
                    "participant_id": sample.participant_id,
                    "project_id": sample.project_id,
                    "cancer_subtype": sample.cancer_subtype,
                    "label": sample.label,
                    "sample_type_code": sample.sample_type_code,
                    "sample_type": sample.sample_type,
                }
            )


def write_representatives(
    path: Path,
    samples: list[Sample],
    roles: list[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        fieldnames = (
            "row_index",
            "project_id",
            "cancer_subtype",
            "label",
            "construction",
            "pool_sample_count",
            "pool_participant_count",
        )
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for cohort in COHORTS:
            pool_samples = [
                sample
                for sample, role in zip(samples, roles, strict=True)
                if role == "representative_pool" and sample.label == cohort.label
            ]
            writer.writerow(
                {
                    "row_index": cohort.label,
                    "project_id": cohort.project_id,
                    "cancer_subtype": cohort.subtype,
                    "label": cohort.label,
                    "construction": "arithmetic_mean_centroid",
                    "pool_sample_count": len(pool_samples),
                    "pool_participant_count": len(
                        {sample.participant_id for sample in pool_samples}
                    ),
                }
            )


def write_features(path: Path, feature_ids: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("column_index", "ensembl_id"))
        writer.writerows(enumerate(feature_ids))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(HASH_BLOCK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, object]:
    return {
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def count_by(values: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return dict(sorted(result.items()))


def build_dataset_metadata(
    cohort_matrices: list[CohortMatrix],
    samples: list[Sample],
    roles: list[str],
    retained_sample_types: tuple[str, ...],
    feature_count: int,
    args: argparse.Namespace,
    staging_dir: Path,
) -> dict[str, object]:
    generated_files = {
        name: file_record(staging_dir / name)
        for name in sorted(CURRENT_OUTPUT_NAMES - {"dataset.json"})
        if (staging_dir / name).is_file()
    }
    source_files = {
        matrix.path.name: {
            **file_record(matrix.path),
            "path": str(matrix.path),
            "source_sample_count": len(matrix.sample_ids),
            "retained_sample_count": matrix.output_end - matrix.output_start,
        }
        for matrix in cohort_matrices
    }

    participant_role_pairs = {
        (sample.participant_id, role)
        for sample, role in zip(samples, roles, strict=True)
    }
    participant_roles = [role for _, role in participant_role_pairs]
    representative_pool_count = roles.count("representative_pool")
    query_count = roles.count("query")

    return {
        "dataset_name": "TCGA Kidney Cancers",
        "format_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "matrices": {
            "representative_pool": {
                "file": "representative_pool.npy",
                "shape": [representative_pool_count, feature_count],
                "purpose": "samples used to construct representatives",
            },
            "representatives": {
                "file": "representatives.npy",
                "shape": [len(COHORTS), feature_count],
                "purpose": "one arithmetic-mean centroid per cancer subtype",
            },
            "queries": {
                "file": "queries.npy",
                "shape": [query_count, feature_count],
                "purpose": "held-out evaluation queries",
            },
            "common_layout": {
                "orientation": "rows_by_features",
                "dtype": "float32",
                "byte_order": "little-endian",
                "memory_order": "C",
                "value_unit": "log2(FPKM+1)",
            },
        },
        "labels": {
            "representative_file": "representative_labels.npy",
            "representative_pool_file": "representative_pool_labels.npy",
            "query_file": "query_labels.npy",
            "dtype": "uint16",
            "mapping": {
                str(cohort.label): cohort.project_id for cohort in COHORTS
            },
        },
        "representative_construction": {
            "metadata_file": "representatives.csv",
            "method": "arithmetic_mean_centroid",
            "source": "representative_pool.npy",
            "count_per_project": 1,
        },
        "features": {
            "file": "features.csv",
            "identifier": "versioned Ensembl gene ID",
            "count": feature_count,
        },
        "samples": {
            "file": "samples.csv",
            "count": len(samples),
            "counts_by_project": count_by(
                [sample.project_id for sample in samples]
            ),
            "counts_by_sample_type": count_by(
                [sample.sample_type_code for sample in samples]
            ),
            "counts_by_role": count_by(roles),
        },
        "sample_filter": {
            "retained_sample_type_codes": list(retained_sample_types),
            "definitions": {
                code: SAMPLE_TYPE_NAMES.get(code, "Unknown TCGA sample type")
                for code in retained_sample_types
            },
        },
        "role_assignment": {
            "unit": "participant_id",
            "stratified_by": "project_id",
            "seed": args.seed,
            "requested_fractions": {
                "representative_pool": args.representative_fraction,
                "query": 1.0 - args.representative_fraction,
            },
            "participant_counts": count_by(participant_roles),
        },
        "transformations": [
            "filter samples by TCGA sample-type code",
            "transpose source matrices from features_by_samples to samples_by_features",
            "convert finite expression values to float32",
            "construct one arithmetic-mean centroid per project from the "
            "representative pool",
        ],
        "normalization": "none beyond source log2(FPKM+1) values",
        "numpy_version": np.__version__,
        "source_files": source_files,
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
        retained_sample_types = validate_args(args)
        input_dir = args.input_dir.expanduser().resolve()
        output_dir = args.output_dir.expanduser().resolve()
        if not input_dir.is_dir():
            raise RuntimeError(f"input directory does not exist: {input_dir}")

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

        cohort_matrices, samples = collect_samples(
            input_dir,
            retained_sample_types,
        )
        feature_count = args.expected_features
        if feature_count == 0:
            feature_count = count_feature_rows(cohort_matrices[0].path)

        roles = assign_participant_roles(
            samples,
            args.representative_fraction,
            args.seed,
        )
        representative_pool_indices = indices_for_role(
            roles, "representative_pool"
        )
        query_indices = indices_for_role(roles, "query")

        print(
            f"Retained {len(samples):,} samples: "
            f"{len(representative_pool_indices):,} in the representative "
            f"pool and {len(query_indices):,} queries; expect "
            f"{feature_count:,} features.",
            flush=True,
        )

        with tempfile.TemporaryDirectory(
            prefix=".tcga-transform-",
            dir=output_dir,
        ) as temporary_directory:
            staging_dir = Path(temporary_directory)
            feature_ids = convert_expression_matrix(
                cohort_matrices,
                samples,
                roles,
                feature_count,
                staging_dir,
            )
            write_labels(
                staging_dir / "representative_pool_labels.npy",
                samples,
                representative_pool_indices,
            )
            write_representative_labels(
                staging_dir / "representative_labels.npy"
            )
            write_labels(
                staging_dir / "query_labels.npy",
                samples,
                query_indices,
            )
            write_samples(staging_dir / "samples.csv", samples, roles)
            write_representatives(
                staging_dir / "representatives.csv", samples, roles
            )
            write_features(staging_dir / "features.csv", feature_ids)

            source_manifest = input_dir / "manifest.json"
            if source_manifest.is_file():
                shutil.copyfile(
                    source_manifest,
                    staging_dir / "source_manifest.json",
                )

            metadata = build_dataset_metadata(
                cohort_matrices,
                samples,
                roles,
                retained_sample_types,
                feature_count,
                args,
                staging_dir,
            )
            with (staging_dir / "dataset.json").open(
                "w", encoding="utf-8"
            ) as output:
                json.dump(metadata, output, indent=2, sort_keys=True)
                output.write("\n")

            install_staged_outputs(
                staging_dir,
                output_dir,
                args.force,
            )

    except (OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(
        f"Saved {len(COHORTS)} representatives and "
        f"{len(query_indices):,} queries with {feature_count:,} features to "
        f"{output_dir}.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
