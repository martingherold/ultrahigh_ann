#!/usr/bin/env python3
"""Transform the 33 TCGA Pan-Cancer matrices for representative search."""

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

from download_tcga_pancancer import COHORTS, Cohort

try:
    import numpy as np
except ModuleNotFoundError as error:
    print(
        "transform_tcga_pancancer.py requires NumPy; install it with "
        "'python3 -m pip install numpy'.",
        file=sys.stderr,
    )
    raise SystemExit(2) from error


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "tcga_pancancer"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "processed" / "tcga_pancancer_v1"
)
DEFAULT_FEATURE_COUNT = 60_660
HASH_BLOCK_SIZE = 1024 * 1024
TRANSPOSE_BLOCK_ROWS = 32

# TCGA uses a different primary-disease sample type for LAML. Every other
# downloaded Pan-Cancer project is a solid-tumor cohort.
DEFAULT_SOLID_SAMPLE_TYPES = ("01",)
DEFAULT_SAMPLE_TYPES_BY_PROJECT = {"TCGA-LAML": ("03",)}

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

REQUIRED_OUTPUT_NAMES = {
    "representatives.npy",
    "representatives.csv",
    "queries.npy",
    "query_labels.npy",
    "samples.csv",
    "features.csv",
    "dataset.json",
    "source_manifest.json",
}
OPTIONAL_OUTPUT_NAMES = {
    "representative_pool.npy",
    "representative_pool_labels.npy",
}
MANAGED_OUTPUT_NAMES = REQUIRED_OUTPUT_NAMES | OPTIONAL_OUTPUT_NAMES


@dataclass(frozen=True)
class Sample:
    row_index: int
    sample_id: str
    participant_id: str
    project_id: str
    cancer_type: str
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


@dataclass(frozen=True)
class CohortRoleLayout:
    pool_local_indices: np.ndarray
    pool_output_start: int
    pool_output_end: int
    query_local_indices: np.ndarray
    query_output_start: int
    query_output_end: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert the 33 TCGA Pan-Cancer STAR-FPKM matrices into one "
            "centroid per project and participant-disjoint held-out queries."
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
        "--representative-fraction",
        type=float,
        default=0.70,
        help=(
            "fraction of participants used to construct each project centroid "
            "(default: 0.70); the remainder are queries"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="participant-role assignment random seed (default: 42)",
    )
    parser.add_argument(
        "--sample-types",
        action="append",
        default=[],
        metavar="PROJECT=CODE[,CODE...]",
        help=(
            "override retained sample types for one project; repeat as needed "
            "(defaults: 01 for solid tumors and 03 for TCGA-LAML)"
        ),
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
        "--write-representative-pool",
        action="store_true",
        help=(
            "also materialize representative_pool.npy and its labels; this "
            "requires roughly 1.6 GiB for the default split"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace managed files from an earlier transformation",
    )
    return parser.parse_args()


def validate_sample_type_code(code: str) -> None:
    if len(code) != 2 or not code.isdigit():
        raise ValueError(
            f"invalid TCGA sample-type code {code!r}; expected two digits"
        )


def build_sample_type_policy(
    specifications: list[str],
) -> dict[str, tuple[str, ...]]:
    known_projects = {cohort.project_id for cohort in COHORTS}
    policy = {
        cohort.project_id: DEFAULT_SAMPLE_TYPES_BY_PROJECT.get(
            cohort.project_id,
            DEFAULT_SOLID_SAMPLE_TYPES,
        )
        for cohort in COHORTS
    }
    overridden: set[str] = set()

    for specification in specifications:
        project_id, separator, codes_text = specification.partition("=")
        if not separator or not project_id or not codes_text:
            raise ValueError(
                "--sample-types expects PROJECT=CODE[,CODE...]"
            )
        if project_id not in known_projects:
            raise ValueError(f"unknown TCGA project in --sample-types: {project_id}")
        if project_id in overridden:
            raise ValueError(
                f"sample types were specified more than once for {project_id}"
            )

        codes = tuple(
            dict.fromkeys(code.strip() for code in codes_text.split(","))
        )
        if not codes or any(not code for code in codes):
            raise ValueError(
                f"no sample-type codes were specified for {project_id}"
            )
        for code in codes:
            validate_sample_type_code(code)
        policy[project_id] = codes
        overridden.add(project_id)

    return policy


def validate_args(args: argparse.Namespace) -> dict[str, tuple[str, ...]]:
    if not 0.0 < args.representative_fraction < 1.0:
        raise ValueError(
            "--representative-fraction must be strictly between zero and one"
        )
    if args.expected_features < 0:
        raise ValueError("--expected-features cannot be negative")
    return build_sample_type_policy(args.sample_types)


def open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8", newline="")
    return path.open(mode="r", encoding="utf-8", newline="")


def resolve_matrix_path(input_dir: Path, cohort: Cohort) -> Path:
    compressed = input_dir / cohort.file_name
    uncompressed = input_dir / cohort.file_name.removesuffix(".gz")
    existing = [
        path for path in (compressed, uncompressed) if path.is_file()
    ]
    if not existing:
        raise RuntimeError(
            f"missing {cohort.project_id} matrix: expected "
            f"{compressed.name} or {uncompressed.name}"
        )
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
    return "-".join(fields[:3]), sample_type_code


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
    sample_types_by_project: dict[str, tuple[str, ...]],
) -> tuple[list[CohortMatrix], list[Sample]]:
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

        retained_types = set(sample_types_by_project[cohort.project_id])
        selected_columns: list[int] = []
        output_start = len(samples)
        for source_column, sample_id in enumerate(sample_ids):
            participant_id, sample_type_code = parse_sample_id(sample_id)
            if sample_type_code not in retained_types:
                continue
            selected_columns.append(source_column)
            samples.append(
                Sample(
                    row_index=len(samples),
                    sample_id=sample_id,
                    participant_id=participant_id,
                    project_id=cohort.project_id,
                    cancer_type=cohort.cancer_type,
                    label=cohort.label,
                    sample_type_code=sample_type_code,
                    sample_type=SAMPLE_TYPE_NAMES.get(
                        sample_type_code,
                        "Unknown TCGA sample type",
                    ),
                )
            )

        if not selected_columns:
            selected_text = ", ".join(
                sample_types_by_project[cohort.project_id]
            )
            raise RuntimeError(
                f"{path.name} has no samples with requested types: "
                f"{selected_text}"
            )
        cohort_matrices.append(
            CohortMatrix(
                cohort=cohort,
                path=path,
                sample_ids=sample_ids,
                selected_columns=np.asarray(
                    selected_columns,
                    dtype=np.intp,
                ),
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
                f"participant {sample.participant_id} appears under "
                "multiple labels"
            )
        participants_by_label[sample.label].add(sample.participant_id)

    random_engine = random.Random(seed)
    participant_roles: dict[str, str] = {}
    for label in sorted(participants_by_label):
        participants = sorted(participants_by_label[label])
        if len(participants) < 2:
            project_id = COHORTS[label].project_id
            raise RuntimeError(
                f"{project_id} has fewer than two retained participants; "
                "cannot construct a representative and an independent query"
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
            f"{path.name} feature {feature_id!r} contains {values.size} "
            f"values; expected {expected_values}"
        )
    if not np.isfinite(values).all():
        raise RuntimeError(
            f"{path.name} feature {feature_id!r} contains a non-finite value"
        )
    return feature_id, values


def indices_for_role(roles: list[str], role: str) -> np.ndarray:
    return np.asarray(
        [index for index, value in enumerate(roles) if value == role],
        dtype=np.intp,
    )


def role_row_indices(roles: list[str]) -> list[int]:
    next_row = {"representative_pool": 0, "query": 0}
    result: list[int] = []
    for role in roles:
        result.append(next_row[role])
        next_row[role] += 1
    return result


def build_role_layouts(
    cohort_matrices: list[CohortMatrix],
    roles: list[str],
) -> list[CohortRoleLayout]:
    output_rows = role_row_indices(roles)
    layouts: list[CohortRoleLayout] = []
    for matrix in cohort_matrices:
        pool_local: list[int] = []
        pool_rows: list[int] = []
        query_local: list[int] = []
        query_rows: list[int] = []
        for local_index, sample_index in enumerate(
            range(matrix.output_start, matrix.output_end)
        ):
            if roles[sample_index] == "representative_pool":
                pool_local.append(local_index)
                pool_rows.append(output_rows[sample_index])
            else:
                query_local.append(local_index)
                query_rows.append(output_rows[sample_index])

        if not pool_rows or not query_rows:
            raise RuntimeError(
                f"{matrix.cohort.project_id} does not have both roles"
            )
        if pool_rows != list(range(pool_rows[0], pool_rows[-1] + 1)):
            raise RuntimeError("representative-pool row layout is not contiguous")
        if query_rows != list(range(query_rows[0], query_rows[-1] + 1)):
            raise RuntimeError("query row layout is not contiguous")

        layouts.append(
            CohortRoleLayout(
                pool_local_indices=np.asarray(pool_local, dtype=np.intp),
                pool_output_start=pool_rows[0],
                pool_output_end=pool_rows[-1] + 1,
                query_local_indices=np.asarray(query_local, dtype=np.intp),
                query_output_start=query_rows[0],
                query_output_end=query_rows[-1] + 1,
            )
        )
    return layouts


def write_sample_major(
    feature_major: np.ndarray,
    path: Path,
) -> None:
    output = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.dtype("<f4"),
        shape=(feature_major.shape[1], feature_major.shape[0]),
        fortran_order=False,
    )
    try:
        for start in range(0, feature_major.shape[1], TRANSPOSE_BLOCK_ROWS):
            end = min(start + TRANSPOSE_BLOCK_ROWS, feature_major.shape[1])
            output[start:end, :] = feature_major[:, start:end].T
        output.flush()
    finally:
        del output


def convert_expression_matrices(
    cohort_matrices: list[CohortMatrix],
    roles: list[str],
    feature_count: int,
    staging_dir: Path,
    write_representative_pool: bool,
) -> list[str]:
    query_count = roles.count("query")
    pool_count = roles.count("representative_pool")
    query_temporary_path = staging_dir / "queries_features_by_samples.npy"
    pool_temporary_path = staging_dir / "pool_features_by_samples.npy"
    query_feature_major = np.lib.format.open_memmap(
        query_temporary_path,
        mode="w+",
        dtype=np.dtype("<f4"),
        shape=(feature_count, query_count),
        fortran_order=False,
    )
    pool_feature_major = None
    if write_representative_pool:
        pool_feature_major = np.lib.format.open_memmap(
            pool_temporary_path,
            mode="w+",
            dtype=np.dtype("<f4"),
            shape=(feature_count, pool_count),
            fortran_order=False,
        )
    representatives = np.lib.format.open_memmap(
        staging_dir / "representatives.npy",
        mode="w+",
        dtype=np.dtype("<f4"),
        shape=(len(COHORTS), feature_count),
        fortran_order=False,
    )
    layouts = build_role_layouts(cohort_matrices, roles)
    feature_ids: list[str] = []

    try:
        with ExitStack() as stack:
            streams = [
                stack.enter_context(open_text(matrix.path))
                for matrix in cohort_matrices
            ]
            for stream, matrix in zip(streams, cohort_matrices, strict=True):
                header = stream.readline().rstrip("\r\n")
                expected = "\t".join(("Ensembl_ID", *matrix.sample_ids))
                if header != expected:
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
                for line, matrix, layout in zip(
                    lines,
                    cohort_matrices,
                    layouts,
                    strict=True,
                ):
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
                            "cohort matrices do not have matching Ensembl gene "
                            f"order: expected {shared_feature_id!r}, found "
                            f"{feature_id!r} in {matrix.path.name}"
                        )

                    retained = values[matrix.selected_columns]
                    pool_values = retained[layout.pool_local_indices]
                    representatives[
                        matrix.cohort.label,
                        feature_index,
                    ] = np.mean(pool_values, dtype=np.float64)
                    query_feature_major[
                        feature_index,
                        layout.query_output_start : layout.query_output_end,
                    ] = retained[layout.query_local_indices]
                    if pool_feature_major is not None:
                        pool_feature_major[
                            feature_index,
                            layout.pool_output_start : layout.pool_output_end,
                        ] = pool_values

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

        representatives.flush()
        query_feature_major.flush()
        if pool_feature_major is not None:
            pool_feature_major.flush()

        print("Writing sample-major query data...", flush=True)
        write_sample_major(
            query_feature_major,
            staging_dir / "queries.npy",
        )
        if pool_feature_major is not None:
            print("Writing sample-major representative pool...", flush=True)
            write_sample_major(
                pool_feature_major,
                staging_dir / "representative_pool.npy",
            )
    finally:
        del representatives
        del query_feature_major
        if pool_feature_major is not None:
            del pool_feature_major
        if query_temporary_path.exists():
            query_temporary_path.unlink()
        if pool_temporary_path.exists():
            pool_temporary_path.unlink()

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


def write_samples(path: Path, samples: list[Sample], roles: list[str]) -> None:
    output_rows = role_row_indices(roles)
    with path.open("w", encoding="utf-8", newline="") as output:
        fieldnames = (
            "sample_index",
            "role",
            "role_row_index",
            "sample_id",
            "participant_id",
            "project_id",
            "cancer_type",
            "label",
            "sample_type_code",
            "sample_type",
        )
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for sample, role, role_row in zip(
            samples,
            roles,
            output_rows,
            strict=True,
        ):
            writer.writerow(
                {
                    "sample_index": sample.row_index,
                    "role": role,
                    "role_row_index": role_row,
                    "sample_id": sample.sample_id,
                    "participant_id": sample.participant_id,
                    "project_id": sample.project_id,
                    "cancer_type": sample.cancer_type,
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
            "cancer_type",
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
                    "cancer_type": cohort.cancer_type,
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
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def count_by(values: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return dict(sorted(result.items()))


def counts_by_project_and_role(
    samples: list[Sample],
    roles: list[str],
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for sample, role in zip(samples, roles, strict=True):
        project_counts = result.setdefault(
            sample.project_id,
            {"representative_pool": 0, "query": 0},
        )
        project_counts[role] += 1
    return dict(sorted(result.items()))


def read_source_manifest(input_dir: Path) -> dict[str, object] | None:
    path = input_dir / "manifest.json"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise RuntimeError("source manifest must contain a JSON object")
    return value


def build_dataset_metadata(
    cohort_matrices: list[CohortMatrix],
    samples: list[Sample],
    roles: list[str],
    sample_types_by_project: dict[str, tuple[str, ...]],
    feature_count: int,
    args: argparse.Namespace,
    staging_dir: Path,
    source_manifest: dict[str, object] | None,
) -> dict[str, object]:
    generated_files = {
        path.name: file_record(path)
        for path in sorted(staging_dir.iterdir())
        if path.is_file() and path.name != "dataset.json"
    }
    manifest_files = (
        source_manifest.get("files", {}) if source_manifest is not None else {}
    )
    if not isinstance(manifest_files, dict):
        manifest_files = {}
    source_files: dict[str, dict[str, object]] = {}
    for matrix in cohort_matrices:
        record: dict[str, object] = {
            "path": str(matrix.path),
            "bytes": matrix.path.stat().st_size,
            "source_sample_count": len(matrix.sample_ids),
            "retained_sample_count": matrix.output_end - matrix.output_start,
        }
        manifest_record = manifest_files.get(matrix.path.name)
        if isinstance(manifest_record, dict) and isinstance(
            manifest_record.get("sha256"), str
        ):
            record["sha256"] = manifest_record["sha256"]
        source_files[matrix.path.name] = record

    participant_role_pairs = {
        (sample.participant_id, role)
        for sample, role in zip(samples, roles, strict=True)
    }
    participant_roles = [role for _, role in participant_role_pairs]
    pool_count = roles.count("representative_pool")
    query_count = roles.count("query")
    matrices: dict[str, object] = {
        "representatives": {
            "file": "representatives.npy",
            "shape": [len(COHORTS), feature_count],
            "purpose": "one arithmetic-mean centroid per cancer project",
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
    }
    if args.write_representative_pool:
        matrices["representative_pool"] = {
            "file": "representative_pool.npy",
            "shape": [pool_count, feature_count],
            "purpose": "samples used to construct representatives",
        }

    return {
        "dataset_name": "TCGA Pan-Cancer Primary Disease",
        "format_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "matrices": matrices,
        "labels": {
            "query_file": "query_labels.npy",
            "representative_pool_file": (
                "representative_pool_labels.npy"
                if args.write_representative_pool
                else None
            ),
            "dtype": "uint16",
            "mapping": {
                str(cohort.label): cohort.project_id for cohort in COHORTS
            },
        },
        "representative_construction": {
            "metadata_file": "representatives.csv",
            "method": "arithmetic_mean_centroid",
            "source": "samples assigned role=representative_pool in samples.csv",
            "count_per_project": 1,
            "pool_matrix_materialized": args.write_representative_pool,
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
            "counts_by_project_and_role": counts_by_project_and_role(
                samples,
                roles,
            ),
            "counts_by_sample_type": count_by(
                [sample.sample_type_code for sample in samples]
            ),
            "counts_by_role": count_by(roles),
        },
        "sample_filter": {
            "policy": "project_specific_primary_disease",
            "retained_sample_type_codes_by_project": {
                project_id: list(codes)
                for project_id, codes in sample_types_by_project.items()
            },
            "definitions": {
                code: SAMPLE_TYPE_NAMES.get(code, "Unknown TCGA sample type")
                for codes in sample_types_by_project.values()
                for code in codes
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
            "filter samples using project-specific TCGA sample-type codes",
            "split participants into representative-pool and query roles",
            "transpose held-out queries from features_by_samples to "
            "samples_by_features",
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
        sample_types_by_project = validate_args(args)
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
            sample_types_by_project,
        )
        feature_count = args.expected_features
        if feature_count == 0:
            feature_count = count_feature_rows(cohort_matrices[0].path)
        roles = assign_participant_roles(
            samples,
            args.representative_fraction,
            args.seed,
        )
        pool_indices = indices_for_role(roles, "representative_pool")
        query_indices = indices_for_role(roles, "query")

        print(
            f"Retained {len(samples):,} primary-disease samples from "
            f"{len(COHORTS)} projects: {len(pool_indices):,} in the "
            f"representative pool and {len(query_indices):,} queries; "
            f"expect {feature_count:,} features.",
            flush=True,
        )

        with tempfile.TemporaryDirectory(
            prefix=".tcga-pancancer-transform-",
            dir=output_dir,
        ) as temporary_directory:
            staging_dir = Path(temporary_directory)
            feature_ids = convert_expression_matrices(
                cohort_matrices,
                roles,
                feature_count,
                staging_dir,
                args.write_representative_pool,
            )
            write_labels(
                staging_dir / "query_labels.npy",
                samples,
                query_indices,
            )
            if args.write_representative_pool:
                write_labels(
                    staging_dir / "representative_pool_labels.npy",
                    samples,
                    pool_indices,
                )
            write_samples(staging_dir / "samples.csv", samples, roles)
            write_representatives(
                staging_dir / "representatives.csv",
                samples,
                roles,
            )
            write_features(staging_dir / "features.csv", feature_ids)

            source_manifest = read_source_manifest(input_dir)
            if source_manifest is not None:
                shutil.copyfile(
                    input_dir / "manifest.json",
                    staging_dir / "source_manifest.json",
                )
            else:
                with (staging_dir / "source_manifest.json").open(
                    "w",
                    encoding="utf-8",
                ) as output:
                    json.dump({"available": False}, output, indent=2)
                    output.write("\n")

            metadata = build_dataset_metadata(
                cohort_matrices,
                samples,
                roles,
                sample_types_by_project,
                feature_count,
                args,
                staging_dir,
                source_manifest,
            )
            with (staging_dir / "dataset.json").open(
                "w",
                encoding="utf-8",
            ) as output:
                json.dump(metadata, output, indent=2, sort_keys=True)
                output.write("\n")

            install_staged_outputs(staging_dir, output_dir, args.force)

    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
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
