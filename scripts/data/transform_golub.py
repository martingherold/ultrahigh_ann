#!/usr/bin/env python3
"""Transform the Golub ALL/AML data into an original-split centroid benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from download_golub import (
    EXPECTED_FEATURES,
    EXPECTED_MEMBERS,
    EXPECTED_TEST_COUNTS,
    EXPECTED_TEST_SAMPLES,
    EXPECTED_TRAIN_COUNTS,
    EXPECTED_TRAIN_SAMPLES,
    PACKAGE_NAME,
)

try:
    import numpy as np
except ModuleNotFoundError as error:
    print(
        "transform_golub.py requires NumPy; install it with "
        "'python3 -m pip install numpy'.",
        file=sys.stderr,
    )
    raise SystemExit(2) from error


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "golub"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "golub_v1"
DEFAULT_EXPORT_SCRIPT = Path(__file__).resolve().with_name("export_golub_esets.R")
HASH_BLOCK_SIZE = 1024 * 1024
LABELS = {"ALL": 0, "AML": 1}
LABEL_NAMES = {0: "ALL", 1: "AML"}
PHENOTYPE_COLUMNS = (
    "Samples",
    "ALL.AML",
    "BM.PB",
    "T.B.cell",
    "FAB",
    "Date",
    "Gender",
    "pctBlasts",
    "Treatment",
    "PS",
    "Source",
)
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
            "Extract the Bioconductor Golub training/test ExpressionSets, "
            "preserve the published split, fit per-probe z-scores only on "
            "the 38 training samples, and construct one centroid per class."
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
        "--rscript",
        default="Rscript",
        help="Rscript executable used only to deserialize ExpressionSet files",
    )
    parser.add_argument(
        "--export-script",
        type=Path,
        default=DEFAULT_EXPORT_SCRIPT,
        help=f"R data exporter (default: {DEFAULT_EXPORT_SCRIPT})",
    )
    parser.add_argument(
        "--expected-features",
        type=int,
        default=EXPECTED_FEATURES,
        help=f"expected probe count (default: {EXPECTED_FEATURES})",
    )
    parser.add_argument(
        "--expected-train-samples",
        type=int,
        default=EXPECTED_TRAIN_SAMPLES,
        help=f"expected original training count (default: {EXPECTED_TRAIN_SAMPLES})",
    )
    parser.add_argument(
        "--expected-test-samples",
        type=int,
        default=EXPECTED_TEST_SAMPLES,
        help=f"expected original test count (default: {EXPECTED_TEST_SAMPLES})",
    )
    parser.add_argument(
        "--expected-train-all",
        type=int,
        default=EXPECTED_TRAIN_COUNTS["ALL"],
        help=f"expected ALL training count (default: {EXPECTED_TRAIN_COUNTS['ALL']})",
    )
    parser.add_argument(
        "--expected-train-aml",
        type=int,
        default=EXPECTED_TRAIN_COUNTS["AML"],
        help=f"expected AML training count (default: {EXPECTED_TRAIN_COUNTS['AML']})",
    )
    parser.add_argument(
        "--expected-test-all",
        type=int,
        default=EXPECTED_TEST_COUNTS["ALL"],
        help=f"expected ALL test count (default: {EXPECTED_TEST_COUNTS['ALL']})",
    )
    parser.add_argument(
        "--expected-test-aml",
        type=int,
        default=EXPECTED_TEST_COUNTS["AML"],
        help=f"expected AML test count (default: {EXPECTED_TEST_COUNTS['AML']})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace managed files from an earlier transformation",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.expected_features < 1:
        raise ValueError("--expected-features must be positive")
    counts = (
        args.expected_train_samples,
        args.expected_test_samples,
        args.expected_train_all,
        args.expected_train_aml,
        args.expected_test_all,
        args.expected_test_aml,
    )
    if any(count < 1 for count in counts):
        raise ValueError("all expected sample and class counts must be positive")
    if args.expected_train_all + args.expected_train_aml != args.expected_train_samples:
        raise ValueError("expected training class counts must sum to the training count")
    if args.expected_test_all + args.expected_test_aml != args.expected_test_samples:
        raise ValueError("expected test class counts must sum to the test count")


def extract_rdata(package_path: Path, destination: Path) -> tuple[Path, Path]:
    destination.mkdir()
    outputs: dict[str, Path] = {}
    try:
        with tarfile.open(package_path, mode="r:gz") as archive:
            members = {member.name: member for member in archive.getmembers()}
            for split in ("train", "test"):
                archive_name = EXPECTED_MEMBERS[split]
                member = members.get(archive_name)
                if member is None or not member.isfile() or member.size <= 0:
                    raise RuntimeError(
                        f"source package is missing regular file {archive_name}"
                    )
                source = archive.extractfile(member)
                if source is None:
                    raise RuntimeError(f"cannot read {archive_name}")
                output_path = destination / Path(archive_name).name
                with source, output_path.open("wb") as output:
                    shutil.copyfileobj(source, output)
                if output_path.stat().st_size != member.size:
                    raise RuntimeError(f"extracted file is truncated: {archive_name}")
                outputs[split] = output_path
    except (tarfile.TarError, OSError) as error:
        raise RuntimeError(f"cannot extract {package_path.name}: {error}") from error
    return outputs["train"], outputs["test"]


def resolve_executable(command: str) -> str:
    expanded = str(Path(command).expanduser())
    if Path(expanded).parent != Path("."):
        path = Path(expanded).resolve()
        if not path.is_file():
            raise RuntimeError(f"Rscript executable does not exist: {path}")
        return str(path)
    executable = shutil.which(command)
    if executable is None:
        raise RuntimeError(
            "Rscript is required to deserialize Bioconductor's .rda files; "
            "install R or pass --rscript PATH"
        )
    return executable


def export_rdata(
    rscript: str,
    export_script: Path,
    train_rdata: Path,
    test_rdata: Path,
    destination: Path,
) -> None:
    if not export_script.is_file():
        raise RuntimeError(f"Golub R exporter does not exist: {export_script}")
    completed = subprocess.run(
        (
            rscript,
            "--vanilla",
            str(export_script),
            str(train_rdata),
            str(test_rdata),
            str(destination),
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        if len(detail) > 2_000:
            detail = detail[-2_000:]
        raise RuntimeError(
            f"R data export failed with exit code {completed.returncode}: {detail}"
        )
    expected = {
        destination / f"{split}_{kind}.tsv"
        for split in ("train", "test")
        for kind in ("expression", "samples")
    }
    missing = sorted(str(path) for path in expected if not path.is_file())
    if missing:
        raise RuntimeError(f"R exporter did not create: {', '.join(missing)}")


def load_expression_tsv(
    path: Path,
    expected_samples: int,
    expected_features: int,
) -> tuple[tuple[str, ...], tuple[str, ...], np.ndarray]:
    try:
        with path.open(encoding="utf-8", newline="") as source:
            rows = csv.reader(source, delimiter="\t")
            try:
                header = next(rows)
            except StopIteration as error:
                raise RuntimeError(f"{path.name} is empty") from error
            if len(header) != expected_samples + 1 or header[0] != "":
                raise RuntimeError(f"{path.name} has an unexpected header")
            sample_ids = tuple(header[1:])
            if any(not sample_id for sample_id in sample_ids):
                raise RuntimeError(f"{path.name} has an empty sample identifier")
            if len(set(sample_ids)) != len(sample_ids):
                raise RuntimeError(f"{path.name} has duplicate sample identifiers")

            feature_ids: list[str] = []
            matrix = np.empty((expected_samples, expected_features), dtype="<f4")
            for feature_index, row in enumerate(rows):
                if feature_index >= expected_features:
                    raise RuntimeError(
                        f"{path.name} has more than {expected_features} features"
                    )
                if len(row) != expected_samples + 1:
                    raise RuntimeError(
                        f"{path.name} row {feature_index + 2} has "
                        f"{len(row) - 1} values; expected {expected_samples}"
                    )
                feature_id = row[0]
                if not feature_id:
                    raise RuntimeError(
                        f"{path.name} row {feature_index + 2} has no feature ID"
                    )
                try:
                    values = np.asarray(row[1:], dtype="<f4")
                except ValueError as error:
                    raise RuntimeError(
                        f"{path.name} row {feature_index + 2} is non-numeric"
                    ) from error
                if not np.isfinite(values).all():
                    raise RuntimeError(
                        f"{path.name} row {feature_index + 2} is non-finite"
                    )
                feature_ids.append(feature_id)
                matrix[:, feature_index] = values
    except (UnicodeDecodeError, csv.Error) as error:
        raise RuntimeError(f"cannot parse {path.name}: {error}") from error

    if len(feature_ids) != expected_features:
        raise RuntimeError(
            f"{path.name} has {len(feature_ids)} features; expected {expected_features}"
        )
    if len(set(feature_ids)) != len(feature_ids):
        raise RuntimeError(f"{path.name} has duplicate feature identifiers")
    return sample_ids, tuple(feature_ids), matrix


def load_phenotypes(
    path: Path,
    expression_sample_ids: tuple[str, ...],
    expected_counts: dict[str, int],
) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source, delimiter="\t")
            if tuple(reader.fieldnames or ()) != PHENOTYPE_COLUMNS:
                raise RuntimeError(f"{path.name} has unexpected phenotype columns")
            rows = [dict(row) for row in reader]
    except (UnicodeDecodeError, csv.Error) as error:
        raise RuntimeError(f"cannot parse {path.name}: {error}") from error

    sample_ids = [row["Samples"] for row in rows]
    if tuple(sample_ids) != expression_sample_ids:
        raise RuntimeError(f"{path.name} does not align with expression columns")
    if len(set(sample_ids)) != len(sample_ids):
        raise RuntimeError(f"{path.name} has duplicate sample identifiers")
    diagnoses = [row["ALL.AML"] for row in rows]
    if any(diagnosis not in LABELS for diagnosis in diagnoses):
        raise RuntimeError(f"{path.name} contains an unknown ALL/AML label")
    counts = Counter(diagnoses)
    if dict(counts) != expected_counts:
        raise RuntimeError(
            f"{path.name} class counts are {dict(counts)}; expected {expected_counts}"
        )
    return rows


def normalize_from_training(
    training: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    mean = np.mean(training, axis=0, dtype=np.float64).astype("<f4")
    scale = np.std(training, axis=0, dtype=np.float64).astype("<f4")
    constant = scale == 0.0
    constant_count = int(np.count_nonzero(constant))
    scale[constant] = np.float32(1.0)
    np.subtract(training, mean, out=training)
    np.divide(training, scale, out=training)
    np.subtract(test, mean, out=test)
    np.divide(test, scale, out=test)
    if not np.isfinite(training).all() or not np.isfinite(test).all():
        raise RuntimeError("normalization produced non-finite values")
    return mean, scale, constant_count


def save_array(path: Path, values: np.ndarray, dtype: str) -> None:
    np.save(path, np.ascontiguousarray(values, dtype=dtype), allow_pickle=False)


def write_samples(
    path: Path,
    train_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
) -> None:
    fieldnames = (
        "sample_id",
        "source_split",
        "source_column",
        "role",
        "role_row",
        "label",
        "diagnosis",
        "specimen",
        "lineage",
        "fab",
        "date",
        "gender",
        "pct_blasts",
        "treatment",
        "prediction_strength",
        "source",
    )
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for split, role, rows in (
            ("published_train", "representative_pool", train_rows),
            ("published_test", "query", test_rows),
        ):
            for index, row in enumerate(rows):
                diagnosis = row["ALL.AML"]
                writer.writerow(
                    {
                        "sample_id": row["Samples"],
                        "source_split": split,
                        "source_column": index,
                        "role": role,
                        "role_row": index,
                        "label": LABELS[diagnosis],
                        "diagnosis": diagnosis,
                        "specimen": row["BM.PB"],
                        "lineage": row["T.B.cell"],
                        "fab": row["FAB"],
                        "date": row["Date"],
                        "gender": row["Gender"],
                        "pct_blasts": row["pctBlasts"],
                        "treatment": row["Treatment"],
                        "prediction_strength": row["PS"],
                        "source": row["Source"],
                    }
                )


def write_features(path: Path, feature_ids: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("feature_index", "affymetrix_probe_id"))
        writer.writerows(enumerate(feature_ids))


def write_representatives(path: Path, labels: np.ndarray) -> None:
    counts = Counter(int(label) for label in labels)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(
            (
                "representative_row",
                "label",
                "diagnosis",
                "construction",
                "source_sample_count",
            )
        )
        for label in sorted(LABEL_NAMES):
            writer.writerow(
                (
                    label,
                    label,
                    LABEL_NAMES[label],
                    "published_training_mean_centroid",
                    counts[label],
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
    constant_feature_count: int,
    source_manifest_available: bool,
) -> dict[str, object]:
    generated_files = {
        path.name: file_record(path)
        for path in sorted(staging_dir.iterdir())
        if path.is_file() and path.name != "dataset.json"
    }
    return {
        "dataset_name": "Golub published-split ALL-versus-AML benchmark",
        "format_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "matrices": {
            "representatives": {
                "file": "representatives.npy",
                "shape": [2, args.expected_features],
                "purpose": "one published-training centroid per diagnosis",
            },
            "representative_pool": {
                "file": "representative_pool.npy",
                "shape": [args.expected_train_samples, args.expected_features],
                "purpose": "all samples in the original Golub training split",
            },
            "queries": {
                "file": "queries.npy",
                "shape": [args.expected_test_samples, args.expected_features],
                "purpose": "all samples in the original Golub test split",
            },
            "common_layout": {
                "orientation": "rows_by_features",
                "feature_order": "Hgu6800 probe order in golubEsets",
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
            "source_column": "ALL.AML",
        },
        "split": {
            "policy": "original Golub published train/test split",
            "representative_pool_count": args.expected_train_samples,
            "query_count": args.expected_test_samples,
            "representative_pool_counts_by_class": {
                "ALL": args.expected_train_all,
                "AML": args.expected_train_aml,
            },
            "query_counts_by_class": {
                "ALL": args.expected_test_all,
                "AML": args.expected_test_aml,
            },
            "leakage_control": (
                "normalization statistics and class centroids use only the "
                "published training split"
            ),
        },
        "normalization": {
            "method": "per-probe population z-score without log transform",
            "fit_rows": "published training split only",
            "mean_file": "normalization_mean.npy",
            "scale_file": "normalization_scale.npy",
            "zero_variance_probes": constant_feature_count,
            "zero_variance_policy": "retain the probe and use scale 1",
            "log_transform_omitted_because": (
                "packaged golubEsets expression values include negative values"
            ),
        },
        "representative_construction": {
            "method": "arithmetic mean centroid",
            "count_per_class": 1,
            "metadata_file": "representatives.csv",
        },
        "features": {
            "file": "features.csv",
            "count": args.expected_features,
            "definition": "Affymetrix Hgu6800 probes retained in full",
        },
        "samples": {
            "file": "samples.csv",
            "count": args.expected_train_samples + args.expected_test_samples,
        },
        "source_manifest_available": source_manifest_available,
        "transformations": [
            "deserialize Golub_Train and Golub_Test from the pinned package",
            "retain all packaged Affymetrix probes in their original order",
            "preserve the published training and test split",
            "fit per-probe means and scales on published training rows only",
            "apply those training-only z-scores to both splits",
            "construct one arithmetic-mean training centroid per diagnosis",
        ],
        "numpy_version": np.__version__,
        "generated_files": generated_files,
    }


def install_staged_outputs(staging_dir: Path, output_dir: Path, force: bool) -> None:
    staged_names = {path.name for path in staging_dir.iterdir() if path.is_file()}
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
                f"refusing to write through symbolic-link output: {requested_output_dir}"
            )
        output_dir = requested_output_dir.resolve()
        if not input_dir.is_dir():
            raise RuntimeError(f"input directory does not exist: {input_dir}")
        package_path = input_dir / PACKAGE_NAME
        if not package_path.is_file():
            raise RuntimeError(f"missing source package: {package_path}")
        rscript = resolve_executable(args.rscript)
        export_script = args.export_script.expanduser().resolve()

        output_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(
            name for name in MANAGED_OUTPUT_NAMES if (output_dir / name).exists()
        )
        if existing and not args.force:
            print(
                f"Refusing to overwrite existing files in {output_dir}: "
                f"{', '.join(existing)}\nRun again with --force to replace them.",
                file=sys.stderr,
            )
            return 1

        with tempfile.TemporaryDirectory(
            prefix=".golub-transform-", dir=output_dir
        ) as temporary_directory:
            temporary_root = Path(temporary_directory)
            extracted = temporary_root / "rdata"
            exported = temporary_root / "exported"
            train_rdata, test_rdata = extract_rdata(package_path, extracted)
            print("Exporting the two Bioconductor ExpressionSets...", flush=True)
            export_rdata(
                rscript,
                export_script,
                train_rdata,
                test_rdata,
                exported,
            )

            print(
                f"Loading all {args.expected_features:,} Golub probes and "
                "validating the published split...",
                flush=True,
            )
            train_ids, train_features, training = load_expression_tsv(
                exported / "train_expression.tsv",
                args.expected_train_samples,
                args.expected_features,
            )
            test_ids, test_features, queries = load_expression_tsv(
                exported / "test_expression.tsv",
                args.expected_test_samples,
                args.expected_features,
            )
            if train_features != test_features:
                raise RuntimeError("training and test feature orders differ")
            if set(train_ids) & set(test_ids):
                raise RuntimeError("published training and test samples overlap")
            train_rows = load_phenotypes(
                exported / "train_samples.tsv",
                train_ids,
                {
                    "ALL": args.expected_train_all,
                    "AML": args.expected_train_aml,
                },
            )
            test_rows = load_phenotypes(
                exported / "test_samples.tsv",
                test_ids,
                {
                    "ALL": args.expected_test_all,
                    "AML": args.expected_test_aml,
                },
            )
            train_labels = np.asarray(
                [LABELS[row["ALL.AML"]] for row in train_rows], dtype="<u2"
            )
            query_labels = np.asarray(
                [LABELS[row["ALL.AML"]] for row in test_rows], dtype="<u2"
            )

            mean, scale, constant_feature_count = normalize_from_training(
                training, queries
            )
            representatives = np.stack(
                [
                    training[train_labels == label].mean(axis=0, dtype=np.float64)
                    for label in sorted(LABEL_NAMES)
                ]
            ).astype("<f4")
            representative_labels = np.asarray(sorted(LABEL_NAMES), dtype="<u2")

            staging_dir = temporary_root / "staged"
            staging_dir.mkdir()
            save_array(staging_dir / "representatives.npy", representatives, "<f4")
            save_array(
                staging_dir / "representative_labels.npy",
                representative_labels,
                "<u2",
            )
            save_array(staging_dir / "representative_pool.npy", training, "<f4")
            save_array(
                staging_dir / "representative_pool_labels.npy", train_labels, "<u2"
            )
            save_array(staging_dir / "queries.npy", queries, "<f4")
            save_array(staging_dir / "query_labels.npy", query_labels, "<u2")
            save_array(staging_dir / "normalization_mean.npy", mean, "<f4")
            save_array(staging_dir / "normalization_scale.npy", scale, "<f4")
            write_samples(staging_dir / "samples.csv", train_rows, test_rows)
            write_features(staging_dir / "features.csv", train_features)
            write_representatives(
                staging_dir / "representatives.csv", train_labels
            )

            source_manifest = read_source_manifest(input_dir)
            if source_manifest is None:
                with (staging_dir / "source_manifest.json").open(
                    "w", encoding="utf-8"
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
                constant_feature_count,
                source_manifest is not None,
            )
            with (staging_dir / "dataset.json").open(
                "w", encoding="utf-8"
            ) as output:
                json.dump(metadata, output, indent=2, sort_keys=True)
                output.write("\n")
            install_staged_outputs(staging_dir, output_dir, args.force)
    except (
        OSError,
        RuntimeError,
        ValueError,
        json.JSONDecodeError,
        subprocess.SubprocessError,
    ) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(
        f"Saved 2 representatives and {args.expected_test_samples} published-test "
        f"queries with {args.expected_features:,} dimensions to {output_dir}.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
