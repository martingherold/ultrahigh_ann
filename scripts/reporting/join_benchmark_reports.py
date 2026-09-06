#!/usr/bin/env python3
"""Join compatible schema-version 6 benchmark checkpoint reports."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "benchmark"))

from benchmark_reporting import validate_report_provenance  # noqa: E402


REPORT_SCHEMA_VERSION = 6
REFERENCE_TIMING_FIELDS = {
    "trial_ms",
    "median_ms",
    "median_microseconds_per_query",
    "median_queries_per_second",
}
PROBABILITY_TIMING_FIELDS = {
    "build_ms",
    "load_ms",
    "host_to_device_ms",
    "inverse_distance_ms",
    "coordinate_maximum_ms",
    "device_to_host_ms",
    "total_ms",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Join independently completed benchmark checkpoints into one "
            "schema-version 6 JSON/CSV pair for reporting and figures."
        )
    )
    parser.add_argument(
        "--reports",
        type=Path,
        nargs="+",
        required=True,
        help="source JSON reports in their intended output order",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="joined JSON output",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        help="joined CSV output (default: JSON output with .csv suffix)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace existing joined outputs",
    )
    return parser.parse_args()


def require_mapping(value: object, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{description} is not an object")
    return value


def require_list(value: object, description: str) -> list[Any]:
    if not isinstance(value, list):
        raise RuntimeError(f"{description} is not an array")
    return value


def require_number(value: object, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{description} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"{description} is not finite")
    return result


def load_report(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as source:
            report = require_mapping(json.load(source), f"report {path}")
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read report {path}: {error}") from error
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise RuntimeError(
            f"report {path} does not use schema version {REPORT_SCHEMA_VERSION}"
        )
    validate_report_provenance(report)
    references = [
        require_mapping(run, f"run in {path}")
        for run in require_list(report.get("runs"), f"runs in {path}")
        if isinstance(run, dict) and run.get("reference") is True
    ]
    if len(references) != 1 or references[0].get("index") != "exact":
        raise RuntimeError(f"report {path} must contain one exact reference")
    return report


def without_keys(value: object, ignored: set[str]) -> object:
    if isinstance(value, dict):
        return {
            key: without_keys(item, ignored)
            for key, item in value.items()
            if key not in ignored
        }
    if isinstance(value, list):
        return [without_keys(item, ignored) for item in value]
    return value


def comparable_dataset(report: dict[str, Any]) -> object:
    return without_keys(
        require_mapping(report.get("dataset"), "dataset"),
        {"hash_ms", "load_ms"},
    )


def comparable_probabilities(report: dict[str, Any]) -> object:
    return without_keys(
        require_mapping(
            report.get("sampling_probabilities"),
            "sampling probabilities",
        ),
        PROBABILITY_TIMING_FIELDS,
    )


def comparable_provenance(report: dict[str, Any]) -> object:
    provenance = require_mapping(report.get("provenance"), "provenance")
    return {
        key: copy.deepcopy(value)
        for key, value in provenance.items()
        if key != "invocation"
    }


def reference_run(report: dict[str, Any]) -> dict[str, Any]:
    return next(
        require_mapping(run, "reference run")
        for run in require_list(report.get("runs"), "runs")
        if isinstance(run, dict) and run.get("reference") is True
    )


def reference_structure(run: dict[str, Any]) -> object:
    return without_keys(
        run,
        {"build_ms", "measurements", "trials", "warmups"},
    )


def measurements_by_batch(run: dict[str, Any]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for raw in require_list(run.get("measurements"), "reference measurements"):
        measurement = require_mapping(raw, "reference measurement")
        batch_size = measurement.get("batch_size")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise RuntimeError("reference batch size is not an integer")
        if batch_size in result:
            raise RuntimeError(f"reference repeats batch size {batch_size}")
        result[batch_size] = measurement
    if not result:
        raise RuntimeError("reference run has no measurements")
    return result


def reference_measurement_structure(measurement: dict[str, Any]) -> object:
    return without_keys(measurement, REFERENCE_TIMING_FIELDS)


def require_compatible(reports: list[dict[str, Any]], paths: list[Path]) -> None:
    first = reports[0]
    expected_dataset = comparable_dataset(first)
    expected_provenance = comparable_provenance(first)
    expected_settings = first.get("settings")
    expected_probabilities = comparable_probabilities(first)
    expected_geometry = require_mapping(
        first.get("diagnostic_execution"),
        "diagnostic execution",
    ).get("query_geometry")
    expected_reference = reference_structure(reference_run(first))
    expected_measurements = measurements_by_batch(reference_run(first))

    for report, path in zip(reports[1:], paths[1:]):
        if comparable_provenance(report) != expected_provenance:
            raise RuntimeError(f"report {path} has different provenance")
        if comparable_dataset(report) != expected_dataset:
            raise RuntimeError(f"report {path} describes a different dataset")
        if report.get("settings") != expected_settings:
            raise RuntimeError(f"report {path} uses different benchmark settings")
        if comparable_probabilities(report) != expected_probabilities:
            raise RuntimeError(f"report {path} uses different sampling probabilities")
        geometry = require_mapping(
            report.get("diagnostic_execution"),
            f"diagnostic execution in {path}",
        ).get("query_geometry")
        if geometry != expected_geometry:
            raise RuntimeError(f"report {path} uses different query geometry")
        current_reference = reference_run(report)
        if reference_structure(current_reference) != expected_reference:
            raise RuntimeError(f"report {path} uses a different exact reference")
        current_measurements = measurements_by_batch(current_reference)
        if current_measurements.keys() != expected_measurements.keys():
            raise RuntimeError(f"report {path} uses different reference batches")
        for batch_size, expected in expected_measurements.items():
            current = current_measurements[batch_size]
            if reference_measurement_structure(
                current
            ) != reference_measurement_structure(expected):
                raise RuntimeError(
                    f"report {path} disagrees on reference results at "
                    f"batch size {batch_size}"
                )


def merged_reference(
    references: list[dict[str, Any]],
    query_count: int,
) -> tuple[dict[str, Any], list[float]]:
    result = copy.deepcopy(references[0])
    build_times = [
        require_number(run.get("build_ms"), "reference build time")
        for run in references
    ]
    result["build_ms"] = statistics.median(build_times)
    result["warmups"] = sum(
        run.get("warmups")
        for run in references
        if isinstance(run.get("warmups"), int)
    )

    source_measurements = [measurements_by_batch(run) for run in references]
    merged_measurements: list[dict[str, Any]] = []
    trial_counts: set[int] = set()
    for batch_size in source_measurements[0]:
        measurement = copy.deepcopy(source_measurements[0][batch_size])
        trials: list[float] = []
        for measurements in source_measurements:
            raw_trials = require_list(
                measurements[batch_size].get("trial_ms"),
                "reference trial times",
            )
            trials.extend(
                require_number(value, "reference trial time")
                for value in raw_trials
            )
        if not trials:
            raise RuntimeError("reference measurements contain no trials")
        median_ms = statistics.median(trials)
        measurement["trial_ms"] = trials
        measurement["median_ms"] = median_ms
        measurement["median_microseconds_per_query"] = (
            median_ms * 1000.0 / query_count
        )
        measurement["median_queries_per_second"] = query_count * 1000.0 / median_ms
        merged_measurements.append(measurement)
        trial_counts.add(len(trials))
    if len(trial_counts) != 1:
        raise RuntimeError("reference batches contain different numbers of joined trials")
    result["trials"] = trial_counts.pop()
    result["measurements"] = merged_measurements
    return result, build_times


def portable_path(path: Path) -> str:
    absolute = path.resolve()
    try:
        return absolute.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(absolute)


def numeric_sum(mappings: list[dict[str, Any]], field: str) -> float:
    return sum(require_number(mapping.get(field), field) for mapping in mappings)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_joined_report(
    reports: list[dict[str, Any]],
    paths: list[Path],
    output: Path,
    csv_output: Path,
) -> tuple[dict[str, Any], list[float]]:
    require_compatible(reports, paths)
    dataset = require_mapping(reports[0].get("dataset"), "dataset")
    query_count = dataset.get("query_count_run")
    if isinstance(query_count, bool) or not isinstance(query_count, int) or query_count <= 0:
        raise RuntimeError("dataset query_count_run is not a positive integer")

    references = [reference_run(report) for report in reports]
    joined_reference, reference_build_times = merged_reference(references, query_count)
    joined_runs = [joined_reference]
    seen_names = {str(joined_reference.get("name"))}
    for report, path in zip(reports, paths):
        for raw_run in require_list(report.get("runs"), f"runs in {path}"):
            run = require_mapping(raw_run, f"run in {path}")
            if run.get("reference") is True:
                continue
            name = run.get("name")
            if not isinstance(name, str) or not name:
                raise RuntimeError(f"report {path} contains an unnamed run")
            if name in seen_names:
                raise RuntimeError(f"duplicate non-reference run name: {name}")
            seen_names.add(name)
            joined_runs.append(copy.deepcopy(run))

    result = copy.deepcopy(reports[0])
    result["generated_at_utc"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    result["outputs"] = {
        "json": portable_path(output),
        "csv": portable_path(csv_output),
    }
    result["runs"] = joined_runs
    result_dataset = require_mapping(result.get("dataset"), "joined dataset")
    source_datasets = [
        require_mapping(report.get("dataset"), "source dataset")
        for report in reports
    ]
    result_dataset["load_ms"] = numeric_sum(source_datasets, "load_ms")
    result_dataset["hash_ms"] = numeric_sum(source_datasets, "hash_ms")
    source_probabilities = [
        require_mapping(report.get("sampling_probabilities"), "sampling probabilities")
        for report in reports
    ]
    result_probabilities = require_mapping(
        result.get("sampling_probabilities"),
        "joined sampling probabilities",
    )
    result_probabilities["build_ms"] = numeric_sum(source_probabilities, "build_ms")
    result_probabilities["load_ms"] = numeric_sum(source_probabilities, "load_ms")
    source_diagnostics = [
        require_mapping(report.get("diagnostic_execution"), "diagnostic execution")
        for report in reports
    ]
    result_diagnostics = require_mapping(
        result.get("diagnostic_execution"),
        "joined diagnostic execution",
    )
    result_diagnostics["build_ms"] = numeric_sum(source_diagnostics, "build_ms")
    result_diagnostics["evaluation_ms"] = numeric_sum(
        source_diagnostics,
        "evaluation_ms",
    )
    result_diagnostics["payload_bytes"] = max(
        int(diagnostic.get("payload_bytes", 0))
        for diagnostic in source_diagnostics
    )
    result["joined_checkpoints"] = {
        "source_reports": [portable_path(path) for path in paths],
        "source_setup_files": [report.get("setup_file") for report in reports],
        "reference_build_ms": reference_build_times,
    }
    result_provenance = require_mapping(
        result.get("provenance"), "joined provenance"
    )
    result_provenance["invocation"] = {
        "tool": "scripts/reporting/join_benchmark_reports.py",
        "source_reports": [
            {"path": portable_path(path), "sha256": sha256_file(path)}
            for path in paths
        ],
    }
    return result, reference_build_times


def source_csv_path(report: dict[str, Any], report_path: Path) -> Path:
    outputs = require_mapping(report.get("outputs"), f"outputs in {report_path}")
    value = outputs.get("csv")
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"report {report_path} does not name a CSV output")
    path = Path(value)
    if path.is_absolute():
        return path
    project_path = PROJECT_ROOT / path
    if project_path.is_file():
        return project_path
    return report_path.parent / path.name


def load_csv_rows(
    reports: list[dict[str, Any]],
    report_paths: list[Path],
) -> tuple[list[str], list[dict[str, str]]]:
    header: list[str] | None = None
    rows: list[dict[str, str]] = []
    for report, report_path in zip(reports, report_paths):
        path = source_csv_path(report, report_path)
        try:
            with path.open(encoding="utf-8", newline="") as source:
                reader = csv.DictReader(source)
                current_header = reader.fieldnames
                if current_header is None:
                    raise RuntimeError(f"CSV {path} has no header")
                if header is None:
                    header = current_header
                elif current_header != header:
                    raise RuntimeError(f"CSV {path} uses a different header")
                rows.extend(dict(row) for row in reader)
        except OSError as error:
            raise RuntimeError(f"cannot read CSV {path}: {error}") from error
    if header is None:
        raise RuntimeError("no CSV input was loaded")
    return header, rows


def joined_csv_rows(
    rows: list[dict[str, str]],
    reference_name: str,
    reference_build_ms: float,
    reference_warmups: int,
) -> list[dict[str, str]]:
    reference_rows: list[dict[str, str]] = []
    other_rows: list[dict[str, str]] = []
    for row in rows:
        if row.get("run_name") == reference_name and row.get("reference") in {
            "1",
            "true",
            "True",
        }:
            reference_rows.append(row)
        else:
            other_rows.append(row)
    if not reference_rows:
        raise RuntimeError("source CSV files contain no reference rows")
    next_trial: dict[str, int] = {}
    for row in reference_rows:
        batch = row.get("batch_size", "")
        trial = next_trial.get(batch, 0)
        row["trial"] = str(trial)
        row["build_ms"] = f"{reference_build_ms:.17g}"
        row["warmups"] = str(reference_warmups)
        next_trial[batch] = trial + 1
    return reference_rows + other_rows


def atomic_json_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary = Path(output.name)
            json.dump(value, output, indent=2)
            output.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_csv_write(
    path: Path,
    header: list[str],
    rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary = Path(output.name)
            writer = csv.DictWriter(output, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    try:
        paths: list[Path] = []
        seen_paths: set[Path] = set()
        for raw_path in args.reports:
            path = raw_path.expanduser().resolve()
            if path not in seen_paths:
                paths.append(path)
                seen_paths.add(path)
        if len(paths) < 2:
            raise RuntimeError("at least two distinct reports are required")
        output = args.output.expanduser().resolve()
        csv_output = (
            args.csv_output.expanduser().resolve()
            if args.csv_output is not None
            else output.with_suffix(".csv")
        )
        if output == csv_output:
            raise RuntimeError("JSON and CSV outputs must be different files")
        if output in paths or csv_output in paths:
            raise RuntimeError("a joined output cannot also be a source report")
        existing = [path for path in (output, csv_output) if path.exists()]
        if existing and not args.force:
            raise RuntimeError(
                "joined output already exists; use --force to replace it: "
                + ", ".join(str(path) for path in existing)
            )

        reports = [load_report(path) for path in paths]
        joined, reference_build_times = build_joined_report(
            reports,
            paths,
            output,
            csv_output,
        )
        header, source_rows = load_csv_rows(reports, paths)
        reference = reference_run(joined)
        reference_name = reference.get("name")
        if not isinstance(reference_name, str):
            raise RuntimeError("joined reference has no name")
        reference_warmups = reference.get("warmups")
        if isinstance(reference_warmups, bool) or not isinstance(
            reference_warmups, int
        ):
            raise RuntimeError("joined reference has invalid warmups")
        csv_rows = joined_csv_rows(
            source_rows,
            reference_name,
            statistics.median(reference_build_times),
            reference_warmups,
        )
        atomic_json_write(output, joined)
        atomic_csv_write(csv_output, header, csv_rows)
        print(
            f"Joined {len(paths)} reports and {len(joined['runs'])} runs into "
            f"{output}"
        )
        print(f"Wrote joined CSV to {csv_output}")
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
