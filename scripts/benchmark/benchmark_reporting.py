"""Shared setup parsing and benchmark-report summary utilities."""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
from pathlib import Path


FORMAT_HEADER = "ultrahigh_ann_benchmark_setup_v2"
REPORT_SCHEMA_VERSION = 5
SUMMARY_SCHEMA_VERSION = 1
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_setup(path: Path) -> tuple[Path, list[dict[str, object]]]:
    directives: dict[str, str] = {}
    raw_runs: list[tuple[int, list[str]]] = []
    header_seen = False
    with path.open(encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if not header_seen:
                if line != FORMAT_HEADER:
                    raise RuntimeError(
                        f"{path}:{line_number}: expected {FORMAT_HEADER}"
                    )
                header_seen = True
                continue

            fields = [field.strip() for field in raw_line.rstrip("\r\n").split("\t")]
            if fields[0] == "run":
                if len(fields) < 3 or not fields[1] or not fields[2]:
                    raise RuntimeError(f"{path}:{line_number}: malformed run directive")
                raw_runs.append((line_number, fields))
                continue
            if len(fields) != 2 or not fields[1]:
                raise RuntimeError(
                    f"{path}:{line_number}: {fields[0]} requires one value"
                )
            if fields[0] in directives:
                raise RuntimeError(
                    f"{path}:{line_number}: duplicate directive: {fields[0]}"
                )
            directives[fields[0]] = fields[1]

    if not header_seen:
        raise RuntimeError(f"benchmark setup is empty: {path}")
    for required in ("dataset", "json_output", "csv_output", "reference"):
        if required not in directives:
            raise RuntimeError(f"benchmark setup does not define {required}: {path}")
    if not raw_runs:
        raise RuntimeError(f"benchmark setup does not define runs: {path}")

    def integer(value: str, line_number: int, name: str, *, positive=False) -> int:
        try:
            parsed = int(value)
        except ValueError as error:
            raise RuntimeError(
                f"{path}:{line_number}: {name} must be an integer"
            ) from error
        if parsed < 0 or (positive and parsed == 0):
            qualifier = "positive" if positive else "nonnegative"
            raise RuntimeError(f"{path}:{line_number}: {name} must be {qualifier}")
        return parsed

    def batch_sizes(value: str, line_number: int) -> list[int]:
        parsed = [
            integer(field.strip(), line_number, "batch_sizes", positive=True)
            for field in value.split(",")
        ]
        if not parsed or len(parsed) != len(set(parsed)):
            raise RuntimeError(f"{path}:{line_number}: batch_sizes must be unique")
        return parsed

    global_batches = batch_sizes(directives.get("batch_sizes", "1"), 0)
    global_warmups = integer(directives.get("warmups", "0"), 0, "warmups")
    global_trials = integer(directives.get("trials", "1"), 0, "trials", positive=True)
    reference_name = directives["reference"]
    runs: list[dict[str, object]] = []
    seen_names: set[str] = set()
    allowed_indexes = {"exact", "flat", "uniform", "hierarchical"}
    allowed_parameters = {
        "backend",
        "strategy",
        "batch_sizes",
        "warmups",
        "trials",
        "repetitions",
        "seed",
        "projection_dimension",
    }
    for line_number, fields in raw_runs:
        name, index = fields[1], fields[2]
        if name in seen_names:
            raise RuntimeError(f"{path}:{line_number}: duplicate run name: {name}")
        seen_names.add(name)
        if index not in allowed_indexes:
            raise RuntimeError(f"{path}:{line_number}: unsupported index: {index}")
        parameters: dict[str, str] = {}
        for field in fields[3:]:
            key, separator, value = field.partition("=")
            if not separator or not key or not value:
                raise RuntimeError(f"{path}:{line_number}: malformed run parameter")
            if key in parameters:
                raise RuntimeError(
                    f"{path}:{line_number}: duplicate run parameter: {key}"
                )
            if key not in allowed_parameters:
                raise RuntimeError(
                    f"{path}:{line_number}: unsupported run parameter: {key}"
                )
            parameters[key] = value

        backend = parameters.get("backend", "cpu")
        strategy = parameters.get(
            "strategy", "gemm" if backend == "cuda" else "sequential"
        )
        approximate = index != "exact"
        if approximate and not {"repetitions", "seed"} <= parameters.keys():
            raise RuntimeError(
                f"{path}:{line_number}: approximate run needs repetitions and seed"
            )
        if not approximate and ({"repetitions", "seed"} & parameters.keys()):
            raise RuntimeError(
                f"{path}:{line_number}: exact run cannot set repetitions or seed"
            )
        if index == "hierarchical" and "projection_dimension" not in parameters:
            raise RuntimeError(
                f"{path}:{line_number}: hierarchical run needs projection_dimension"
            )
        if index != "hierarchical" and "projection_dimension" in parameters:
            raise RuntimeError(
                f"{path}:{line_number}: projection_dimension requires hierarchy"
            )
        runs.append(
            {
                "name": name,
                "index": index,
                "backend": backend,
                "strategy": strategy,
                "reference": name == reference_name,
                "repetitions": (
                    integer(
                        parameters["repetitions"],
                        line_number,
                        "repetitions",
                        positive=True,
                    )
                    if approximate
                    else 0
                ),
                "seed": (
                    integer(parameters["seed"], line_number, "seed")
                    if approximate
                    else 0
                ),
                "projection_dimension": (
                    integer(
                        parameters["projection_dimension"],
                        line_number,
                        "projection_dimension",
                        positive=True,
                    )
                    if index == "hierarchical"
                    else 0
                ),
                "warmups": integer(
                    parameters.get("warmups", str(global_warmups)),
                    line_number,
                    "warmups",
                ),
                "trials": integer(
                    parameters.get("trials", str(global_trials)),
                    line_number,
                    "trials",
                    positive=True,
                ),
                "batch_sizes": batch_sizes(
                    parameters.get(
                        "batch_sizes",
                        ",".join(str(value) for value in global_batches),
                    ),
                    line_number,
                ),
            }
        )
    references = [run for run in runs if run["reference"]]
    if len(references) != 1 or references[0]["index"] != "exact":
        raise RuntimeError("benchmark reference must name exactly one exact run")

    configured = Path(directives["json_output"])
    output_path = (
        configured if configured.is_absolute() else path.parent / configured
    ).resolve()
    return output_path, runs


def require_mapping(value: object, description: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RuntimeError(f"batch report {description} is not an object")
    return value


def require_number(mapping: dict[str, object], key: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"batch report field {key!r} is not numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise RuntimeError(f"batch report field {key!r} is not finite")
    return converted


def require_integer(mapping: dict[str, object], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"batch report field {key!r} is not an integer")
    return value


def require_list(value: object, description: str) -> list[object]:
    if not isinstance(value, list):
        raise RuntimeError(f"batch report {description} is not an array")
    return value


def require_optional_number(
    mapping: dict[str, object],
    key: str,
) -> float | None:
    if mapping.get(key) is None:
        return None
    return require_number(mapping, key)


def require_string(mapping: dict[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"batch report field {key!r} is not a nonempty string")
    return value


def require_boolean(mapping: dict[str, object], key: str) -> bool:
    value = mapping.get(key)
    if not isinstance(value, bool):
        raise RuntimeError(f"batch report field {key!r} is not boolean")
    return value


def require_sha256(
    mapping: dict[str, object],
    key: str,
    *,
    optional: bool = False,
) -> str | None:
    value = mapping.get(key)
    if value is None and optional:
        return None
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise RuntimeError(
            f"batch report field {key!r} is not a lowercase SHA-256 digest"
        )
    return value


def validate_report_provenance(report: dict[str, object]) -> None:
    """Validate the schema-5 source/build/input identity contract."""
    provenance = require_mapping(report.get("provenance"), "provenance")
    source = require_mapping(provenance.get("source"), "provenance source")
    require_string(source, "project_version")
    git = require_mapping(source.get("git"), "Git provenance")
    git_available = require_boolean(git, "available")
    commit = git.get("commit")
    dirty = git.get("dirty")
    if git_available:
        if not isinstance(commit, str) or re.fullmatch(
            r"(?:[0-9a-f]{40}|[0-9a-f]{64})", commit
        ) is None:
            raise RuntimeError("batch report Git commit is malformed")
        if not isinstance(dirty, bool):
            raise RuntimeError("batch report Git dirty state is not boolean")
    elif commit is not None or dirty is not None:
        raise RuntimeError(
            "batch report without Git metadata must use null commit and dirty state"
        )

    build = require_mapping(provenance.get("build"), "build provenance")
    require_string(build, "cmake_version")
    require_string(build, "build_type")
    cxx = require_mapping(build.get("cxx"), "C++ build provenance")
    if require_integer(cxx, "standard") != 20:
        raise RuntimeError("batch report was not built as C++20")
    require_string(cxx, "compiler_id")
    require_string(cxx, "compiler_version")
    if not isinstance(cxx.get("flags"), str):
        raise RuntimeError("batch report C++ flags are not a string")

    cuda = require_mapping(build.get("cuda"), "CUDA build provenance")
    cuda_enabled = require_boolean(cuda, "enabled")
    cuda_fields = ("compiler_id", "compiler_version", "architectures", "flags")
    if cuda_enabled:
        for field in cuda_fields:
            require_string(cuda, field)
    elif any(cuda.get(field) is not None for field in cuda_fields):
        raise RuntimeError(
            "CPU-only batch report must use null CUDA compiler fields"
        )

    host = require_mapping(provenance.get("host"), "host provenance")
    operating_system = require_mapping(
        host.get("operating_system"), "operating-system provenance"
    )
    for field in ("name", "version", "architecture"):
        require_string(operating_system, field)
    processor_model = host.get("processor_model")
    if processor_model is not None and not isinstance(processor_model, str):
        raise RuntimeError("batch report processor model is malformed")
    memory = host.get("physical_memory_bytes")
    if memory is not None and (
        isinstance(memory, bool) or not isinstance(memory, int) or memory <= 0
    ):
        raise RuntimeError("batch report physical memory is malformed")

    cuda_runtime = provenance.get("cuda_runtime")
    if cuda_runtime is not None:
        runtime = require_mapping(cuda_runtime, "CUDA runtime provenance")
        device = require_integer(runtime, "device")
        if device < 0:
            raise RuntimeError("batch report CUDA device is negative")
        require_string(runtime, "device_name")
        require_string(runtime, "compute_capability")
        memory = require_integer(runtime, "total_global_memory_bytes")
        if memory <= 0:
            raise RuntimeError("batch report CUDA memory is not positive")
        for field in ("compiled_runtime", "runtime", "driver"):
            version = require_mapping(runtime.get(field), f"CUDA {field} version")
            if require_integer(version, "encoded") <= 0:
                raise RuntimeError(f"batch report CUDA {field} version is invalid")
            require_string(version, "version")

    invocation = require_mapping(provenance.get("invocation"), "invocation")
    arguments = require_list(invocation.get("arguments"), "invocation arguments")
    if not arguments or any(
        not isinstance(value, str) or not value for value in arguments
    ):
        raise RuntimeError("batch report invocation arguments are malformed")
    require_sha256(invocation, "setup_sha256")

    dataset = require_mapping(report.get("dataset"), "dataset")
    representatives_sha256 = require_sha256(dataset, "representatives_sha256")
    require_sha256(dataset, "queries_sha256")
    labels_available = require_boolean(dataset, "labels_available")
    representative_labels_sha256 = require_sha256(
        dataset, "representative_labels_sha256", optional=True
    )
    query_labels_sha256 = require_sha256(
        dataset, "query_labels_sha256", optional=True
    )
    if labels_available != (representative_labels_sha256 is not None):
        raise RuntimeError(
            "batch report representative-label checksum availability is inconsistent"
        )
    if labels_available != (query_labels_sha256 is not None):
        raise RuntimeError(
            "batch report query-label checksum availability is inconsistent"
        )

    probabilities = require_mapping(
        report.get("sampling_probabilities"), "sampling probabilities"
    )
    probabilities_required = require_boolean(probabilities, "required")
    probability_source = probabilities.get("source_file")
    if probability_source is not None and not isinstance(probability_source, str):
        raise RuntimeError("batch report probability source is malformed")
    source_sha256 = require_sha256(probabilities, "source_sha256", optional=True)
    if (probability_source is None) != (source_sha256 is None):
        raise RuntimeError(
            "batch report probability source and checksum are inconsistent"
        )
    probability_representatives_sha256 = require_sha256(
        probabilities, "representatives_sha256", optional=True
    )
    if probabilities_required:
        if probability_representatives_sha256 != representatives_sha256:
            raise RuntimeError(
                "batch report probability file is not bound to the dataset"
            )
    elif probability_representatives_sha256 is not None:
        raise RuntimeError(
            "batch report without sampling probabilities has a source binding"
        )


def approximation_summary_fields(
    measurement: dict[str, object],
) -> dict[str, object]:
    approximation = require_mapping(
        measurement.get("approximation"),
        "approximation diagnostics",
    )
    distance_ratio = require_mapping(
        approximation.get("distance_ratio"),
        "distance ratio diagnostics",
    )
    raw_returned_rank = approximation.get("returned_representative_rank")
    returned_rank = (
        None
        if raw_returned_rank is None
        else require_mapping(raw_returned_rank, "returned rank diagnostics")
    )
    failures = approximation.get("approximation_guarantee_failures")
    if not isinstance(failures, list):
        raise RuntimeError("approximation failures must be an array")
    failure_rates: dict[float, float] = {}
    for raw_failure in failures:
        failure = require_mapping(raw_failure, "approximation failure")
        failure_rates[require_number(failure, "epsilon")] = require_number(
            failure,
            "violation_rate",
        )

    fields: dict[str, object] = {
        "optimal_representative_count": require_integer(
            approximation,
            "distance_optimal_count",
        ),
        "optimal_representative_rate": require_number(
            approximation,
            "distance_optimal_rate",
        ),
        "non_optimal_count": require_integer(
            approximation,
            "non_optimal_count",
        ),
        "non_optimal_rate": require_number(
            approximation,
            "non_optimal_rate",
        ),
        "ratio_eligible_count": require_integer(
            distance_ratio,
            "count",
        ),
        "zero_optimum_queries_excluded": require_integer(
            approximation,
            "zero_optimum_query_count",
        ),
        "mean_distance_ratio": require_optional_number(
            distance_ratio,
            "mean",
        ),
        "median_distance_ratio": require_optional_number(
            distance_ratio,
            "median",
        ),
        "distance_ratio_percentile_95": require_optional_number(
            distance_ratio,
            "percentile_95",
        ),
        "distance_ratio_percentile_99": require_optional_number(
            distance_ratio,
            "percentile_99",
        ),
        "maximum_distance_ratio": require_optional_number(
            distance_ratio,
            "maximum",
        ),
        "conditional_non_optimal_ratio_count": require_integer(
            approximation,
            "non_optimal_ratio_count",
        ),
        "zero_optimum_non_optimal_count": require_integer(
            approximation,
            "zero_optimum_non_optimal_count",
        ),
        "conditional_mean_distance_ratio": require_optional_number(
            approximation,
            "non_optimal_distance_ratio_mean",
        ),
        "conditional_mean_relative_excess": require_optional_number(
            approximation,
            "non_optimal_relative_excess_mean",
        ),
        "returned_rank_mean": (
            require_optional_number(returned_rank, "mean")
            if returned_rank is not None
            else None
        ),
        "returned_rank_percentile_95": (
            require_optional_number(returned_rank, "percentile_95")
            if returned_rank is not None
            else None
        ),
        "returned_rank_maximum": (
            require_optional_number(returned_rank, "maximum")
            if returned_rank is not None
            else None
        ),
    }
    conventional_suffixes = (
        (0.001, "0_001"),
        (0.005, "0_005"),
        (0.01, "0_01"),
        (0.02, "0_02"),
        (0.05, "0_05"),
        (0.10, "0_10"),
        (0.20, "0_20"),
    )
    for epsilon, rate in sorted(failure_rates.items()):
        suffix = next(
            (
                candidate
                for expected, candidate in conventional_suffixes
                if math.isclose(epsilon, expected)
            ),
            format(epsilon, ".12g").replace(".", "_").replace("-", "m"),
        )
        fields[f"approximation_failure_rate_epsilon_{suffix}"] = rate
    return fields


def report_runs(report: dict[str, object]) -> list[dict[str, object]]:
    return [
        require_mapping(value, "run")
        for value in require_list(report.get("runs"), "runs")
    ]


def reference_run(report: dict[str, object]) -> dict[str, object]:
    references = [run for run in report_runs(report) if run.get("reference") is True]
    if len(references) != 1:
        raise RuntimeError("batch report must contain exactly one reference run")
    return references[0]


def measurements_by_batch(run: dict[str, object]) -> dict[int, dict[str, object]]:
    measurements: dict[int, dict[str, object]] = {}
    for raw_measurement in require_list(run.get("measurements"), "measurements"):
        measurement = require_mapping(raw_measurement, "measurement")
        batch_size = require_integer(measurement, "batch_size")
        if batch_size in measurements:
            raise RuntimeError(
                f"run {run.get('name')!r} repeats batch size {batch_size}"
            )
        measurements[batch_size] = measurement
    if not measurements:
        raise RuntimeError(f"run {run.get('name')!r} has no measurements")
    return measurements


def load_and_validate_report(
    report_path: Path,
    setup_path: Path,
    configured_runs: list[dict[str, object]],
) -> dict[str, object]:
    try:
        with report_path.open(encoding="utf-8") as source:
            report = json.load(source)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"cannot read batch report {report_path}: {error}"
        ) from error
    report = require_mapping(report, "root")
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise RuntimeError(
            f"benchmark sweep requires a schema-version {REPORT_SCHEMA_VERSION} "
            "batch report"
        )
    validate_report_provenance(report)
    reported_setup = Path(str(report.get("setup_file", "")))
    if not reported_setup.is_absolute():
        reported_setup = PROJECT_ROOT / reported_setup
    reported_setup = reported_setup.resolve()
    if reported_setup != setup_path:
        raise RuntimeError(
            f"report was generated from {reported_setup}, not {setup_path}"
        )
    observed = []
    for run in report_runs(report):
        observed.append(
            {
                "name": run.get("name"),
                "index": run.get("index"),
                "backend": run.get("backend"),
                "strategy": run.get("strategy"),
                "reference": run.get("reference"),
                "repetitions": run.get("repetitions"),
                "seed": run.get("seed"),
                "projection_dimension": run.get("projection_dimension"),
                "warmups": run.get("warmups"),
                "trials": run.get("trials"),
                "batch_sizes": list(measurements_by_batch(run)),
            }
        )
    if observed != configured_runs:
        raise RuntimeError(
            "existing batch report does not match the setup; use --force "
            "to regenerate it"
        )
    reference = reference_run(report)
    if reference.get("index") != "exact":
        raise RuntimeError("batch report reference run is not exact")
    require_mapping(report.get("dataset"), "dataset")
    require_mapping(report.get("settings"), "settings")
    require_mapping(report.get("sampling_probabilities"), "sampling probabilities")
    require_mapping(report.get("diagnostic_execution"), "diagnostic execution")
    return report


def summarize_metric(
    rows: list[dict[str, object]],
    key: str,
) -> dict[str, float | int | None]:
    values = [float(row[key]) for row in rows if row[key] is not None]
    if not values:
        return {
            "observation_count": 0,
            "mean": None,
            "sample_standard_deviation": None,
            "minimum": None,
            "maximum": None,
        }
    return {
        "observation_count": len(values),
        "mean": statistics.mean(values),
        "sample_standard_deviation": (
            statistics.stdev(values) if len(values) > 1 else 0.0
        ),
        "minimum": min(values),
        "maximum": max(values),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def flatten_direct_runs(
    report: dict[str, object],
    method: str,
) -> list[dict[str, object]]:
    if method not in ("flat", "uniform"):
        raise ValueError(f"cannot create direct-sampling summaries for {method!r}")
    dataset = require_mapping(report.get("dataset"), "dataset")
    dimension = require_integer(dataset, "dimension")
    query_count = require_integer(dataset, "query_count_run")
    exact = reference_run(report)
    exact_space = require_mapping(exact.get("space"), "exact space")
    exact_measurements = measurements_by_batch(exact)
    exact_build_ms = require_number(exact, "build_ms")
    exact_index_bytes = require_integer(exact_space, "index_payload_bytes")
    rows: list[dict[str, object]] = []
    for run in report_runs(report):
        if run.get("index") != method or run.get("reference") is True:
            continue
        space = require_mapping(run.get("space"), "run space")
        repetitions = require_integer(run, "repetitions")
        build_ms = require_number(run, "build_ms")
        index_bytes = require_integer(space, "index_payload_bytes")
        unique_coordinates = require_integer(space, "unique_coordinates")
        multiplicity = require_integer(space, "sampled_multiplicity")
        for batch_size, measurement in measurements_by_batch(run).items():
            if batch_size not in exact_measurements:
                raise RuntimeError(
                    f"reference run omits batch size {batch_size} used by "
                    f"{run.get('name')!r}"
                )
            exact_measurement = exact_measurements[batch_size]
            exact_accuracy = require_number(exact_measurement, "accuracy")
            exact_query_us = require_number(
                exact_measurement, "median_microseconds_per_query"
            )
            exact_query_ms = require_number(exact_measurement, "median_ms")
            accuracy = require_number(measurement, "accuracy")
            agreement = require_number(measurement, "exact_choice_agreement")
            label_agreement = (
                require_number(measurement, "label_agreement_with_exact")
                if measurement.get("label_agreement_with_exact") is not None
                else agreement
            )
            query_us = require_number(measurement, "median_microseconds_per_query")
            query_ms = require_number(measurement, "median_ms")
            agreement_count = require_integer(
                measurement, "exact_choice_agreement_count"
            )
            rows.append(
                {
                    "name": run.get("name"),
                    "method": method,
                    "backend": run.get("backend"),
                    "strategy": run.get("strategy"),
                    "batch_size": batch_size,
                    "repetitions": repetitions,
                    "seed": require_integer(run, "seed"),
                    "query_count": query_count,
                    "dimension": dimension,
                    "representative_count": require_integer(
                        dataset,
                        "representative_count",
                    ),
                    "exact_correct": require_integer(exact_measurement, "correct"),
                    f"{method}_correct": require_integer(measurement, "correct"),
                    "exact_accuracy": exact_accuracy,
                    f"{method}_accuracy": accuracy,
                    "accuracy_delta": accuracy - exact_accuracy,
                    "exact_agreement": agreement,
                    "exact_label_agreement": label_agreement,
                    "agreement_count": agreement_count,
                    "disagreement_count": query_count - agreement_count,
                    "exact_query_us": exact_query_us,
                    f"{method}_query_us": query_us,
                    "query_speedup": exact_query_us / query_us,
                    "exact_build_ms": exact_build_ms,
                    f"{method}_build_ms": build_ms,
                    "exact_end_to_end_ms": exact_build_ms + exact_query_ms,
                    f"{method}_end_to_end_ms": build_ms + query_ms,
                    "unique_coordinates": unique_coordinates,
                    "dimension_fraction": unique_coordinates / dimension,
                    "sampled_multiplicity": multiplicity,
                    "sampling_mass_estimate": multiplicity / repetitions,
                    "exact_index_bytes": exact_index_bytes,
                    f"{method}_index_bytes": index_bytes,
                    # A Bernoulli/Poisson coordinate sample can legitimately
                    # be empty at very small T.
                    "index_compression": (
                        exact_index_bytes / index_bytes if index_bytes else None
                    ),
                    **approximation_summary_fields(measurement),
                }
            )
    if not rows:
        raise RuntimeError(f"batch report does not contain any {method} runs")
    return rows


def flatten_flat_runs(
    report: dict[str, object],
) -> list[dict[str, object]]:
    return flatten_direct_runs(report, "flat")


def aggregate_direct_runs(
    rows: list[dict[str, object]],
    method: str,
) -> list[dict[str, object]]:
    base_metric_names = (
        "exact_agreement",
        "exact_label_agreement",
        f"{method}_accuracy",
        "accuracy_delta",
        f"{method}_query_us",
        "query_speedup",
        "unique_coordinates",
        "dimension_fraction",
        "sampled_multiplicity",
        "sampling_mass_estimate",
        f"{method}_index_bytes",
        "index_compression",
        "optimal_representative_rate",
        "non_optimal_rate",
        "mean_distance_ratio",
        "median_distance_ratio",
        "distance_ratio_percentile_95",
        "distance_ratio_percentile_99",
        "maximum_distance_ratio",
        "conditional_mean_distance_ratio",
        "conditional_mean_relative_excess",
        "returned_rank_mean",
        "returned_rank_percentile_95",
        "returned_rank_maximum",
    )
    failure_metrics = sorted(
        {
            key
            for row in rows
            for key in row
            if key.startswith("approximation_failure_rate_epsilon_")
        }
    )
    metric_names = (*base_metric_names, *failure_metrics)
    aggregates: list[dict[str, object]] = []
    groups = sorted(
        {
            (
                int(row["repetitions"]),
                int(row["batch_size"]),
                str(row["backend"]),
                str(row["strategy"]),
            )
            for row in rows
        }
    )
    for repetitions, batch_size, backend, strategy in groups:
        group = [
            row
            for row in rows
            if row["repetitions"] == repetitions
            and row["batch_size"] == batch_size
            and row["backend"] == backend
            and row["strategy"] == strategy
        ]
        aggregates.append(
            {
                "repetitions": repetitions,
                "batch_size": batch_size,
                "backend": backend,
                "strategy": strategy,
                "run_count": len(group),
                "seeds": [int(row["seed"]) for row in group],
                "metrics": {
                    metric: summarize_metric(group, metric) for metric in metric_names
                },
            }
        )
    return aggregates


def aggregate_flat_runs(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return aggregate_direct_runs(rows, "flat")


def direct_aggregate_csv_rows(
    aggregates: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for aggregate in aggregates:
        metrics = require_mapping(aggregate["metrics"], "aggregate metrics")
        row: dict[str, object] = {
            "repetitions": aggregate["repetitions"],
            "batch_size": aggregate["batch_size"],
            "backend": aggregate["backend"],
            "strategy": aggregate["strategy"],
            "run_count": aggregate["run_count"],
            "seeds": ";".join(str(seed) for seed in aggregate["seeds"]),
        }
        for metric_name, raw_summary in metrics.items():
            summary = require_mapping(raw_summary, "metric summary")
            for statistic, value in summary.items():
                row[f"{metric_name}_{statistic}"] = value
        rows.append(row)
    return rows


def write_direct_summaries(
    report_path: Path,
    rows: list[dict[str, object]],
    method: str,
    report: dict[str, object] | None = None,
    filename_suffix: str = "",
) -> None:
    aggregates = aggregate_direct_runs(rows, method)
    stem = report_path.with_suffix("")
    suffix = f"_{filename_suffix}" if filename_suffix else ""
    summary_path = stem.parent / f"{stem.name}{suffix}_summary.json"
    runs_path = stem.parent / f"{stem.name}{suffix}_runs.csv"
    aggregates_path = stem.parent / f"{stem.name}{suffix}_aggregates.csv"
    summary = {
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "source_report_schema_version": REPORT_SCHEMA_VERSION,
        "source_report": str(report_path),
        "method": method,
        "runs": rows,
        "aggregates": aggregates,
    }
    if report is not None and isinstance(report.get("sampling_probabilities"), dict):
        summary["sampling_probabilities"] = report["sampling_probabilities"]
    if report is not None and isinstance(report.get("diagnostic_execution"), dict):
        summary["diagnostic_execution"] = report["diagnostic_execution"]
    with summary_path.open("w", encoding="utf-8") as output:
        json.dump(summary, output, indent=2, sort_keys=True)
        output.write("\n")
    write_csv(runs_path, rows)
    write_csv(aggregates_path, direct_aggregate_csv_rows(aggregates))


def write_flat_summaries(
    report_path: Path,
    rows: list[dict[str, object]],
    report: dict[str, object] | None = None,
    filename_suffix: str = "",
) -> None:
    write_direct_summaries(
        report_path,
        rows,
        "flat",
        report,
        filename_suffix,
    )


def write_uniform_summaries(
    report_path: Path,
    rows: list[dict[str, object]],
    report: dict[str, object] | None = None,
    filename_suffix: str = "",
) -> None:
    write_direct_summaries(
        report_path,
        rows,
        "uniform",
        report,
        filename_suffix,
    )


def print_direct_aggregate_table(
    rows: list[dict[str, object]],
    method: str,
) -> None:
    print(f"\n{method.capitalize()} sweep aggregate means:")
    print(
        f"{'T':>7} {'B':>5} {'prototype':>11} {'class':>9} {'accuracy':>10} "
        f"{'genes':>10} {'dimension':>11} {'query us':>11} {'speedup':>9}"
    )
    for aggregate in aggregate_direct_runs(rows, method):
        metrics = require_mapping(aggregate["metrics"], "aggregate metrics")
        agreement = require_mapping(
            metrics["exact_agreement"],
            "agreement summary",
        )["mean"]
        accuracy = require_mapping(
            metrics[f"{method}_accuracy"],
            "accuracy summary",
        )["mean"]
        label_agreement = require_mapping(
            metrics["exact_label_agreement"],
            "label agreement summary",
        )["mean"]
        coordinates = require_mapping(
            metrics["unique_coordinates"],
            "coordinate summary",
        )["mean"]
        dimension = require_mapping(
            metrics["dimension_fraction"],
            "dimension summary",
        )["mean"]
        query_us = require_mapping(
            metrics[f"{method}_query_us"],
            "query summary",
        )["mean"]
        speedup = require_mapping(
            metrics["query_speedup"],
            "speedup summary",
        )["mean"]
        print(
            f"{aggregate['repetitions']:7d} "
            f"{aggregate['batch_size']:5d} "
            f"{100.0 * float(agreement):10.2f}% "
            f"{100.0 * float(label_agreement):8.2f}% "
            f"{100.0 * float(accuracy):9.2f}% "
            f"{float(coordinates):10.1f} "
            f"{100.0 * float(dimension):10.2f}% "
            f"{float(query_us):11.2f} {float(speedup):8.2f}x"
        )


def print_flat_aggregate_table(rows: list[dict[str, object]]) -> None:
    print_direct_aggregate_table(rows, "flat")


def print_uniform_aggregate_table(rows: list[dict[str, object]]) -> None:
    print_direct_aggregate_table(rows, "uniform")


def run_identity(run: dict[str, object]) -> tuple[int, int]:
    return (
        require_integer(run, "repetitions"),
        require_integer(run, "seed"),
    )


def flatten_hierarchical_comparisons(
    report: dict[str, object],
    expected_distance: str | None = None,
) -> list[dict[str, object]]:
    report_settings = require_mapping(report.get("settings"), "settings")
    distance = report_settings.get("distance")
    if distance not in ("l1", "l2"):
        raise RuntimeError("hierarchical sweep report distance is neither l1 nor l2")
    if expected_distance is not None and distance != expected_distance:
        raise RuntimeError(
            "hierarchical sweep report distance is not " f"{expected_distance}"
        )

    dataset = require_mapping(report.get("dataset"), "dataset")
    exact = reference_run(report)
    exact_measurements = measurements_by_batch(exact)
    representative_count = require_integer(dataset, "representative_count")
    dimension = require_integer(dataset, "dimension")
    query_count = require_integer(dataset, "query_count_run")

    runs = report_runs(report)
    flat_runs: dict[tuple[int, int, str, str], dict[str, object]] = {}
    for run in runs:
        if run.get("index") != "flat":
            continue
        repetitions, seed = run_identity(run)
        backend = str(run.get("backend"))
        strategy = str(run.get("strategy"))
        key = (repetitions, seed, backend, strategy)
        if key in flat_runs:
            raise RuntimeError(
                "batch report contains duplicate matched flat runs for "
                f"T={repetitions}, seed={seed}, backend={backend}, "
                f"strategy={strategy}"
            )
        flat_runs[key] = run

    rows: list[dict[str, object]] = []
    for hierarchical_run in runs:
        if hierarchical_run.get("index") != "hierarchical":
            continue
        repetitions, seed = run_identity(hierarchical_run)
        backend = str(hierarchical_run.get("backend"))
        strategy = str(hierarchical_run.get("strategy"))
        flat_run = flat_runs.get((repetitions, seed, backend, strategy))
        if flat_run is None:
            raise RuntimeError(
                "hierarchical run has no matched flat run for "
                f"T={repetitions}, seed={seed}, backend={backend}, "
                f"strategy={strategy}"
            )
        projection_dimension = require_integer(
            hierarchical_run,
            "projection_dimension",
        )
        flat_space = require_mapping(flat_run.get("space"), "flat space")
        hierarchical_space = require_mapping(
            hierarchical_run.get("space"),
            "hierarchical space",
        )
        coordinate_keys = ("unique_coordinates", "sampled_multiplicity")
        for key in coordinate_keys:
            if require_integer(flat_space, key) != require_integer(
                hierarchical_space,
                key,
            ):
                raise RuntimeError(
                    "matched flat and hierarchical sampling differs for "
                    f"T={repetitions}, seed={seed}"
                )

        flat_index_bytes = require_integer(flat_space, "index_payload_bytes")
        hierarchical_index_bytes = require_integer(
            hierarchical_space,
            "index_payload_bytes",
        )
        flat_measurements = measurements_by_batch(flat_run)
        for batch_size, hierarchical in measurements_by_batch(hierarchical_run).items():
            flat = flat_measurements.get(batch_size)
            if flat is None:
                raise RuntimeError(
                    "hierarchical run has no matched flat measurement for "
                    f"T={repetitions}, seed={seed}, batch={batch_size}"
                )
            exact_measurement = exact_measurements.get(batch_size)
            if exact_measurement is None:
                raise RuntimeError(f"reference run omits batch size {batch_size}")
            flat_accuracy = require_number(flat, "accuracy")
            hierarchical_accuracy = require_number(hierarchical, "accuracy")
            flat_agreement = require_number(flat, "exact_choice_agreement")
            hierarchical_agreement = require_number(
                hierarchical,
                "exact_choice_agreement",
            )
            flat_label_agreement = (
                require_number(flat, "label_agreement_with_exact")
                if flat.get("label_agreement_with_exact") is not None
                else flat_agreement
            )
            hierarchical_label_agreement = (
                require_number(hierarchical, "label_agreement_with_exact")
                if hierarchical.get("label_agreement_with_exact") is not None
                else hierarchical_agreement
            )
            flat_query_us = require_number(flat, "median_microseconds_per_query")
            hierarchical_query_us = require_number(
                hierarchical,
                "median_microseconds_per_query",
            )
            exact_query_us = require_number(
                exact_measurement, "median_microseconds_per_query"
            )
            unique_coordinates = require_integer(
                hierarchical_space,
                "unique_coordinates",
            )
            flat_approximation = {
                f"flat_{key}": value
                for key, value in approximation_summary_fields(flat).items()
            }
            hierarchical_approximation = {
                f"hierarchical_{key}": value
                for key, value in approximation_summary_fields(hierarchical).items()
            }
            rows.append(
                {
                    "name": hierarchical_run.get("name"),
                    "matched_flat_name": flat_run.get("name"),
                    "backend": hierarchical_run.get("backend"),
                    "strategy": hierarchical_run.get("strategy"),
                    "batch_size": batch_size,
                    "repetitions": repetitions,
                    "projection_dimension": projection_dimension,
                    "seed": seed,
                    "query_count": query_count,
                    "representative_count": representative_count,
                    "dimension": dimension,
                    "exact_accuracy": require_number(exact_measurement, "accuracy"),
                    "flat_accuracy": flat_accuracy,
                    "hierarchical_accuracy": hierarchical_accuracy,
                    "hierarchical_accuracy_delta_vs_flat": (
                        hierarchical_accuracy - flat_accuracy
                    ),
                    "flat_agreement_with_exact": flat_agreement,
                    "hierarchical_agreement_with_exact": hierarchical_agreement,
                    "hierarchical_agreement_delta_vs_flat": (
                        hierarchical_agreement - flat_agreement
                    ),
                    "flat_label_agreement_with_exact": flat_label_agreement,
                    "hierarchical_label_agreement_with_exact": (
                        hierarchical_label_agreement
                    ),
                    "hierarchical_label_agreement_delta_vs_flat": (
                        hierarchical_label_agreement - flat_label_agreement
                    ),
                    "exact_query_us": exact_query_us,
                    "flat_query_us": flat_query_us,
                    "hierarchical_query_us": hierarchical_query_us,
                    "hierarchical_speedup_vs_flat": (
                        flat_query_us / hierarchical_query_us
                    ),
                    "hierarchical_speedup_vs_exact": (
                        exact_query_us / hierarchical_query_us
                    ),
                    "flat_build_ms": require_number(flat_run, "build_ms"),
                    "hierarchical_build_ms": require_number(
                        hierarchical_run,
                        "build_ms",
                    ),
                    "unique_coordinates": unique_coordinates,
                    "dimension_fraction": unique_coordinates / dimension,
                    "sampled_multiplicity": require_integer(
                        hierarchical_space,
                        "sampled_multiplicity",
                    ),
                    "flat_index_bytes": flat_index_bytes,
                    "hierarchical_index_bytes": hierarchical_index_bytes,
                    "hierarchical_index_size_ratio_vs_flat": (
                        hierarchical_index_bytes / flat_index_bytes
                        if flat_index_bytes
                        else None
                    ),
                    "flat_workspace_bytes": require_integer(
                        flat,
                        "workspace_payload_bytes",
                    ),
                    "hierarchical_workspace_bytes": require_integer(
                        hierarchical,
                        "workspace_payload_bytes",
                    ),
                    **flat_approximation,
                    **hierarchical_approximation,
                }
            )

    if not rows:
        raise RuntimeError("batch report does not contain hierarchical runs")
    return rows


def aggregate_hierarchical_rows(
    rows: list[dict[str, object]]
) -> list[dict[str, object]]:
    base_metric_names = (
        "hierarchical_accuracy",
        "hierarchical_accuracy_delta_vs_flat",
        "hierarchical_agreement_with_exact",
        "hierarchical_agreement_delta_vs_flat",
        "hierarchical_label_agreement_with_exact",
        "hierarchical_label_agreement_delta_vs_flat",
        "hierarchical_query_us",
        "hierarchical_speedup_vs_flat",
        "hierarchical_speedup_vs_exact",
        "hierarchical_build_ms",
        "unique_coordinates",
        "dimension_fraction",
        "hierarchical_index_bytes",
        "hierarchical_index_size_ratio_vs_flat",
        "hierarchical_optimal_representative_rate",
        "hierarchical_non_optimal_rate",
        "hierarchical_mean_distance_ratio",
        "hierarchical_median_distance_ratio",
        "hierarchical_distance_ratio_percentile_95",
        "hierarchical_distance_ratio_percentile_99",
        "hierarchical_maximum_distance_ratio",
        "hierarchical_conditional_mean_distance_ratio",
        "hierarchical_conditional_mean_relative_excess",
        "hierarchical_returned_rank_mean",
        "hierarchical_returned_rank_percentile_95",
        "hierarchical_returned_rank_maximum",
    )
    failure_metrics = sorted(
        {
            key
            for row in rows
            for key in row
            if key.startswith("hierarchical_approximation_failure_rate_epsilon_")
        }
    )
    metric_names = (*base_metric_names, *failure_metrics)
    groups = sorted(
        {
            (
                int(row["repetitions"]),
                int(row["projection_dimension"]),
                int(row["batch_size"]),
                str(row["backend"]),
                str(row["strategy"]),
            )
            for row in rows
        }
    )
    aggregates: list[dict[str, object]] = []
    for repetitions, projection_dimension, batch_size, backend, strategy in groups:
        group = [
            row
            for row in rows
            if row["repetitions"] == repetitions
            and row["projection_dimension"] == projection_dimension
            and row["batch_size"] == batch_size
            and row["backend"] == backend
            and row["strategy"] == strategy
        ]
        aggregates.append(
            {
                "repetitions": repetitions,
                "projection_dimension": projection_dimension,
                "batch_size": batch_size,
                "backend": backend,
                "strategy": strategy,
                "run_count": len(group),
                "seeds": [int(row["seed"]) for row in group],
                "metrics": {
                    metric: summarize_metric(group, metric) for metric in metric_names
                },
            }
        )
    return aggregates


def hierarchical_aggregate_csv_rows(
    aggregates: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for aggregate in aggregates:
        metrics = require_mapping(aggregate["metrics"], "aggregate metrics")
        row: dict[str, object] = {
            "repetitions": aggregate["repetitions"],
            "projection_dimension": aggregate["projection_dimension"],
            "batch_size": aggregate["batch_size"],
            "backend": aggregate["backend"],
            "strategy": aggregate["strategy"],
            "run_count": aggregate["run_count"],
            "seeds": ";".join(str(seed) for seed in aggregate["seeds"]),
        }
        for metric_name, raw_summary in metrics.items():
            summary = require_mapping(raw_summary, "metric summary")
            for statistic, value in summary.items():
                row[f"{metric_name}_{statistic}"] = value
        rows.append(row)
    return rows


def write_hierarchical_summaries(
    report_path: Path,
    rows: list[dict[str, object]],
    report: dict[str, object] | None = None,
) -> None:
    aggregates = aggregate_hierarchical_rows(rows)
    stem = report_path.with_suffix("")
    summary_path = stem.parent / f"{stem.name}_summary.json"
    comparisons_path = stem.parent / f"{stem.name}_comparisons.csv"
    aggregates_path = stem.parent / f"{stem.name}_aggregates.csv"
    summary = {
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "source_report_schema_version": REPORT_SCHEMA_VERSION,
        "source_report": str(report_path),
        "comparison": "hierarchical versus matched flat with identical T and seed",
        "runs": rows,
        "aggregates": aggregates,
    }
    if report is not None and isinstance(report.get("sampling_probabilities"), dict):
        summary["sampling_probabilities"] = report["sampling_probabilities"]
    if report is not None and isinstance(report.get("diagnostic_execution"), dict):
        summary["diagnostic_execution"] = report["diagnostic_execution"]
    with summary_path.open("w", encoding="utf-8") as output:
        json.dump(summary, output, indent=2, sort_keys=True)
        output.write("\n")
    write_csv(comparisons_path, rows)
    write_csv(aggregates_path, hierarchical_aggregate_csv_rows(aggregates))


def hierarchical_mean_metric(aggregate: dict[str, object], metric: str) -> float:
    metrics = require_mapping(aggregate["metrics"], "aggregate metrics")
    summary = require_mapping(metrics[metric], f"{metric} summary")
    return float(summary["mean"])


def print_hierarchical_aggregate_table(
    rows: list[dict[str, object]],
    distance_label: str = "Metric",
) -> None:
    print(f"\n{distance_label} hierarchical sweep aggregate means:")
    print(
        f"{'T':>6} {'p':>4} {'B':>5} {'class agr':>11} {'delta flat':>11} "
        f"{'prototype':>10} "
        f"{'accuracy':>10} {'query us':>10} {'vs flat':>9} {'vs exact':>9}"
    )
    for aggregate in aggregate_hierarchical_rows(rows):
        class_agreement = hierarchical_mean_metric(
            aggregate, "hierarchical_label_agreement_with_exact"
        )
        class_delta = hierarchical_mean_metric(
            aggregate, "hierarchical_label_agreement_delta_vs_flat"
        )
        prototype_agreement = hierarchical_mean_metric(
            aggregate, "hierarchical_agreement_with_exact"
        )
        accuracy = hierarchical_mean_metric(aggregate, "hierarchical_accuracy")
        query_us = hierarchical_mean_metric(aggregate, "hierarchical_query_us")
        flat_speedup = hierarchical_mean_metric(
            aggregate, "hierarchical_speedup_vs_flat"
        )
        exact_speedup = hierarchical_mean_metric(
            aggregate, "hierarchical_speedup_vs_exact"
        )
        print(
            f"{aggregate['repetitions']:6d} "
            f"{aggregate['projection_dimension']:4d} "
            f"{aggregate['batch_size']:5d} "
            f"{100.0 * class_agreement:10.2f}% "
            f"{100.0 * class_delta:+10.2f} "
            f"{100.0 * prototype_agreement:9.2f}% "
            f"{100.0 * accuracy:9.2f}% "
            f"{query_us:10.2f} "
            f"{flat_speedup:8.2f}x "
            f"{exact_speedup:8.2f}x"
        )
