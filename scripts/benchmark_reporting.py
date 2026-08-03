"""Shared setup parsing and benchmark-report summary utilities."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path


FORMAT_HEADER = "ultrahigh_ann_benchmark_setup_v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_setup(path: Path) -> tuple[Path, list[dict[str, object]]]:
    output_path: Path | None = None
    runs: list[dict[str, object]] = []
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
            if fields[0] == "output":
                if len(fields) != 2 or not fields[1] or output_path is not None:
                    raise RuntimeError(
                        f"{path}:{line_number}: invalid or duplicate output"
                    )
                configured = Path(fields[1])
                output_path = (
                    configured if configured.is_absolute() else path.parent / configured
                ).resolve()
            elif fields[0] == "run":
                if len(fields) < 5:
                    raise RuntimeError(f"{path}:{line_number}: malformed run directive")
                parameters: dict[str, int] = {}
                for field in fields[3:]:
                    key, separator, value = field.partition("=")
                    if not separator or not key or not value:
                        raise RuntimeError(
                            f"{path}:{line_number}: malformed run parameter"
                        )
                    parameters[key] = int(value)
                runs.append(
                    {
                        "name": fields[1],
                        "method": fields[2],
                        "settings": parameters,
                    }
                )

    if not header_seen:
        raise RuntimeError(f"benchmark setup is empty: {path}")
    if output_path is None:
        raise RuntimeError(f"benchmark setup does not define output: {path}")
    if not runs:
        raise RuntimeError(f"benchmark setup does not define runs: {path}")
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


def require_optional_number(
    mapping: dict[str, object],
    key: str,
) -> float | None:
    if mapping.get(key) is None:
        return None
    return require_number(mapping, key)


def approximation_summary_fields(
    result: dict[str, object],
) -> dict[str, object]:
    approximation = require_mapping(
        result.get("approximation"),
        "approximation diagnostics",
    )
    distance_ratio = require_mapping(
        approximation.get("distance_ratio"),
        "distance ratio diagnostics",
    )
    ratio_distribution = require_mapping(
        distance_ratio.get("distribution"),
        "distance ratio distribution",
    )
    conditional = require_mapping(
        approximation.get("non_optimal_distance_ratio"),
        "conditional distance ratio diagnostics",
    )
    returned_rank = require_mapping(
        approximation.get("returned_representative_rank"),
        "returned rank diagnostics",
    )
    rank_distribution = require_mapping(
        returned_rank.get("distribution"),
        "returned rank distribution",
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
            "optimal_representative_count",
        ),
        "optimal_representative_rate": require_number(
            approximation,
            "optimal_representative_rate",
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
            ratio_distribution,
            "count",
        ),
        "zero_optimum_queries_excluded": require_integer(
            distance_ratio,
            "zero_optimum_queries_excluded",
        ),
        "mean_distance_ratio": require_optional_number(
            ratio_distribution,
            "mean",
        ),
        "median_distance_ratio": require_optional_number(
            ratio_distribution,
            "median",
        ),
        "distance_ratio_percentile_95": require_optional_number(
            ratio_distribution,
            "percentile_95",
        ),
        "distance_ratio_percentile_99": require_optional_number(
            ratio_distribution,
            "percentile_99",
        ),
        "maximum_distance_ratio": require_optional_number(
            ratio_distribution,
            "maximum",
        ),
        "conditional_non_optimal_ratio_count": require_integer(
            conditional,
            "eligible_non_optimal_count",
        ),
        "zero_optimum_non_optimal_count": require_integer(
            conditional,
            "zero_optimum_non_optimal_count",
        ),
        "conditional_mean_distance_ratio": require_optional_number(
            conditional,
            "mean",
        ),
        "conditional_mean_relative_excess": require_optional_number(
            conditional,
            "mean_relative_excess",
        ),
        "returned_rank_mean": require_optional_number(
            rank_distribution,
            "mean",
        ),
        "returned_rank_percentile_95": require_optional_number(
            rank_distribution,
            "percentile_95",
        ),
        "returned_rank_maximum": require_optional_number(
            rank_distribution,
            "maximum",
        ),
    }
    for epsilon, suffix in (
        (0.01, "0_01"),
        (0.05, "0_05"),
        (0.10, "0_10"),
        (0.20, "0_20"),
    ):
        matching = next(
            (
                rate
                for reported_epsilon, rate in failure_rates.items()
                if math.isclose(reported_epsilon, epsilon)
            ),
            None,
        )
        if matching is None:
            raise RuntimeError(f"approximation failures omit epsilon={epsilon}")
        fields[f"approximation_failure_rate_epsilon_{suffix}"] = matching
    return fields


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
    if report.get("schema_version") != 4:
        raise RuntimeError("benchmark sweep requires a schema-version 4 batch report")
    reported_setup = Path(str(report.get("setup_file", "")))
    if not reported_setup.is_absolute():
        reported_setup = PROJECT_ROOT / reported_setup
    reported_setup = reported_setup.resolve()
    if reported_setup != setup_path:
        raise RuntimeError(
            f"report was generated from {reported_setup}, not {setup_path}"
        )
    reported_runs = report.get("runs")
    if not isinstance(reported_runs, list):
        raise RuntimeError("batch report does not contain a runs array")
    observed = [
        {
            "name": run.get("name"),
            "method": run.get("method"),
            "settings": run.get("settings"),
        }
        for run in reported_runs
        if isinstance(run, dict)
    ]
    if observed != configured_runs:
        raise RuntimeError(
            "existing batch report does not match the setup; use --force "
            "to regenerate it"
        )
    require_mapping(report.get("exact"), "exact result")
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
    exact = require_mapping(report["exact"], "exact result")
    dataset = require_mapping(report.get("dataset"), "dataset")
    exact_space = require_mapping(exact.get("space"), "exact space")
    exact_accuracy = require_number(exact, "accuracy")
    exact_query_us = require_number(exact, "microseconds_per_query")
    exact_build_ms = require_number(exact, "build_ms")
    exact_query_ms = require_number(exact, "query_total_ms")
    exact_index_bytes = require_integer(exact_space, "index_payload_bytes")
    report_runs = report["runs"]
    assert isinstance(report_runs, list)
    rows: list[dict[str, object]] = []
    for value in report_runs:
        run = require_mapping(value, "run")
        if run.get("method") != method:
            continue
        settings = require_mapping(run.get("settings"), "run settings")
        result = require_mapping(run.get("result"), "run result")
        space = require_mapping(result.get("space"), "run space")
        coordinates = require_mapping(
            result.get("coordinate_access"),
            "run coordinate access",
        )
        repetitions = require_integer(settings, "repetitions")
        query_count = require_integer(result, "query_count")
        accuracy = require_number(result, "accuracy")
        agreement = require_number(result, "agreement_with_exact")
        label_agreement = (
            require_number(result, "label_agreement_with_exact")
            if result.get("label_agreement_with_exact") is not None
            else agreement
        )
        query_us = require_number(result, "microseconds_per_query")
        build_ms = require_number(result, "build_ms")
        query_ms = require_number(result, "query_total_ms")
        index_bytes = require_integer(space, "index_payload_bytes")
        multiplicity = require_integer(coordinates, "sampled_multiplicity")
        agreement_count = round(agreement * query_count)
        rows.append(
            {
                "name": run.get("name"),
                "method": method,
                "repetitions": repetitions,
                "seed": require_integer(settings, "seed"),
                "query_count": query_count,
                "dimension": require_integer(dataset, "dimension"),
                "representative_count": require_integer(
                    dataset,
                    "representative_count",
                ),
                "exact_correct": require_integer(exact, "correct"),
                f"{method}_correct": require_integer(result, "correct"),
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
                "unique_coordinates": require_integer(
                    coordinates,
                    "unique_coordinates",
                ),
                "dimension_fraction": require_number(
                    coordinates,
                    "dimension_fraction",
                ),
                "sampled_multiplicity": multiplicity,
                "sampling_mass_estimate": multiplicity / repetitions,
                "exact_index_bytes": exact_index_bytes,
                f"{method}_index_bytes": index_bytes,
                "index_compression": exact_index_bytes / index_bytes,
                **approximation_summary_fields(result),
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
    metric_names = (
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
        "approximation_failure_rate_epsilon_0_01",
        "approximation_failure_rate_epsilon_0_05",
        "approximation_failure_rate_epsilon_0_10",
        "approximation_failure_rate_epsilon_0_20",
        "returned_rank_mean",
        "returned_rank_percentile_95",
        "returned_rank_maximum",
    )
    aggregates: list[dict[str, object]] = []
    for repetitions in sorted({int(row["repetitions"]) for row in rows}):
        group = [row for row in rows if row["repetitions"] == repetitions]
        aggregates.append(
            {
                "repetitions": repetitions,
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
        "schema_version": 1,
        "source_report": str(report_path),
        "method": method,
        "runs": rows,
        "aggregates": aggregates,
    }
    if report is not None and isinstance(report.get("shared_preprocessing"), dict):
        summary["shared_preprocessing"] = report["shared_preprocessing"]
    if report is not None and isinstance(report.get("diagnostics"), dict):
        summary["diagnostics"] = report["diagnostics"]
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
        f"{'T':>7} {'prototype':>11} {'class':>9} {'accuracy':>10} "
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
    settings = require_mapping(run.get("settings"), "run settings")
    return (
        require_integer(settings, "repetitions"),
        require_integer(settings, "seed"),
    )


def require_result(run: dict[str, object]) -> dict[str, object]:
    return require_mapping(run.get("result"), "run result")


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

    exact = require_mapping(report.get("exact"), "exact result")
    dataset = require_mapping(report.get("dataset"), "dataset")
    exact_query_us = require_number(exact, "microseconds_per_query")
    exact_accuracy = require_number(exact, "accuracy")
    representative_count = require_integer(dataset, "representative_count")
    dimension = require_integer(dataset, "dimension")

    report_runs = report.get("runs")
    if not isinstance(report_runs, list):
        raise RuntimeError("batch report does not contain a runs array")
    runs = [require_mapping(value, "run") for value in report_runs]
    flat_runs: dict[tuple[int, int], dict[str, object]] = {}
    for run in runs:
        if run.get("method") != "flat":
            continue
        identity = run_identity(run)
        if identity in flat_runs:
            raise RuntimeError(
                "batch report contains duplicate matched flat runs for "
                f"T={identity[0]}, seed={identity[1]}"
            )
        flat_runs[identity] = run

    rows: list[dict[str, object]] = []
    for hierarchical_run in runs:
        if hierarchical_run.get("method") != "hierarchical":
            continue
        repetitions, seed = run_identity(hierarchical_run)
        flat_run = flat_runs.get((repetitions, seed))
        if flat_run is None:
            raise RuntimeError(
                "hierarchical run has no matched flat run for "
                f"T={repetitions}, seed={seed}"
            )
        hierarchical_settings = require_mapping(
            hierarchical_run.get("settings"),
            "hierarchical settings",
        )
        projection_dimension = require_integer(
            hierarchical_settings,
            "projection_dimension",
        )
        flat = require_result(flat_run)
        hierarchical = require_result(hierarchical_run)
        flat_space = require_mapping(flat.get("space"), "flat space")
        hierarchical_space = require_mapping(
            hierarchical.get("space"),
            "hierarchical space",
        )
        flat_coordinates = require_mapping(
            flat.get("coordinate_access"),
            "flat coordinate access",
        )
        hierarchical_coordinates = require_mapping(
            hierarchical.get("coordinate_access"),
            "hierarchical coordinate access",
        )

        query_count = require_integer(flat, "query_count")
        if require_integer(hierarchical, "query_count") != query_count:
            raise RuntimeError("matched flat and hierarchical query counts differ")
        coordinate_keys = ("unique_coordinates", "sampled_multiplicity")
        for key in coordinate_keys:
            if require_integer(flat_coordinates, key) != require_integer(
                hierarchical_coordinates,
                key,
            ):
                raise RuntimeError(
                    "matched flat and hierarchical sampling differs for "
                    f"T={repetitions}, seed={seed}"
                )

        flat_accuracy = require_number(flat, "accuracy")
        hierarchical_accuracy = require_number(hierarchical, "accuracy")
        flat_agreement = require_number(flat, "agreement_with_exact")
        hierarchical_agreement = require_number(
            hierarchical,
            "agreement_with_exact",
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
        flat_query_us = require_number(flat, "microseconds_per_query")
        hierarchical_query_us = require_number(
            hierarchical,
            "microseconds_per_query",
        )
        flat_index_bytes = require_integer(flat_space, "index_payload_bytes")
        hierarchical_index_bytes = require_integer(
            hierarchical_space,
            "index_payload_bytes",
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
                "repetitions": repetitions,
                "projection_dimension": projection_dimension,
                "seed": seed,
                "query_count": query_count,
                "representative_count": representative_count,
                "dimension": dimension,
                "exact_accuracy": exact_accuracy,
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
                "hierarchical_speedup_vs_flat": (flat_query_us / hierarchical_query_us),
                "hierarchical_speedup_vs_exact": (
                    exact_query_us / hierarchical_query_us
                ),
                "flat_build_ms": require_number(flat, "build_ms"),
                "hierarchical_build_ms": require_number(
                    hierarchical,
                    "build_ms",
                ),
                "unique_coordinates": require_integer(
                    hierarchical_coordinates,
                    "unique_coordinates",
                ),
                "dimension_fraction": require_number(
                    hierarchical_coordinates,
                    "dimension_fraction",
                ),
                "sampled_multiplicity": require_integer(
                    hierarchical_coordinates,
                    "sampled_multiplicity",
                ),
                "flat_index_bytes": flat_index_bytes,
                "hierarchical_index_bytes": hierarchical_index_bytes,
                "hierarchical_index_size_ratio_vs_flat": (
                    hierarchical_index_bytes / flat_index_bytes
                ),
                "flat_workspace_bytes": require_integer(
                    flat_space,
                    "query_workspace_payload_bytes",
                ),
                "hierarchical_workspace_bytes": require_integer(
                    hierarchical_space,
                    "query_workspace_payload_bytes",
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
    metric_names = (
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
        "hierarchical_approximation_failure_rate_epsilon_0_01",
        "hierarchical_approximation_failure_rate_epsilon_0_05",
        "hierarchical_approximation_failure_rate_epsilon_0_10",
        "hierarchical_approximation_failure_rate_epsilon_0_20",
        "hierarchical_returned_rank_mean",
        "hierarchical_returned_rank_percentile_95",
        "hierarchical_returned_rank_maximum",
    )
    groups = sorted(
        {(int(row["repetitions"]), int(row["projection_dimension"])) for row in rows}
    )
    aggregates: list[dict[str, object]] = []
    for repetitions, projection_dimension in groups:
        group = [
            row
            for row in rows
            if row["repetitions"] == repetitions
            and row["projection_dimension"] == projection_dimension
        ]
        aggregates.append(
            {
                "repetitions": repetitions,
                "projection_dimension": projection_dimension,
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
        "schema_version": 1,
        "source_report": str(report_path),
        "comparison": "hierarchical versus matched flat with identical T and seed",
        "runs": rows,
        "aggregates": aggregates,
    }
    if report is not None and isinstance(report.get("shared_preprocessing"), dict):
        summary["shared_preprocessing"] = report["shared_preprocessing"]
    if report is not None and isinstance(report.get("diagnostics"), dict):
        summary["diagnostics"] = report["diagnostics"]
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
        f"{'T':>6} {'p':>4} {'class agr':>11} {'delta flat':>11} "
        f"{'prototype':>10} "
        f"{'accuracy':>10} {'query us':>10} {'vs flat':>9} {'vs exact':>9}"
    )
    for aggregate in aggregate_hierarchical_rows(rows):
        print(
            f"{aggregate['repetitions']:6d} "
            f"{aggregate['projection_dimension']:4d} "
            f"{100.0 * hierarchical_mean_metric(aggregate, 'hierarchical_label_agreement_with_exact'):10.2f}% "
            f"{100.0 * hierarchical_mean_metric(aggregate, 'hierarchical_label_agreement_delta_vs_flat'):+10.2f} "
            f"{100.0 * hierarchical_mean_metric(aggregate, 'hierarchical_agreement_with_exact'):9.2f}% "
            f"{100.0 * hierarchical_mean_metric(aggregate, 'hierarchical_accuracy'):9.2f}% "
            f"{hierarchical_mean_metric(aggregate, 'hierarchical_query_us'):10.2f} "
            f"{hierarchical_mean_metric(aggregate, 'hierarchical_speedup_vs_flat'):8.2f}x "
            f"{hierarchical_mean_metric(aggregate, 'hierarchical_speedup_vs_exact'):8.2f}x"
        )
