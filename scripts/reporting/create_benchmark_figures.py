#!/usr/bin/env python3
"""Create publication-oriented figures from a schema-version 6 report."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "benchmark"))

from benchmark_reporting import validate_report_provenance  # noqa: E402


DEFAULT_REPORT = (
    PROJECT_ROOT
    / "results"
    / "raw"
    / "tcga_pancancer_r33_l2_cpu_sparse_sweep.json"
)
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "results" / "plots"
DISTANCE_RATIO_FIELDS = {
    "mean": "mean",
    "median": "median",
    "p95": "percentile_95",
    "p99": "percentile_99",
    "maximum": "maximum",
}
DISTANCE_STATISTIC_LABELS = {
    "mean": "Mean",
    "median": "Median",
    "p95": "95th-percentile",
    "p99": "99th-percentile",
    "maximum": "Maximum",
}


def configure_matplotlib_cache() -> None:
    """Avoid Matplotlib warnings in read-only home-directory environments."""
    if "MPLCONFIGDIR" in os.environ:
        return
    default_config = Path.home() / ".config" / "matplotlib"
    if os.access(default_config.parent, os.W_OK):
        return
    cache = Path(tempfile.gettempdir()) / "ultrahigh-ann-matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache)


configure_matplotlib_cache()

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
except ModuleNotFoundError as import_error:  # pragma: no cover - environment-specific
    plt = None
    ScalarMappable = None
    Normalize = None
    MATPLOTLIB_IMPORT_ERROR: ModuleNotFoundError | None = import_error
else:
    MATPLOTLIB_IMPORT_ERROR = None


@dataclass(frozen=True, order=True)
class Configuration:
    method: str
    repetitions: int
    projection_dimension: int = 0
    batch_size: int = 1
    backend: str = "cpu"
    strategy: str = "sequential"

    def label(self) -> str:
        if self.method == "exact":
            label = "Exact"
        elif self.method == "flat":
            label = f"Flat, T={self.repetitions}"
        elif self.method == "uniform":
            label = f"Uniform, T={self.repetitions}"
        else:
            label = (
                f"Hierarchy, T={self.repetitions}, " f"p={self.projection_dimension}"
            )
        if self.batch_size != 1:
            label += f", B={self.batch_size}"
        if self.backend != "cpu" or self.strategy != "sequential":
            label += f", {self.backend}/{self.strategy}"
        return label


@dataclass
class Observation:
    configuration: Configuration
    query_microseconds: float
    exact_agreement: float
    accuracy: float
    margin_error_rates: list[float]
    approximation_epsilons: list[float]
    approximation_failure_rates: list[float]
    distance_ratios: dict[str, float] | None


@dataclass
class Aggregate:
    configuration: Configuration
    query_microseconds: tuple[float, float, float]
    exact_agreement: tuple[float, float, float]
    accuracy: tuple[float, float, float]
    margin_error_rates: list[tuple[float, float, float]]
    approximation_epsilons: list[float]
    approximation_failure_rates: list[tuple[float, float, float]]
    distance_ratios: dict[str, tuple[float, float, float]] | None
    run_count: int


@dataclass
class ReportData:
    report_path: Path
    dataset_name: str
    distance: str
    reference_vector_count: int
    dimension: int
    exact_query_microseconds: float
    exact_accuracy: float
    batch_size: int
    exact_batch_size: int
    margin_labels: list[str]
    margin_query_shares: list[float]
    aggregates: list[Aggregate]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create figures for query margins, approximation failures, "
            "agreement with exact search, and relative distance excess "
            "from a schema-version 6 benchmark report."
        )
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help=f"raw batch JSON report (default: {DEFAULT_REPORT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help=f"figure directory (default: {DEFAULT_OUTPUT_DIRECTORY})",
    )
    parser.add_argument(
        "--prefix",
        help="output filename prefix (default: report filename stem)",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("pdf", "png", "svg"),
        default=("pdf", "png"),
        help="output formats (default: pdf png)",
    )
    parser.add_argument(
        "--figures",
        nargs="+",
        choices=("margin", "failure", "pareto", "distance"),
        default=("margin", "failure", "pareto"),
        help="figures to produce (default: margin failure pareto)",
    )
    parser.add_argument(
        "--line-method",
        choices=(
            "flat",
            "uniform",
            "hierarchical",
            "flat-uniform",
            "flat-hierarchical",
            "both",
            "all",
        ),
        default="flat",
        help=(
            "methods shown in the margin and failure line plots "
            "('both' aliases flat-hierarchical; default: flat)"
        ),
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        nargs="*",
        help="optional T filter for margin and failure plots",
    )
    parser.add_argument(
        "--projection-dimensions",
        type=int,
        nargs="*",
        help="optional p filter for hierarchical line plots",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help=(
            "measurement batch size; required when a report contains "
            "multiple batch sizes"
        ),
    )
    parser.add_argument(
        "--distance-statistic",
        choices=tuple(DISTANCE_RATIO_FIELDS),
        default="mean",
        help=(
            "distance-ratio statistic for the distance figure "
            "(default: mean)"
        ),
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help=(
            "use compact Pareto/distance styling for dense sweeps: retain "
            "every point, omit per-point labels, encode hierarchical T by "
            "marker size, and collapse its legend"
        ),
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=15,
        help="reject line plots with more than this many curves (default: 15)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="raster output resolution (default: 300)",
    )
    return parser.parse_args()


def require_mapping(value: object, description: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{description} is not a JSON object")
    return value


def require_list(value: object, description: str) -> list[object]:
    if not isinstance(value, list):
        raise RuntimeError(f"{description} is not a JSON array")
    return value


def require_number(mapping: dict[str, object], key: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"field {key!r} is not numeric")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"field {key!r} is not finite")
    return number


def require_integer(mapping: dict[str, object], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"field {key!r} is not an integer")
    return value


def format_bound(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def margin_label(bucket: dict[str, object]) -> str:
    lower = require_number(bucket, "lower_inclusive")
    upper = bucket.get("upper_exclusive")
    if upper is None:
        return f"≥{format_bound(lower)}"
    if isinstance(upper, bool) or not isinstance(upper, (int, float)):
        raise RuntimeError("margin upper bound is not numeric or null")
    return f"{format_bound(lower)}–{format_bound(float(upper))}"


def parse_observation(raw_run: object, raw_measurement: object) -> Observation:
    run = require_mapping(raw_run, "run")
    method = run.get("index")
    if method not in ("exact", "flat", "uniform", "hierarchical"):
        raise RuntimeError(f"unsupported comparison method: {method!r}")
    repetitions = require_integer(run, "repetitions")
    projection_dimension = (
        require_integer(run, "projection_dimension") if method == "hierarchical" else 0
    )
    measurement = require_mapping(raw_measurement, "measurement")
    raw_approximation = measurement.get("approximation")
    if raw_approximation is None:
        margin_buckets: list[object] = []
        failures: list[object] = []
        distance_ratios: dict[str, float] | None = None
    else:
        approximation = require_mapping(
            raw_approximation,
            "approximation diagnostics",
        )
        margin_buckets = require_list(
            approximation.get("by_multiplicative_margin"),
            "margin diagnostics",
        )
        failures = require_list(
            approximation.get("approximation_guarantee_failures"),
            "approximation failures",
        )
        raw_distance_ratio = require_mapping(
            approximation.get("distance_ratio"),
            "distance-ratio diagnostics",
        )
        distance_ratios = {
            statistic: require_number(raw_distance_ratio, field)
            for statistic, field in DISTANCE_RATIO_FIELDS.items()
        }
    margin_rates: list[float] = []
    for raw_bucket in margin_buckets:
        bucket = require_mapping(raw_bucket, "margin bucket")
        count = require_integer(bucket, "query_count")
        non_optimal = require_integer(bucket, "non_optimal_count")
        # Empty margin buckets contain no observed errors.
        margin_rates.append(non_optimal / count if count else 0.0)
    epsilons: list[float] = []
    failure_rates: list[float] = []
    for raw_failure in failures:
        failure = require_mapping(raw_failure, "approximation failure")
        epsilons.append(require_number(failure, "epsilon"))
        failure_rates.append(require_number(failure, "violation_rate"))
    return Observation(
        configuration=Configuration(
            method=str(method),
            repetitions=repetitions,
            projection_dimension=projection_dimension,
            batch_size=require_integer(measurement, "batch_size"),
            backend=str(run.get("backend")),
            strategy=str(run.get("strategy")),
        ),
        query_microseconds=require_number(measurement, "median_microseconds_per_query"),
        exact_agreement=require_number(measurement, "exact_neighbor_agreement"),
        accuracy=require_number(measurement, "accuracy"),
        margin_error_rates=margin_rates,
        approximation_epsilons=epsilons,
        approximation_failure_rates=failure_rates,
        distance_ratios=distance_ratios,
    )


def mean_range(values: list[float]) -> tuple[float, float, float]:
    if not values:
        raise RuntimeError("cannot aggregate an empty metric")
    return statistics.mean(values), min(values), max(values)


def aggregate_observations(observations: list[Observation]) -> list[Aggregate]:
    grouped: dict[Configuration, list[Observation]] = {}
    for observation in observations:
        grouped.setdefault(observation.configuration, []).append(observation)

    aggregates: list[Aggregate] = []
    for configuration, runs in sorted(grouped.items()):
        margin_count = len(runs[0].margin_error_rates)
        epsilons = runs[0].approximation_epsilons
        for run in runs[1:]:
            if len(run.margin_error_rates) != margin_count:
                raise RuntimeError("runs use different margin buckets")
            if run.approximation_epsilons != epsilons:
                raise RuntimeError("runs use different epsilon thresholds")
            if (run.distance_ratios is None) != (runs[0].distance_ratios is None):
                raise RuntimeError("runs use inconsistent distance-ratio diagnostics")
        distance_ratios = (
            None
            if runs[0].distance_ratios is None
            else {
                statistic: mean_range(
                    [
                        run.distance_ratios[statistic]
                        for run in runs
                        if run.distance_ratios is not None
                    ]
                )
                for statistic in DISTANCE_RATIO_FIELDS
            }
        )
        aggregates.append(
            Aggregate(
                configuration=configuration,
                query_microseconds=mean_range([run.query_microseconds for run in runs]),
                exact_agreement=mean_range([run.exact_agreement for run in runs]),
                accuracy=mean_range([run.accuracy for run in runs]),
                margin_error_rates=[
                    mean_range([run.margin_error_rates[index] for run in runs])
                    for index in range(margin_count)
                ],
                approximation_epsilons=list(epsilons),
                approximation_failure_rates=[
                    mean_range([run.approximation_failure_rates[index] for run in runs])
                    for index in range(len(epsilons))
                ],
                distance_ratios=distance_ratios,
                run_count=len(runs),
            )
        )
    return aggregates


def load_report(path: Path, requested_batch_size: int | None = None) -> ReportData:
    try:
        with path.open(encoding="utf-8") as source:
            root = require_mapping(json.load(source), "report root")
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read report {path}: {error}") from error
    if root.get("schema_version") != 6:
        raise RuntimeError("figures require a schema-version 6 report")
    validate_report_provenance(root)

    dataset = require_mapping(root.get("dataset"), "dataset")
    settings = require_mapping(root.get("settings"), "settings")
    runs = [
        require_mapping(run, "run") for run in require_list(root.get("runs"), "runs")
    ]
    references = [run for run in runs if run.get("reference") is True]
    if len(references) != 1 or references[0].get("index") != "exact":
        raise RuntimeError("report must contain exactly one exact reference run")
    exact = references[0]

    measurements: dict[int, list[tuple[dict[str, object], dict[str, object]]]] = {}
    for run in runs:
        seen_batches: set[int] = set()
        for raw_measurement in require_list(
            run.get("measurements"), "run measurements"
        ):
            measurement = require_mapping(raw_measurement, "measurement")
            batch_size = require_integer(measurement, "batch_size")
            if batch_size in seen_batches:
                raise RuntimeError(
                    f"run {run.get('name')!r} repeats batch size {batch_size}"
                )
            seen_batches.add(batch_size)
            measurements.setdefault(batch_size, []).append((run, measurement))
        if not seen_batches:
            raise RuntimeError(f"run {run.get('name')!r} has no measurements")

    available_batches = sorted(measurements)
    if requested_batch_size is None:
        if len(available_batches) != 1:
            choices = ", ".join(str(value) for value in available_batches)
            raise RuntimeError(
                "report contains multiple batch sizes; select one with "
                f"--batch-size ({choices})"
            )
        batch_size = available_batches[0]
    else:
        if requested_batch_size <= 0:
            raise RuntimeError("--batch-size must be positive")
        if requested_batch_size not in measurements:
            raise RuntimeError(
                f"report has no batch-size {requested_batch_size} measurement"
            )
        batch_size = requested_batch_size

    exact_measurements = [
        measurement for run, measurement in measurements[batch_size] if run is exact
    ]
    if not exact_measurements:
        reference_measurements = [
            require_mapping(measurement, "reference measurement")
            for measurement in require_list(
                exact.get("measurements"),
                "reference measurements",
            )
        ]
        if len(reference_measurements) == 1:
            exact_measurements = reference_measurements
    if len(exact_measurements) != 1:
        raise RuntimeError(
            f"reference run must contain batch-size {batch_size} exactly once, "
            "or contain exactly one fixed-baseline measurement"
        )
    exact_measurement = exact_measurements[0]

    diagnostic_execution = require_mapping(
        root.get("diagnostic_execution"), "diagnostic execution"
    )
    raw_geometry = diagnostic_execution.get("query_geometry")
    if raw_geometry is None:
        margin_labels: list[str] = []
        margin_query_shares: list[float] = []
    else:
        geometry = require_mapping(raw_geometry, "query geometry")
        raw_geometry_buckets = require_list(
            geometry.get("margin_buckets"),
            "query-geometry margin buckets",
        )
        query_count = require_integer(geometry, "query_count")
        if query_count <= 0:
            raise RuntimeError("query geometry has no queries")
        geometry_buckets = [
            require_mapping(bucket, "query-geometry margin bucket")
            for bucket in raw_geometry_buckets
        ]
        margin_labels = [margin_label(bucket) for bucket in geometry_buckets]
        margin_query_shares = [
            require_integer(bucket, "query_count") / query_count
            for bucket in geometry_buckets
        ]

    observations = [
        parse_observation(run, measurement)
        for run, measurement in measurements[batch_size]
        if run.get("reference") is not True
    ]
    if not observations:
        raise RuntimeError("report contains no approximate runs")
    if margin_labels and any(
        len(observation.margin_error_rates) != len(margin_labels)
        for observation in observations
    ):
        raise RuntimeError(
            "query geometry and run diagnostics use different margin buckets"
        )

    directory = Path(str(dataset.get("directory", "dataset")))
    distance = settings.get("distance")
    if distance not in ("l1", "l2"):
        raise RuntimeError("report distance must be l1 or l2")
    return ReportData(
        report_path=path,
        dataset_name=directory.name or "dataset",
        distance=str(distance),
        reference_vector_count=require_integer(dataset, "reference_vector_count"),
        dimension=require_integer(dataset, "dimension"),
        exact_query_microseconds=require_number(
            exact_measurement,
            "median_microseconds_per_query",
        ),
        exact_accuracy=require_number(exact_measurement, "accuracy"),
        batch_size=batch_size,
        exact_batch_size=require_integer(exact_measurement, "batch_size"),
        margin_labels=margin_labels,
        margin_query_shares=margin_query_shares,
        aggregates=aggregate_observations(observations),
    )


def select_line_aggregates(
    report: ReportData,
    method: str,
    repetitions: list[int] | None,
    projection_dimensions: list[int] | None,
    maximum_lines: int,
) -> list[Aggregate]:
    method_groups = {
        "flat-uniform": {"flat", "uniform"},
        "flat-hierarchical": {"flat", "hierarchical"},
        "both": {"flat", "hierarchical"},
        "all": {"flat", "uniform", "hierarchical"},
    }
    methods = method_groups.get(method, {method})
    selected = [
        aggregate
        for aggregate in report.aggregates
        if aggregate.configuration.method in methods
        and (repetitions is None or aggregate.configuration.repetitions in repetitions)
        and (
            aggregate.configuration.method != "hierarchical"
            or projection_dimensions is None
            or aggregate.configuration.projection_dimension in projection_dimensions
        )
    ]
    if not selected:
        raise RuntimeError("line-plot filters select no configurations")
    if len(selected) > maximum_lines:
        raise RuntimeError(
            f"line-plot filters select {len(selected)} configurations, above "
            f"--max-lines={maximum_lines}; filter T or p explicitly"
        )
    return selected


def apply_style() -> None:
    assert plt is not None
    plt.rcParams.update(
        {
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "grid.alpha": 0.25,
            "font.size": 9,
            "legend.fontsize": 8,
            "figure.dpi": 120,
            "savefig.dpi": 300,
        }
    )


def execution_groups(
    aggregates: list[Aggregate],
) -> list[tuple[tuple[str, str], list[Aggregate]]]:
    keys = sorted(
        {
            (aggregate.configuration.backend, aggregate.configuration.strategy)
            for aggregate in aggregates
        }
    )
    return [
        (
            key,
            [
                aggregate
                for aggregate in aggregates
                if (
                    aggregate.configuration.backend,
                    aggregate.configuration.strategy,
                )
                == key
            ],
        )
        for key in keys
    ]


def execution_series_label(
    method: str,
    execution: tuple[str, str],
    execution_count: int,
) -> str:
    if execution_count == 1:
        return method
    backend, strategy = execution
    return f"{method}, {backend}/{strategy}"


def strategy_line_style(strategy: str) -> str:
    return {"direct": "-", "gemm": "--"}.get(strategy, "-")


def strategy_marker_facecolor(strategy: str, color: str) -> str:
    return "none" if strategy == "gemm" else color


def hierarchy_marker_sizes(
    aggregates: list[Aggregate],
) -> dict[int, float]:
    repetitions = sorted(
        {aggregate.configuration.repetitions for aggregate in aggregates}
    )
    if not repetitions:
        return {}
    logarithms = [math.log2(value) for value in repetitions]
    lower = min(logarithms)
    span = max(logarithms) - lower
    return {
        repetition: (
            42.0
            if span == 0.0
            else 24.0 + 48.0 * (math.log2(repetition) - lower) / span
        )
        for repetition in repetitions
    }


def hierarchy_size_legend_values(repetitions: list[int]) -> list[int]:
    if len(repetitions) <= 3:
        return repetitions
    logarithmic_middle = (
        math.log2(repetitions[0]) + math.log2(repetitions[-1])
    ) / 2.0
    middle = min(
        repetitions,
        key=lambda value: abs(math.log2(value) - logarithmic_middle),
    )
    return list(dict.fromkeys((repetitions[0], middle, repetitions[-1])))


def add_hierarchy_size_legend(
    axis,
    marker_sizes: dict[int, float],
) -> None:
    for repetition in hierarchy_size_legend_values(sorted(marker_sizes)):
        axis.scatter(
            [],
            [],
            marker="D",
            s=marker_sizes[repetition],
            facecolor="#777777",
            edgecolor="#555555",
            linewidth=0.5,
            alpha=0.7,
            label=f"Hierarchy size: T={repetition}",
        )


def report_subtitle(report: ReportData) -> str:
    dataset_label = report.dataset_name.replace("_", " ")
    subtitle = (
        f"{dataset_label}, {report.distance.upper()}, "
        f"r={report.reference_vector_count}, d={report.dimension:,}"
    )
    if report.exact_batch_size != report.batch_size:
        subtitle += (
            f", candidate B={report.batch_size}, "
            f"reference B={report.exact_batch_size}"
        )
    elif report.batch_size != 1:
        subtitle += f", B={report.batch_size}"
    return subtitle


def plot_margin_error(
    report: ReportData,
    selected: list[Aggregate],
):
    assert plt is not None
    figure, axis = plt.subplots(figsize=(7.0, 4.3))
    positions = list(range(len(report.margin_labels)))
    distribution_axis = axis.twinx()
    distribution_axis.bar(
        positions,
        [100.0 * share for share in report.margin_query_shares],
        width=0.76,
        color="#999999",
        alpha=0.14,
        label="Query share",
        zorder=0,
    )
    distribution_axis.set_ylabel("Queries in bucket (%)", color="#666666")
    distribution_axis.tick_params(axis="y", colors="#666666")
    distribution_axis.grid(False)
    distribution_axis.spines["right"].set_visible(True)

    colors = plt.get_cmap("tab10")
    for index, aggregate in enumerate(selected):
        means = [100.0 * value[0] for value in aggregate.margin_error_rates]
        minima = [100.0 * value[1] for value in aggregate.margin_error_rates]
        maxima = [100.0 * value[2] for value in aggregate.margin_error_rates]
        color = colors(index % 10)
        axis.plot(
            positions,
            means,
            marker="o",
            linewidth=1.7,
            markersize=4,
            color=color,
            label=aggregate.configuration.label(),
            zorder=3,
        )
        axis.fill_between(
            positions,
            minima,
            maxima,
            color=color,
            alpha=0.10,
            linewidth=0,
            zorder=2,
        )
    axis.set_xticks(positions, report.margin_labels)
    axis.set_xlabel("Multiplicative margin $d_2/d_1$")
    axis.set_ylabel("Non-optimal return rate (%)")
    axis.set_ylim(bottom=0)
    axis.set_title(f"Error concentration by query margin\n{report_subtitle(report)}")
    axis.legend(loc="upper right", frameon=True)
    figure.tight_layout()
    return figure


def plot_approximation_failures(
    report: ReportData,
    selected: list[Aggregate],
):
    assert plt is not None
    figure, axis = plt.subplots(figsize=(6.6, 4.2))
    colors = plt.get_cmap("tab10")
    for index, aggregate in enumerate(selected):
        epsilons = [100.0 * value for value in aggregate.approximation_epsilons]
        means = [100.0 * value[0] for value in aggregate.approximation_failure_rates]
        minima = [100.0 * value[1] for value in aggregate.approximation_failure_rates]
        maxima = [100.0 * value[2] for value in aggregate.approximation_failure_rates]
        color = colors(index % 10)
        axis.plot(
            epsilons,
            means,
            marker="o",
            linewidth=1.7,
            markersize=4,
            color=color,
            label=aggregate.configuration.label(),
        )
        axis.fill_between(
            epsilons,
            minima,
            maxima,
            color=color,
            alpha=0.10,
            linewidth=0,
        )
    axis.set_xlabel("Approximation tolerance ε (%)")
    axis.set_ylabel("Queries with ratio $>1+ε$ (%)")
    axis.set_ylim(bottom=0)
    axis.set_title(f"Empirical approximation-failure curves\n{report_subtitle(report)}")
    axis.legend(loc="upper right", frameon=True)
    figure.tight_layout()
    return figure


def pareto_frontier(
    points: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    frontier: list[tuple[float, float]] = []
    best_agreement = -math.inf
    for latency, agreement in sorted(points):
        if agreement > best_agreement:
            frontier.append((latency, agreement))
            best_agreement = agreement
    return frontier


def plot_latency_agreement_pareto(
    report: ReportData,
    compact: bool = False,
):
    assert plt is not None and ScalarMappable is not None and Normalize is not None
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    flat = [
        aggregate
        for aggregate in report.aggregates
        if aggregate.configuration.method == "flat"
    ]
    uniform = [
        aggregate
        for aggregate in report.aggregates
        if aggregate.configuration.method == "uniform"
    ]
    hierarchical = [
        aggregate
        for aggregate in report.aggregates
        if aggregate.configuration.method == "hierarchical"
    ]
    exact_candidates = [
        aggregate
        for aggregate in report.aggregates
        if aggregate.configuration.method == "exact"
    ]

    if flat:
        groups = execution_groups(flat)
        for execution, group in groups:
            group.sort(key=lambda value: value.configuration.repetitions)
            axis.plot(
                [aggregate.query_microseconds[0] for aggregate in group],
                [100.0 * aggregate.exact_agreement[0] for aggregate in group],
                color="#0072B2",
                linestyle=strategy_line_style(execution[1]),
                marker="s",
                markerfacecolor=strategy_marker_facecolor(
                    execution[1], "#0072B2"
                ),
                markeredgecolor="#0072B2",
                markeredgewidth=1.0,
                linewidth=2.0,
                markersize=5,
                label=execution_series_label("Flat", execution, len(groups)),
                zorder=5,
            )
            if not compact:
                for offset, aggregate in enumerate(group):
                    axis.annotate(
                        f"T={aggregate.configuration.repetitions}",
                        (
                            aggregate.query_microseconds[0],
                            100.0 * aggregate.exact_agreement[0],
                        ),
                        xytext=(4, 5 if offset % 2 == 0 else -11),
                        textcoords="offset points",
                        fontsize=7,
                        color="#005A8D",
                    )

    if uniform:
        groups = execution_groups(uniform)
        for execution, group in groups:
            group.sort(key=lambda value: value.configuration.repetitions)
            axis.plot(
                [aggregate.query_microseconds[0] for aggregate in group],
                [100.0 * aggregate.exact_agreement[0] for aggregate in group],
                color="#D55E00",
                linestyle=strategy_line_style(execution[1]),
                marker="o",
                markerfacecolor=strategy_marker_facecolor(
                    execution[1], "#D55E00"
                ),
                markeredgecolor="#D55E00",
                markeredgewidth=1.0,
                linewidth=2.0,
                markersize=5,
                label=execution_series_label(
                    "Uniform", execution, len(groups)
                ),
                zorder=4,
            )
            if not compact:
                for offset, aggregate in enumerate(group):
                    axis.annotate(
                        f"T={aggregate.configuration.repetitions}",
                        (
                            aggregate.query_microseconds[0],
                            100.0 * aggregate.exact_agreement[0],
                        ),
                        xytext=(4, -11 if offset % 2 == 0 else 5),
                        textcoords="offset points",
                        fontsize=7,
                        color="#A64500",
                    )

    if hierarchical:
        projection_dimensions = [
            aggregate.configuration.projection_dimension for aggregate in hierarchical
        ]
        norm = Normalize(
            vmin=min(projection_dimensions),
            vmax=max(projection_dimensions),
        )
        colormap = plt.get_cmap("viridis")
        groups = execution_groups(hierarchical)
        if compact:
            marker_sizes = hierarchy_marker_sizes(hierarchical)
            for execution, group in groups:
                latencies = [
                    aggregate.query_microseconds[0] for aggregate in group
                ]
                agreements = [
                    100.0 * aggregate.exact_agreement[0]
                    for aggregate in group
                ]
                projection_values = [
                    aggregate.configuration.projection_dimension
                    for aggregate in group
                ]
                sizes = [
                    marker_sizes[aggregate.configuration.repetitions]
                    for aggregate in group
                ]
                label = execution_series_label(
                    "Hierarchy", execution, len(groups)
                )
                if execution[1] == "gemm":
                    axis.scatter(
                        latencies,
                        agreements,
                        facecolors="none",
                        edgecolors=[
                            colormap(norm(value)) for value in projection_values
                        ],
                        marker="D",
                        s=sizes,
                        alpha=0.9,
                        linewidths=1.0,
                        label=label,
                        zorder=3,
                    )
                else:
                    axis.scatter(
                        latencies,
                        agreements,
                        c=projection_values,
                        cmap=colormap,
                        norm=norm,
                        marker="D",
                        s=sizes,
                        alpha=0.78,
                        edgecolors="white",
                        linewidths=0.4,
                        label=label,
                        zorder=3,
                    )
            add_hierarchy_size_legend(axis, marker_sizes)
        else:
            marker_cycle = ("o", "^", "D", "P", "X", "v")
            repetitions = sorted(
                {
                    aggregate.configuration.repetitions
                    for aggregate in hierarchical
                }
            )
            for execution, execution_group in groups:
                for marker_index, repetition in enumerate(repetitions):
                    group = [
                        aggregate
                        for aggregate in execution_group
                        if aggregate.configuration.repetitions == repetition
                    ]
                    if not group:
                        continue
                    label = f"Hierarchy, T={repetition}"
                    if len(groups) > 1:
                        label += f", {execution[0]}/{execution[1]}"
                    latencies = [
                        aggregate.query_microseconds[0] for aggregate in group
                    ]
                    agreements = [
                        100.0 * aggregate.exact_agreement[0]
                        for aggregate in group
                    ]
                    projection_values = [
                        aggregate.configuration.projection_dimension
                        for aggregate in group
                    ]
                    marker = marker_cycle[marker_index % len(marker_cycle)]
                    if execution[1] == "gemm":
                        axis.scatter(
                            latencies,
                            agreements,
                            facecolors="none",
                            edgecolors=[
                                colormap(norm(value))
                                for value in projection_values
                            ],
                            marker=marker,
                            s=35,
                            alpha=0.9,
                            linewidths=1.0,
                            label=label,
                            zorder=3,
                        )
                    else:
                        axis.scatter(
                            latencies,
                            agreements,
                            c=projection_values,
                            cmap=colormap,
                            norm=norm,
                            marker=marker,
                            s=35,
                            alpha=0.78,
                            edgecolors="white",
                            linewidths=0.4,
                            label=label,
                            zorder=3,
                        )
        colorbar = figure.colorbar(
            ScalarMappable(norm=norm, cmap=colormap),
            ax=axis,
            pad=0.02,
        )
        colorbar.set_label("Projection dimension p")

    for aggregate in exact_candidates:
        axis.errorbar(
            [aggregate.query_microseconds[0]],
            [100.0 * aggregate.exact_agreement[0]],
            xerr=(
                [
                    aggregate.query_microseconds[0]
                    - aggregate.query_microseconds[1]
                ],
                [
                    aggregate.query_microseconds[2]
                    - aggregate.query_microseconds[0]
                ],
            ),
            yerr=(
                [
                    100.0
                    * (aggregate.exact_agreement[0] - aggregate.exact_agreement[1])
                ],
                [
                    100.0
                    * (aggregate.exact_agreement[2] - aggregate.exact_agreement[0])
                ],
            ),
            color="#009E73",
            marker="X",
            markerfacecolor=strategy_marker_facecolor(
                aggregate.configuration.strategy, "#009E73"
            ),
            markeredgecolor="#009E73",
            markeredgewidth=1.0,
            markersize=6,
            capsize=2,
            label=aggregate.configuration.label(),
            zorder=5,
        )

    exact_point = (report.exact_query_microseconds, 100.0)
    axis.scatter(
        [exact_point[0]],
        [exact_point[1]],
        marker="*",
        s=130,
        color="#222222",
        label="Exact",
        zorder=6,
    )
    axis.annotate(
        "Exact",
        exact_point,
        xytext=(-28, -13),
        textcoords="offset points",
        fontsize=8,
    )

    all_points = [exact_point] + [
        (
            aggregate.query_microseconds[0],
            100.0 * aggregate.exact_agreement[0],
        )
        for aggregate in report.aggregates
    ]
    frontier = pareto_frontier(all_points)
    axis.plot(
        [point[0] for point in frontier],
        [point[1] for point in frontier],
        linestyle="--",
        linewidth=1.1,
        color="#444444",
        alpha=0.75,
        label="Observed Pareto frontier",
        zorder=2,
    )
    latencies = [point[0] for point in all_points]
    logarithmic_latency = max(latencies) / min(latencies) >= 10.0
    if logarithmic_latency:
        axis.set_xscale("log")
    latency_scale = ", log scale" if logarithmic_latency else ""
    axis.set_xlabel(f"Query latency (µs/query{latency_scale})")
    axis.set_ylabel("Agreement with exact search (%)")
    axis.set_ylim(
        max(0.0, min(point[1] for point in all_points) - 4.0),
        101.5,
    )
    axis.set_title(
        f"Query latency versus agreement with exact search\n{report_subtitle(report)}"
    )
    axis.legend(loc="lower right", frameon=True, ncol=2)
    figure.tight_layout()
    return figure


def distance_excess(
    aggregate: Aggregate,
    statistic: str,
) -> tuple[float, float, float]:
    if aggregate.distance_ratios is None:
        raise RuntimeError(
            "distance figures require selected_distances or "
            "full_distance_table diagnostics"
        )
    return tuple(
        100.0 * (value - 1.0)
        for value in aggregate.distance_ratios[statistic]
    )


def lower_pareto_frontier(
    points: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    frontier: list[tuple[float, float]] = []
    best_error = math.inf
    for latency, error in sorted(points):
        if error < best_error:
            frontier.append((latency, error))
            best_error = error
    return frontier


def plot_latency_distance_excess(
    report: ReportData,
    statistic: str,
    compact: bool = False,
):
    assert plt is not None
    if any(aggregate.distance_ratios is None for aggregate in report.aggregates):
        raise RuntimeError(
            "distance figures require selected_distances or "
            "full_distance_table diagnostics for every comparison run"
        )

    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    styles = {
        "flat": ("#0072B2", "s", "Flat"),
        "uniform": ("#D55E00", "o", "Uniform"),
        "hierarchical": ("#6A3D9A", "D", "Hierarchy"),
    }
    compact_hierarchy = [
        aggregate
        for aggregate in report.aggregates
        if aggregate.configuration.method == "hierarchical"
    ]
    hierarchy_norm = None
    hierarchy_colormap = None
    hierarchy_sizes: dict[int, float] = {}
    if compact and compact_hierarchy:
        assert ScalarMappable is not None and Normalize is not None
        projection_dimensions = [
            aggregate.configuration.projection_dimension
            for aggregate in compact_hierarchy
        ]
        hierarchy_norm = Normalize(
            vmin=min(projection_dimensions),
            vmax=max(projection_dimensions),
        )
        hierarchy_colormap = plt.get_cmap("viridis")
        hierarchy_sizes = hierarchy_marker_sizes(compact_hierarchy)

    for method, (color, marker, label) in styles.items():
        method_aggregates = [
            aggregate
            for aggregate in report.aggregates
            if aggregate.configuration.method == method
        ]
        if not method_aggregates:
            continue
        groups = execution_groups(method_aggregates)
        for execution, group in groups:
            group.sort(
                key=lambda value: (
                    value.configuration.repetitions,
                    value.configuration.projection_dimension,
                )
            )
            latencies = [aggregate.query_microseconds[0] for aggregate in group]
            errors = [
                distance_excess(aggregate, statistic)[0] for aggregate in group
            ]
            if compact and method == "hierarchical":
                assert hierarchy_norm is not None
                assert hierarchy_colormap is not None
                projection_values = [
                    aggregate.configuration.projection_dimension
                    for aggregate in group
                ]
                sizes = [
                    hierarchy_sizes[aggregate.configuration.repetitions]
                    for aggregate in group
                ]
                series_label = execution_series_label(
                    label, execution, len(groups)
                )
                if execution[1] == "gemm":
                    axis.scatter(
                        latencies,
                        errors,
                        facecolors="none",
                        edgecolors=[
                            hierarchy_colormap(hierarchy_norm(value))
                            for value in projection_values
                        ],
                        marker="D",
                        s=sizes,
                        alpha=0.9,
                        linewidths=1.0,
                        label=series_label,
                        zorder=4,
                    )
                else:
                    axis.scatter(
                        latencies,
                        errors,
                        c=projection_values,
                        cmap=hierarchy_colormap,
                        norm=hierarchy_norm,
                        marker="D",
                        s=sizes,
                        alpha=0.78,
                        edgecolors="white",
                        linewidths=0.4,
                        label=series_label,
                        zorder=4,
                    )
                continue
            axis.plot(
                latencies,
                errors,
                color=color,
                linestyle=strategy_line_style(execution[1]),
                marker=marker,
                markerfacecolor=strategy_marker_facecolor(
                    execution[1], color
                ),
                markeredgecolor=color,
                markeredgewidth=1.0,
                linewidth=2.0 if method != "hierarchical" else 1.2,
                markersize=5,
                alpha=1.0 if method != "hierarchical" else 0.78,
                label=execution_series_label(label, execution, len(groups)),
                zorder=5 if method == "flat" else 4,
            )
            axis.errorbar(
                latencies,
                errors,
                xerr=(
                    [
                        aggregate.query_microseconds[0]
                        - aggregate.query_microseconds[1]
                        for aggregate in group
                    ],
                    [
                        aggregate.query_microseconds[2]
                        - aggregate.query_microseconds[0]
                        for aggregate in group
                    ],
                ),
                yerr=(
                    [
                        distance_excess(aggregate, statistic)[0]
                        - distance_excess(aggregate, statistic)[1]
                        for aggregate in group
                    ],
                    [
                        distance_excess(aggregate, statistic)[2]
                        - distance_excess(aggregate, statistic)[0]
                        for aggregate in group
                    ],
                ),
                fmt="none",
                ecolor=color,
                elinewidth=0.8,
                capsize=2,
                alpha=0.45,
                zorder=3,
            )
            if not compact:
                for offset, aggregate in enumerate(group):
                    annotation = f"T={aggregate.configuration.repetitions}"
                    if method == "hierarchical":
                        annotation += (
                            f", p={aggregate.configuration.projection_dimension}"
                        )
                    axis.annotate(
                        annotation,
                        (
                            aggregate.query_microseconds[0],
                            distance_excess(aggregate, statistic)[0],
                        ),
                        xytext=(4, 5 if offset % 2 == 0 else -11),
                        textcoords="offset points",
                        fontsize=7,
                        color=color,
                    )

    if compact and compact_hierarchy:
        assert hierarchy_norm is not None
        assert hierarchy_colormap is not None
        colorbar = figure.colorbar(
            ScalarMappable(norm=hierarchy_norm, cmap=hierarchy_colormap),
            ax=axis,
            pad=0.02,
        )
        colorbar.set_label("Projection dimension p")
        add_hierarchy_size_legend(axis, hierarchy_sizes)

    exact_candidates = [
        aggregate
        for aggregate in report.aggregates
        if aggregate.configuration.method == "exact"
    ]
    for aggregate in exact_candidates:
        error = distance_excess(aggregate, statistic)
        axis.errorbar(
            [aggregate.query_microseconds[0]],
            [error[0]],
            xerr=(
                [
                    aggregate.query_microseconds[0]
                    - aggregate.query_microseconds[1]
                ],
                [
                    aggregate.query_microseconds[2]
                    - aggregate.query_microseconds[0]
                ],
            ),
            yerr=([error[0] - error[1]], [error[2] - error[0]]),
            color="#009E73",
            marker="X",
            markerfacecolor=strategy_marker_facecolor(
                aggregate.configuration.strategy, "#009E73"
            ),
            markeredgecolor="#009E73",
            markeredgewidth=1.0,
            markersize=6,
            capsize=2,
            label=aggregate.configuration.label(),
            zorder=5,
        )

    exact_point = (report.exact_query_microseconds, 0.0)
    axis.scatter(
        [exact_point[0]],
        [exact_point[1]],
        marker="*",
        s=130,
        color="#222222",
        label="Reference",
        zorder=6,
    )
    axis.annotate(
        "Reference",
        exact_point,
        xytext=(-42, 7),
        textcoords="offset points",
        fontsize=8,
    )

    all_points = [exact_point] + [
        (
            aggregate.query_microseconds[0],
            distance_excess(aggregate, statistic)[0],
        )
        for aggregate in report.aggregates
    ]
    frontier = lower_pareto_frontier(all_points)
    axis.plot(
        [point[0] for point in frontier],
        [point[1] for point in frontier],
        linestyle="--",
        linewidth=1.1,
        color="#444444",
        alpha=0.75,
        label="Observed Pareto frontier",
        zorder=2,
    )
    axis.axhline(0.0, linewidth=0.8, color="#666666", alpha=0.5, zorder=1)

    latencies = [point[0] for point in all_points]
    logarithmic_latency = max(latencies) / min(latencies) >= 10.0
    if logarithmic_latency:
        axis.set_xscale("log")
    latency_scale = ", log scale" if logarithmic_latency else ""
    errors = [point[1] for point in all_points]
    error_span = max(errors) - min(errors)
    padding = 0.06 * error_span if error_span > 0.0 else 0.1
    axis.set_ylim(
        min(0.0, min(errors) - padding),
        max(errors) + padding,
    )
    statistic_label = DISTANCE_STATISTIC_LABELS[statistic]
    axis.set_xlabel(f"Query latency (µs/query{latency_scale})")
    axis.set_ylabel(
        f"{statistic_label} relative distance excess (%)\n"
        r"$100\,(d_\mathrm{returned}/d_\mathrm{reference}-1)$"
    )
    axis.set_title(f"Latency–distance trade-off\n{report_subtitle(report)}")
    axis.legend(loc="upper right", frameon=True, ncol=2)
    figure.tight_layout()
    return figure


def save_figure(
    figure,
    output_directory: Path,
    filename: str,
    formats: list[str] | tuple[str, ...],
    dpi: int,
) -> list[Path]:
    assert plt is not None
    paths: list[Path] = []
    for output_format in formats:
        path = output_directory / f"{filename}.{output_format}"
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(path)
    plt.close(figure)
    return paths


def main() -> int:
    args = parse_args()
    try:
        if MATPLOTLIB_IMPORT_ERROR is not None:
            raise RuntimeError(
                "Matplotlib is required; install requirements-figures.txt"
            ) from MATPLOTLIB_IMPORT_ERROR
        if args.max_lines <= 0:
            raise RuntimeError("--max-lines must be positive")
        if args.dpi <= 0:
            raise RuntimeError("--dpi must be positive")
        report_path = args.report.expanduser().resolve()
        report = load_report(report_path, args.batch_size)
        if "margin" in args.figures and not report.margin_labels:
            raise RuntimeError("margin figures require full_distance_table diagnostics")
        output_directory = args.output_dir.expanduser().resolve()
        output_directory.mkdir(parents=True, exist_ok=True)
        prefix = args.prefix or report_path.stem
        selected: list[Aggregate] = []
        if "margin" in args.figures or "failure" in args.figures:
            selected = select_line_aggregates(
                report,
                args.line_method,
                args.repetitions,
                args.projection_dimensions,
                args.max_lines,
            )
        apply_style()

        written: list[Path] = []
        if "margin" in args.figures:
            written.extend(
                save_figure(
                    plot_margin_error(report, selected),
                    output_directory,
                    f"{prefix}_margin_error",
                    args.formats,
                    args.dpi,
                )
            )
        if "failure" in args.figures:
            written.extend(
                save_figure(
                    plot_approximation_failures(report, selected),
                    output_directory,
                    f"{prefix}_approximation_failures",
                    args.formats,
                    args.dpi,
                )
            )
        if "pareto" in args.figures:
            written.extend(
                save_figure(
                    plot_latency_agreement_pareto(report, args.compact),
                    output_directory,
                    f"{prefix}_latency_agreement_pareto",
                    args.formats,
                    args.dpi,
                )
            )
        if "distance" in args.figures:
            written.extend(
                save_figure(
                    plot_latency_distance_excess(
                        report,
                        args.distance_statistic,
                        args.compact,
                    ),
                    output_directory,
                    f"{prefix}_latency_distance_excess_"
                    f"{args.distance_statistic}",
                    args.formats,
                    args.dpi,
                )
            )
        for path in written:
            print(path)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
