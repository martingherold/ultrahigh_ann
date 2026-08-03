#!/usr/bin/env python3
"""Create publication-oriented figures from a schema-version 4 report."""

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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = (
    PROJECT_ROOT
    / "results"
    / "raw"
    / "tcga_pancancer_l2_hierarchical_sweep.json"
)
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "results" / "plots"


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

    def label(self) -> str:
        if self.method == "flat":
            return f"Flat, T={self.repetitions}"
        if self.method == "uniform":
            return f"Uniform, T={self.repetitions}"
        return (
            f"Hierarchy, T={self.repetitions}, "
            f"p={self.projection_dimension}"
        )


@dataclass
class Observation:
    configuration: Configuration
    query_microseconds: float
    exact_agreement: float
    accuracy: float
    margin_error_rates: list[float]
    approximation_epsilons: list[float]
    approximation_failure_rates: list[float]


@dataclass
class Aggregate:
    configuration: Configuration
    query_microseconds: tuple[float, float, float]
    exact_agreement: tuple[float, float, float]
    accuracy: tuple[float, float, float]
    margin_error_rates: list[tuple[float, float, float]]
    approximation_epsilons: list[float]
    approximation_failure_rates: list[tuple[float, float, float]]
    run_count: int


@dataclass
class ReportData:
    report_path: Path
    dataset_name: str
    distance: str
    representative_count: int
    dimension: int
    exact_query_microseconds: float
    exact_accuracy: float
    margin_labels: list[str]
    margin_query_shares: list[float]
    aggregates: list[Aggregate]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create margin-error, approximation-failure, and latency/fidelity "
            "figures from a schema-version 4 benchmark report."
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
        choices=("margin", "failure", "pareto"),
        default=("margin", "failure", "pareto"),
        help="figures to produce (default: all three)",
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


def parse_observation(raw_run: object) -> Observation:
    run = require_mapping(raw_run, "run")
    method = run.get("method")
    if method not in ("flat", "uniform", "hierarchical"):
        raise RuntimeError(f"unsupported approximate method: {method!r}")
    settings = require_mapping(run.get("settings"), "run settings")
    repetitions = require_integer(settings, "repetitions")
    projection_dimension = (
        require_integer(settings, "projection_dimension")
        if method == "hierarchical"
        else 0
    )
    result = require_mapping(run.get("result"), "run result")
    approximation = require_mapping(
        result.get("approximation"),
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
    margin_rates: list[float] = []
    for raw_bucket in margin_buckets:
        bucket = require_mapping(raw_bucket, "margin bucket")
        rate = bucket.get("non_optimal_rate")
        # Empty margin buckets are plotted as zero rather than missing because
        # they contain no observed errors.
        if rate is None:
            margin_rates.append(0.0)
        else:
            margin_rates.append(require_number(bucket, "non_optimal_rate"))
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
        ),
        query_microseconds=require_number(result, "microseconds_per_query"),
        exact_agreement=require_number(result, "agreement_with_exact"),
        accuracy=require_number(result, "accuracy"),
        margin_error_rates=margin_rates,
        approximation_epsilons=epsilons,
        approximation_failure_rates=failure_rates,
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
        aggregates.append(
            Aggregate(
                configuration=configuration,
                query_microseconds=mean_range(
                    [run.query_microseconds for run in runs]
                ),
                exact_agreement=mean_range(
                    [run.exact_agreement for run in runs]
                ),
                accuracy=mean_range([run.accuracy for run in runs]),
                margin_error_rates=[
                    mean_range(
                        [run.margin_error_rates[index] for run in runs]
                    )
                    for index in range(margin_count)
                ],
                approximation_epsilons=list(epsilons),
                approximation_failure_rates=[
                    mean_range(
                        [
                            run.approximation_failure_rates[index]
                            for run in runs
                        ]
                    )
                    for index in range(len(epsilons))
                ],
                run_count=len(runs),
            )
        )
    return aggregates


def load_report(path: Path) -> ReportData:
    try:
        with path.open(encoding="utf-8") as source:
            root = require_mapping(json.load(source), "report root")
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read report {path}: {error}") from error
    if root.get("schema_version") != 4:
        raise RuntimeError("figures require a schema-version 4 report")

    dataset = require_mapping(root.get("dataset"), "dataset")
    settings = require_mapping(root.get("settings"), "settings")
    exact = require_mapping(root.get("exact"), "exact result")
    diagnostics = require_mapping(root.get("diagnostics"), "diagnostics")
    geometry = require_mapping(
        diagnostics.get("query_geometry"),
        "query geometry",
    )
    margin = require_mapping(
        geometry.get("multiplicative_margin"),
        "multiplicative margin",
    )
    raw_geometry_buckets = require_list(
        margin.get("buckets"),
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
        parse_observation(run)
        for run in require_list(root.get("runs"), "runs")
    ]
    if not observations:
        raise RuntimeError("report contains no approximate runs")
    if any(
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
        representative_count=require_integer(dataset, "representative_count"),
        dimension=require_integer(dataset, "dimension"),
        exact_query_microseconds=require_number(
            exact,
            "microseconds_per_query",
        ),
        exact_accuracy=require_number(exact, "accuracy"),
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
        and (
            repetitions is None
            or aggregate.configuration.repetitions in repetitions
        )
        and (
            aggregate.configuration.method != "hierarchical"
            or projection_dimensions is None
            or aggregate.configuration.projection_dimension
            in projection_dimensions
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


def report_subtitle(report: ReportData) -> str:
    dataset_label = report.dataset_name.replace("_", " ")
    return (
        f"{dataset_label}, {report.distance.upper()}, "
        f"r={report.representative_count}, d={report.dimension:,}"
    )


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
        means = [
            100.0 * value[0]
            for value in aggregate.approximation_failure_rates
        ]
        minima = [
            100.0 * value[1]
            for value in aggregate.approximation_failure_rates
        ]
        maxima = [
            100.0 * value[2]
            for value in aggregate.approximation_failure_rates
        ]
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
    axis.set_title(
        f"Empirical approximation-failure curves\n{report_subtitle(report)}"
    )
    axis.legend(loc="upper right", frameon=True)
    figure.tight_layout()
    return figure


def pareto_frontier(
    points: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    frontier: list[tuple[float, float]] = []
    best_fidelity = -math.inf
    for latency, fidelity in sorted(points):
        if fidelity > best_fidelity:
            frontier.append((latency, fidelity))
            best_fidelity = fidelity
    return frontier


def plot_latency_fidelity_pareto(report: ReportData):
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

    if flat:
        flat = sorted(flat, key=lambda value: value.configuration.repetitions)
        axis.plot(
            [aggregate.query_microseconds[0] for aggregate in flat],
            [100.0 * aggregate.exact_agreement[0] for aggregate in flat],
            color="#0072B2",
            marker="s",
            linewidth=2.0,
            markersize=5,
            label="Flat",
            zorder=5,
        )
        for offset, aggregate in enumerate(flat):
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
        uniform = sorted(
            uniform,
            key=lambda value: value.configuration.repetitions,
        )
        axis.plot(
            [aggregate.query_microseconds[0] for aggregate in uniform],
            [100.0 * aggregate.exact_agreement[0] for aggregate in uniform],
            color="#D55E00",
            marker="o",
            linewidth=2.0,
            markersize=5,
            label="Uniform",
            zorder=4,
        )
        for offset, aggregate in enumerate(uniform):
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
            aggregate.configuration.projection_dimension
            for aggregate in hierarchical
        ]
        norm = Normalize(
            vmin=min(projection_dimensions),
            vmax=max(projection_dimensions),
        )
        colormap = plt.get_cmap("viridis")
        marker_cycle = ("o", "^", "v", "D", "P", "X")
        repetitions = sorted(
            {
                aggregate.configuration.repetitions
                for aggregate in hierarchical
            }
        )
        for marker_index, repetition in enumerate(repetitions):
            group = [
                aggregate
                for aggregate in hierarchical
                if aggregate.configuration.repetitions == repetition
            ]
            axis.scatter(
                [aggregate.query_microseconds[0] for aggregate in group],
                [100.0 * aggregate.exact_agreement[0] for aggregate in group],
                c=[
                    aggregate.configuration.projection_dimension
                    for aggregate in group
                ],
                cmap=colormap,
                norm=norm,
                marker=marker_cycle[marker_index % len(marker_cycle)],
                s=35,
                alpha=0.78,
                edgecolors="white",
                linewidths=0.4,
                label=f"Hierarchy, T={repetition}",
                zorder=3,
            )
        colorbar = figure.colorbar(
            ScalarMappable(norm=norm, cmap=colormap),
            ax=axis,
            pad=0.02,
        )
        colorbar.set_label("Projection dimension p")

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
    axis.set_ylabel("Agreement with exact representative (%)")
    axis.set_ylim(
        max(0.0, min(point[1] for point in all_points) - 4.0),
        101.5,
    )
    axis.set_title(
        f"Latency–fidelity trade-off\n{report_subtitle(report)}"
    )
    axis.legend(loc="lower right", frameon=True, ncol=2)
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
        report = load_report(report_path)
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
                    plot_latency_fidelity_pareto(report),
                    output_directory,
                    f"{prefix}_latency_fidelity_pareto",
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
