#!/usr/bin/env python3
"""Execute and summarize any versioned benchmark sweep setup."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from benchmark_reporting import (
    flatten_direct_runs,
    flatten_flat_runs,
    flatten_hierarchical_comparisons,
    load_and_validate_report,
    parse_setup,
    print_flat_aggregate_table,
    print_hierarchical_aggregate_table,
    print_uniform_aggregate_table,
    require_mapping,
    write_flat_summaries,
    write_hierarchical_summaries,
    write_uniform_summaries,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK = PROJECT_ROOT / "build" / "nearest_neighbor_benchmark"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Execute one versioned benchmark setup, computing exact predictions "
            "once, and summarize every configured approximate method."
        )
    )
    parser.add_argument(
        "--setup",
        type=Path,
        required=True,
        help="versioned benchmark setup",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=DEFAULT_BENCHMARK,
        help=f"benchmark executable (default: {DEFAULT_BENCHMARK})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rerun the benchmark even when its combined JSON report exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the benchmark command without running or writing",
    )
    return parser.parse_args()


def summarize(
    report_path: Path,
    configured_runs: list[dict[str, object]],
    report: dict[str, object],
) -> None:
    methods = {str(run["index"]) for run in configured_runs if run["index"] != "exact"}
    if "hierarchical" in methods:
        hierarchical_rows = flatten_hierarchical_comparisons(report)
        write_hierarchical_summaries(report_path, hierarchical_rows, report)
        settings = require_mapping(report.get("settings"), "settings")
        distance_label = str(settings["distance"]).upper()
        print_hierarchical_aggregate_table(
            hierarchical_rows,
            distance_label=distance_label,
        )
        if "flat" in methods:
            flat_rows = flatten_flat_runs(report)
            write_flat_summaries(
                report_path,
                flat_rows,
                report,
                filename_suffix="flat",
            )
            print_flat_aggregate_table(flat_rows)
        if "uniform" in methods:
            uniform_rows = flatten_direct_runs(report, "uniform")
            write_uniform_summaries(
                report_path,
                uniform_rows,
                report,
                filename_suffix="uniform",
            )
            print_uniform_aggregate_table(uniform_rows)
        return
    if "flat" in methods:
        flat_rows = flatten_flat_runs(report)
        write_flat_summaries(report_path, flat_rows, report)
        print_flat_aggregate_table(flat_rows)
        if "uniform" in methods:
            uniform_rows = flatten_direct_runs(report, "uniform")
            write_uniform_summaries(
                report_path,
                uniform_rows,
                report,
                filename_suffix="uniform",
            )
            print_uniform_aggregate_table(uniform_rows)
        return
    if "uniform" in methods:
        uniform_rows = flatten_direct_runs(report, "uniform")
        write_uniform_summaries(report_path, uniform_rows, report)
        print_uniform_aggregate_table(uniform_rows)
        return
    raise RuntimeError("setup contains no supported approximate runs to summarize")


def main() -> int:
    args = parse_args()
    try:
        benchmark = args.benchmark.expanduser().resolve()
        setup_path = args.setup.expanduser().resolve()
        if not benchmark.is_file() or not os.access(benchmark, os.X_OK):
            raise RuntimeError(f"benchmark is missing or not executable: {benchmark}")
        if not setup_path.is_file():
            raise RuntimeError(f"benchmark setup does not exist: {setup_path}")

        report_path, configured_runs = parse_setup(setup_path)
        command = [str(benchmark), "--setup", str(setup_path)]
        if args.dry_run:
            print(" ".join(command))
            return 0

        if report_path.exists() and not args.force:
            print(f"Reusing combined report {report_path}", flush=True)
        else:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"benchmark failed with exit code {completed.returncode}"
                )

        report = load_and_validate_report(
            report_path,
            setup_path,
            configured_runs,
        )
        summarize(report_path, configured_runs, report)
        print(f"\nWrote summaries beside {report_path}", flush=True)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
