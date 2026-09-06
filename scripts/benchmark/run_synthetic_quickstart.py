#!/usr/bin/env python3
"""Generate and execute the deterministic synthetic L2 quickstart."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GENERATOR = PROJECT_ROOT / "scripts" / "data" / "generate_synthetic_dataset.py"
DEFAULT_BENCHMARK = PROJECT_ROOT / "build" / "nearest_neighbor_benchmark"
DEFAULT_SETUP = (
    PROJECT_ROOT / "experiments" / "synthetic" / "synthetic_quickstart_l2.tsv"
)
DEFAULT_FIGURE_SCRIPT = (
    PROJECT_ROOT / "scripts" / "reporting" / "create_benchmark_figures.py"
)
FORMAT_HEADER = "ultrahigh_ann_benchmark_setup_v2"
REQUIRED_DATASET_FILES = {
    "reference_vectors.npy",
    "reference_labels.npy",
    "queries.npy",
    "query_labels.npy",
    "dataset.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the synthetic quickstart dataset when needed and run "
            "its versioned L2 benchmark setup."
        )
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=DEFAULT_BENCHMARK,
        help=f"benchmark executable (default: {DEFAULT_BENCHMARK})",
    )
    parser.add_argument(
        "--setup",
        type=Path,
        default=DEFAULT_SETUP,
        help=f"benchmark setup (default: {DEFAULT_SETUP})",
    )
    parser.add_argument(
        "--generator",
        type=Path,
        default=DEFAULT_GENERATOR,
        help=f"dataset generator (default: {DEFAULT_GENERATOR})",
    )
    parser.add_argument(
        "--force-data",
        action="store_true",
        help="regenerate the dataset even when all expected files exist",
    )
    parser.add_argument(
        "--reuse-report",
        action="store_true",
        help="skip the benchmark when its configured JSON report already exists",
    )
    parser.add_argument(
        "--figures",
        action="store_true",
        help="plot query latency versus agreement with exact search",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the generation, benchmark, and optional figure commands",
    )
    return parser.parse_args()


def setup_paths(path: Path) -> tuple[Path, Path]:
    header_seen = False
    configured: dict[str, Path] = {}
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
            fields = raw_line.rstrip("\r\n").split("\t")
            if fields[0] not in ("dataset", "json_output"):
                continue
            if len(fields) != 2 or not fields[1] or fields[0] in configured:
                raise RuntimeError(
                    f"{path}:{line_number}: invalid or duplicate {fields[0]}"
                )
            value = Path(fields[1])
            configured[fields[0]] = (
                value if value.is_absolute() else (path.parent / value).resolve()
            )
    if not header_seen:
        raise RuntimeError(f"benchmark setup is empty: {path}")
    for required in ("dataset", "json_output"):
        if required not in configured:
            raise RuntimeError(f"benchmark setup does not define {required}: {path}")
    return configured["dataset"], configured["json_output"]


def format_command(command: list[str]) -> str:
    return " ".join(command)


def run(command: list[str]) -> None:
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {completed.returncode}: "
            f"{format_command(command)}"
        )


def main() -> int:
    args = parse_args()
    try:
        setup = args.setup.expanduser().resolve()
        generator = args.generator.expanduser().resolve()
        benchmark = args.benchmark.expanduser().resolve()
        if not setup.is_file():
            raise RuntimeError(f"benchmark setup does not exist: {setup}")
        if not generator.is_file():
            raise RuntimeError(f"dataset generator does not exist: {generator}")
        dataset, report = setup_paths(setup)
        generator_command = [
            sys.executable,
            str(generator),
            "--output-dir",
            str(dataset),
        ]
        if args.force_data:
            generator_command.append("--force")
        benchmark_command = [str(benchmark), "--setup", str(setup)]
        figure_command = [
            sys.executable,
            str(DEFAULT_FIGURE_SCRIPT),
            "--report",
            str(report),
            "--figures",
            "pareto",
        ]

        if args.dry_run:
            print(f"Generate if needed: {format_command(generator_command)}")
            print(f"Benchmark: {format_command(benchmark_command)}")
            if args.figures:
                print(f"Figures: {format_command(figure_command)}")
            return 0

        complete_dataset = all(
            (dataset / name).is_file() for name in REQUIRED_DATASET_FILES
        )
        if args.force_data or not complete_dataset:
            run(generator_command)
        else:
            print(f"Reusing synthetic dataset {dataset}", flush=True)

        if not benchmark.is_file() or not os.access(benchmark, os.X_OK):
            raise RuntimeError(
                f"benchmark is missing or not executable: {benchmark}\n"
                "Build it with 'cmake -S . -B build "
                "-DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel'."
            )
        if report.is_file() and args.reuse_report and not args.force_data:
            print(f"Reusing benchmark report {report}", flush=True)
        else:
            run(benchmark_command)
        if args.figures:
            if not DEFAULT_FIGURE_SCRIPT.is_file():
                raise RuntimeError(
                    f"figure script does not exist: {DEFAULT_FIGURE_SCRIPT}"
                )
            run(figure_command)
        print(f"Synthetic quickstart report: {report}", flush=True)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
