#!/usr/bin/env python3
"""Hierarchy-reporting integration tests for the generic sweep runner."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "benchmark" / "run_benchmark_sweep.py"
FAKE_BENCHMARK_SOURCE = PROJECT_ROOT / "tests" / "scripts" / "fake_schema6_benchmark.py"


class RunBenchmarkSweepHierarchicalTest(unittest.TestCase):
    def create_inputs(self, root: Path) -> tuple[Path, Path, Path]:
        benchmark = root / "fake_benchmark.py"
        benchmark.write_text(
            FAKE_BENCHMARK_SOURCE.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        benchmark.chmod(0o755)
        report = root / "hierarchy.json"
        setup = root / "setup.tsv"
        setup.write_text(
            "ultrahigh_ann_benchmark_setup_v2\n"
            "dataset\tdataset\n"
            f"json_output\t{report.name}\n"
            "csv_output\thierarchy.csv\n"
            "reference\texact\n"
            "diagnostics\tselected_distances\n"
            "distance\tl2\n"
            "max_queries\t0\n"
            "run\texact\texact\n"
            "run\tflat_t64_seed7\tflat\trepetitions=64\tseed=7\n"
            "run\tflat_t128_seed7\tflat\trepetitions=128\tseed=7\n"
            "run\thier_t128_p4_seed7\thierarchical\trepetitions=128\tseed=7"
            "\tprojection_dimension=4\n"
            "run\thier_t128_p8_seed7\thierarchical\trepetitions=128\tseed=7"
            "\tprojection_dimension=8\n"
            "run\tflat_t128_seed42\tflat\trepetitions=128\tseed=42\n"
            "run\thier_t128_p4_seed42\thierarchical\trepetitions=128\tseed=42"
            "\tprojection_dimension=4\n"
            "run\thier_t128_p8_seed42\thierarchical\trepetitions=128\tseed=42"
            "\tprojection_dimension=8\n",
            encoding="utf-8",
        )
        return benchmark, setup, report

    def test_runs_once_and_writes_matched_aggregates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            benchmark, setup, report = self.create_inputs(root)
            command = (
                sys.executable,
                str(SCRIPT),
                "--benchmark",
                str(benchmark),
                "--setup",
                str(setup),
            )
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("L2 hierarchical sweep aggregate means", result.stdout)

            summary = json.loads(
                (root / "hierarchy_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["source_report"], str(report))
            self.assertEqual(
                summary["sampling_probabilities"]["build_ms"],
                12.5,
            )
            self.assertEqual(len(summary["runs"]), 4)
            self.assertEqual(len(summary["aggregates"]), 2)
            self.assertEqual(summary["aggregates"][0]["seeds"], [7, 42])
            self.assertEqual(
                summary["runs"][0]["hierarchical_mean_distance_ratio"],
                1.01,
            )
            flat_summary = json.loads(
                (root / "hierarchy_flat_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(flat_summary["method"], "flat")
            self.assertEqual(len(flat_summary["runs"]), 3)
            self.assertEqual(
                {run["repetitions"] for run in flat_summary["runs"]},
                {64, 128},
            )

            with (root / "hierarchy_comparisons.csv").open(
                encoding="utf-8",
                newline="",
            ) as source:
                comparisons = list(csv.DictReader(source))
            with (root / "hierarchy_aggregates.csv").open(
                encoding="utf-8",
                newline="",
            ) as source:
                aggregates = list(csv.DictReader(source))
            self.assertEqual(len(comparisons), 4)
            self.assertEqual(len(aggregates), 2)
            self.assertAlmostEqual(
                float(comparisons[0]["hierarchical_speedup_vs_flat"]),
                200.0 / 108.0,
            )
            self.assertEqual(
                float(comparisons[0]["hierarchical_mean_distance_ratio"]),
                1.01,
            )

            repeated = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(repeated.returncode, 0, repeated.stderr)
            self.assertIn("Reusing combined report", repeated.stdout)

    def test_dry_run_does_not_create_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            benchmark, setup, report = self.create_inputs(root)
            result = subprocess.run(
                (
                    sys.executable,
                    str(SCRIPT),
                    "--benchmark",
                    str(benchmark),
                    "--setup",
                    str(setup),
                    "--dry-run",
                ),
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--setup", result.stdout)
            self.assertFalse(report.exists())

    def test_matches_flat_and_hierarchy_by_cuda_strategy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            benchmark = root / "fake_benchmark.py"
            benchmark.write_text(
                FAKE_BENCHMARK_SOURCE.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            benchmark.chmod(0o755)
            setup = root / "setup.tsv"
            setup.write_text(
                "ultrahigh_ann_benchmark_setup_v2\n"
                "dataset\tdataset\n"
                "json_output\thierarchy.json\n"
                "csv_output\thierarchy.csv\n"
                "reference\texact_direct\n"
                "diagnostics\tselected_distances\n"
                "distance\tl2\n"
                "run\texact_direct\texact\tbackend=cuda\tstrategy=direct\n"
                "run\tflat_direct\tflat\tbackend=cuda\tstrategy=direct\t"
                "repetitions=32\tseed=42\n"
                "run\tflat_gemm\tflat\tbackend=cuda\tstrategy=gemm\t"
                "repetitions=32\tseed=42\n"
                "run\thier_direct\thierarchical\tbackend=cuda\t"
                "strategy=direct\trepetitions=32\tseed=42\t"
                "projection_dimension=16\n"
                "run\thier_gemm\thierarchical\tbackend=cuda\tstrategy=gemm\t"
                "repetitions=32\tseed=42\tprojection_dimension=16\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                (
                    sys.executable,
                    str(SCRIPT),
                    "--benchmark",
                    str(benchmark),
                    "--setup",
                    str(setup),
                ),
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads(
                (root / "hierarchy_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(summary["runs"]), 2)
            self.assertEqual(
                {run["strategy"] for run in summary["runs"]},
                {"direct", "gemm"},
            )
            self.assertEqual(
                {
                    (run["strategy"], run["matched_flat_name"])
                    for run in summary["runs"]
                },
                {("direct", "flat_direct"), ("gemm", "flat_gemm")},
            )


if __name__ == "__main__":
    unittest.main()
