#!/usr/bin/env python3
"""Flat-reporting integration tests for the generic sweep runner."""

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
FAKE_BENCHMARK_SOURCE = PROJECT_ROOT / "tests" / "scripts" / "fake_schema5_benchmark.py"


class RunBenchmarkSweepFlatTest(unittest.TestCase):
    def create_inputs(
        self,
        root: Path,
    ) -> tuple[Path, Path, Path]:
        benchmark = root / "fake_benchmark.py"
        benchmark.write_text(
            FAKE_BENCHMARK_SOURCE.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        benchmark.chmod(0o755)
        report = root / "batch.json"
        setup = root / "setup.tsv"
        setup.write_text(
            "ultrahigh_ann_benchmark_setup_v2\n"
            "dataset\tdataset\n"
            f"json_output\t{report.name}\n"
            "csv_output\tbatch.csv\n"
            "reference\texact\n"
            "diagnostics\tselected_distances\n"
            "max_queries\t0\n"
            "run\texact\texact\n"
            "run\tflat_t128_seed7\tflat\trepetitions=128\tseed=7\n"
            "run\tflat_t128_seed42\tflat\trepetitions=128\tseed=42\n"
            "run\tflat_t256_seed7\tflat\trepetitions=256\tseed=7\n"
            "run\tflat_t256_seed42\tflat\trepetitions=256\tseed=42\n",
            encoding="utf-8",
        )
        return benchmark, setup, report

    def test_runs_once_and_writes_aggregate_summaries(self) -> None:
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
            self.assertIn("Flat sweep aggregate means", result.stdout)
            summary = json.loads(
                (root / "batch_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["source_report"], str(report))
            self.assertEqual(len(summary["runs"]), 4)
            self.assertEqual(len(summary["aggregates"]), 2)
            self.assertEqual(summary["runs"][0]["mean_distance_ratio"], 1.01)
            self.assertIn(
                "approximation_failure_rate_epsilon_0_05",
                summary["aggregates"][0]["metrics"],
            )
            self.assertEqual(
                summary["aggregates"][0]["metrics"]["sampling_mass_estimate"]["mean"],
                4.0,
            )
            with (root / "batch_runs.csv").open(
                encoding="utf-8",
                newline="",
            ) as source:
                run_rows = list(csv.DictReader(source))
            with (root / "batch_aggregates.csv").open(
                encoding="utf-8",
                newline="",
            ) as source:
                aggregate_rows = list(csv.DictReader(source))
            self.assertEqual(len(run_rows), 4)
            self.assertEqual(len(aggregate_rows), 2)
            self.assertEqual(float(run_rows[0]["query_speedup"]), 5.0)
            self.assertEqual(float(run_rows[0]["mean_distance_ratio"]), 1.01)

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

    def test_zero_byte_sampled_index_has_undefined_compression(self) -> None:
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
            first = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(first.returncode, 0, first.stderr)

            payload = json.loads(report.read_text(encoding="utf-8"))
            payload["runs"][1]["space"]["index_payload_bytes"] = 0
            report.write_text(json.dumps(payload), encoding="utf-8")
            second = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(second.returncode, 0, second.stderr)
            summary = json.loads(
                (root / "batch_summary.json").read_text(encoding="utf-8")
            )
            self.assertIsNone(summary["runs"][0]["index_compression"])
            self.assertEqual(
                summary["aggregates"][0]["metrics"]["index_compression"][
                    "observation_count"
                ],
                1,
            )

    def test_mixed_flat_uniform_setup_summarizes_both_methods(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            benchmark, setup, _ = self.create_inputs(root)
            with setup.open("a", encoding="utf-8") as output:
                output.write(
                    "run\tuniform_t128_seed7\tuniform"
                    "\trepetitions=128\tseed=7\n"
                    "run\tuniform_t128_seed42\tuniform"
                    "\trepetitions=128\tseed=42\n"
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
            self.assertIn("Flat sweep aggregate means", result.stdout)
            self.assertIn("Uniform sweep aggregate means", result.stdout)
            flat_summary = json.loads(
                (root / "batch_summary.json").read_text(encoding="utf-8")
            )
            uniform_summary = json.loads(
                (root / "batch_uniform_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(flat_summary["method"], "flat")
            self.assertEqual(len(flat_summary["runs"]), 4)
            self.assertEqual(uniform_summary["method"], "uniform")
            self.assertEqual(len(uniform_summary["runs"]), 2)
            self.assertIn("uniform_query_us", uniform_summary["runs"][0])

    def test_uniform_only_setup_uses_primary_summary_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            benchmark, setup, _ = self.create_inputs(root)
            setup.write_text(
                "ultrahigh_ann_benchmark_setup_v2\n"
                "dataset\tdataset\n"
                "json_output\tbatch.json\n"
                "csv_output\tbatch.csv\n"
                "reference\texact\n"
                "diagnostics\tselected_distances\n"
                "max_queries\t0\n"
                "run\texact\texact\n"
                "run\tuniform_t128_seed7\tuniform"
                "\trepetitions=128\tseed=7\n",
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
            self.assertIn("Uniform sweep aggregate means", result.stdout)
            summary = json.loads(
                (root / "batch_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["method"], "uniform")
            self.assertEqual(len(summary["runs"]), 1)


if __name__ == "__main__":
    unittest.main()
