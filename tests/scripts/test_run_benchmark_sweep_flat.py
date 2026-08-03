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
SCRIPT = PROJECT_ROOT / "scripts" / "run_benchmark_sweep.py"


FAKE_BENCHMARK = r"""#!/usr/bin/env python3
import json
import sys
from pathlib import Path

setup_path = Path(sys.argv[sys.argv.index("--setup") + 1]).resolve()
output_path = None
runs = []

def approximation(agreement):
    optimal = round(agreement * 10)
    non_optimal = 10 - optimal
    distribution = {
        "count": 10,
        "mean": 1.01,
        "median": 1.0,
        "percentile_95": 1.05,
        "percentile_99": 1.05,
        "maximum": 1.05,
    }
    return {
        "optimal_representative_count": optimal,
        "optimal_representative_rate": agreement,
        "non_optimal_count": non_optimal,
        "non_optimal_rate": 1.0 - agreement,
        "distance_ratio": {
            "zero_optimum_queries_excluded": 0,
            "distribution": distribution,
        },
        "non_optimal_distance_ratio": {
            "eligible_non_optimal_count": non_optimal,
            "zero_optimum_non_optimal_count": 0,
            "mean": 1.05 if non_optimal else None,
            "mean_relative_excess": 0.05 if non_optimal else None,
        },
        "approximation_guarantee_failures": [
            {"epsilon": epsilon, "violation_rate": 1.0 - agreement}
            for epsilon in (0.01, 0.05, 0.10, 0.20)
        ],
        "returned_representative_rank": {"distribution": distribution},
    }

for raw_line in setup_path.read_text(encoding="utf-8").splitlines():
    fields = raw_line.split("\t")
    if fields[0] == "output":
        configured = Path(fields[1])
        output_path = (
            configured if configured.is_absolute() else setup_path.parent / configured
        ).resolve()
    elif fields[0] == "run":
        settings = dict(field.split("=", 1) for field in fields[3:])
        repetitions = int(settings["repetitions"])
        seed = int(settings["seed"])
        agreement = 0.90 + repetitions / 10_000 + seed / 100_000
        runs.append(
            {
                "name": fields[1],
                "method": fields[2],
                "settings": {key: int(value) for key, value in settings.items()},
                "result": {
                    "build_ms": 3.0,
                    "query_total_ms": 2.0,
                    "microseconds_per_query": 200.0,
                    "correct": 8,
                    "query_count": 10,
                    "accuracy": 0.8,
                    "agreement_with_exact": agreement,
                    "approximation": approximation(agreement),
                    "space": {"index_payload_bytes": 1_000},
                    "coordinate_access": {
                        "unique_coordinates": repetitions // 2,
                        "dimension_fraction": repetitions / 200,
                        "sampled_multiplicity": repetitions * 4,
                    },
                },
            }
        )

report = {
    "schema_version": 4,
    "setup_file": str(setup_path),
    "dataset": {"representative_count": 33, "dimension": 100},
    "exact": {
        "build_ms": 2.0,
        "query_total_ms": 10.0,
        "microseconds_per_query": 1000.0,
        "correct": 8,
        "query_count": 10,
        "accuracy": 0.8,
        "agreement_with_exact": None,
        "space": {"index_payload_bytes": 10_000},
    },
    "runs": runs,
}
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(report), encoding="utf-8")
"""


class RunBenchmarkSweepFlatTest(unittest.TestCase):
    def create_inputs(
        self,
        root: Path,
    ) -> tuple[Path, Path, Path]:
        benchmark = root / "fake_benchmark.py"
        benchmark.write_text(FAKE_BENCHMARK, encoding="utf-8")
        benchmark.chmod(0o755)
        report = root / "batch.json"
        setup = root / "setup.tsv"
        setup.write_text(
            "ultrahigh_ann_benchmark_setup_v1\n"
            "dataset\tdataset\n"
            f"output\t{report.name}\n"
            "max_queries\t0\n"
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
                "ultrahigh_ann_benchmark_setup_v1\n"
                "dataset\tdataset\n"
                "output\tbatch.json\n"
                "max_queries\t0\n"
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
