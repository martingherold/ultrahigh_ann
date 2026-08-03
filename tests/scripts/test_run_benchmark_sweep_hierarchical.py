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
        settings = {key: int(value) for key, value in settings.items()}
        repetitions = settings["repetitions"]
        method = fields[2]
        projection = settings.get("projection_dimension", 0)
        query_us = 200.0 if method == "flat" else 100.0 + 2.0 * projection
        accuracy = 0.86 if method == "flat" else 0.85 + projection / 1000.0
        agreement = 0.98 if method == "flat" else 0.90 + projection / 200.0
        runs.append(
            {
                "name": fields[1],
                "method": method,
                "settings": settings,
                "result": {
                    "build_ms": 3.0 + projection,
                    "query_total_ms": query_us / 100.0,
                    "microseconds_per_query": query_us,
                    "correct": round(accuracy * 10),
                    "query_count": 10,
                    "accuracy": accuracy,
                    "agreement_with_exact": agreement,
                    "approximation": approximation(agreement),
                    "space": {
                        "index_payload_bytes": (
                            1_000 if method == "flat" else 500 + 50 * projection
                        ),
                        "query_workspace_payload_bytes": 8 + projection,
                    },
                    "coordinate_access": {
                        "unique_coordinates": repetitions // 2,
                        "dimension_fraction": repetitions / 200.0,
                        "sampled_multiplicity": repetitions * 4,
                    },
                },
            }
        )

report = {
    "schema_version": 4,
    "setup_file": str(setup_path),
    "dataset": {"representative_count": 33, "dimension": 100},
    "settings": {"distance": "l2", "queries_run": 10, "query_limit": None},
    "shared_preprocessing": {
        "sampling_probabilities": {
            "computed_once": True,
            "build_ms": 12.5,
            "coordinate_count": 100,
            "payload_bytes": 800,
        }
    },
    "exact": {
        "build_ms": 2.0,
        "query_total_ms": 10.0,
        "microseconds_per_query": 1000.0,
        "correct": 8,
        "query_count": 10,
        "accuracy": 0.8,
        "agreement_with_exact": None,
        "space": {
            "index_payload_bytes": 10_000,
            "query_workspace_payload_bytes": 0,
        },
    },
    "runs": runs,
}
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(report), encoding="utf-8")
"""


class RunBenchmarkSweepHierarchicalTest(unittest.TestCase):
    def test_k10_combined_setup_contains_all_unique_runs(self) -> None:
        setup = (
            PROJECT_ROOT / "experiments" / "tcga_pancancer_k10_l2_combined_sweep.tsv"
        )
        runs: set[tuple[str, int, int, int | None]] = set()
        run_count = 0
        for line in setup.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if fields[0] != "run":
                continue
            settings = dict(field.split("=", 1) for field in fields[3:])
            run = (
                fields[2],
                int(settings["repetitions"]),
                int(settings["seed"]),
                (
                    int(settings["projection_dimension"])
                    if "projection_dimension" in settings
                    else None
                ),
            )
            runs.add(run)
            run_count += 1

        self.assertEqual(run_count, 153)
        self.assertEqual(len(runs), run_count)
        seeds = {7, 42, 1312}
        self.assertEqual(
            {run for run in runs if run[0] == "flat"},
            {
                ("flat", repetitions, seed, None)
                for repetitions in (1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 128, 256, 512)
                for seed in seeds
            },
        )
        expected_projections = {
            8: {16, 32, 64, 96, 128},
            16: {16, 32, 64, 96, 128},
            32: {16, 32, 64, 96, 128},
            128: {16, 32, 64, 96, 128, 160, 192, 256, 320, 384, 448, 512, 640},
            256: {16, 32, 64, 96, 128},
            512: {16, 32, 64, 96, 128},
        }
        self.assertEqual(
            {run for run in runs if run[0] == "hierarchical"},
            {
                ("hierarchical", repetitions, seed, projection)
                for repetitions, projections in expected_projections.items()
                for seed in seeds
                for projection in projections
            },
        )

    def test_versioned_setup_contains_the_matched_parameter_grid(self) -> None:
        setup = (
            PROJECT_ROOT / "experiments" / "tcga_pancancer_l2_hierarchical_sweep.tsv"
        )
        runs: list[tuple[str, int, int, int | None]] = []
        for line in setup.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if fields[0] != "run":
                continue
            settings = dict(field.split("=", 1) for field in fields[3:])
            runs.append(
                (
                    fields[2],
                    int(settings["repetitions"]),
                    int(settings["seed"]),
                    (
                        int(settings["projection_dimension"])
                        if "projection_dimension" in settings
                        else None
                    ),
                )
            )

        flat = {run for run in runs if run[0] == "flat"}
        hierarchical = {run for run in runs if run[0] == "hierarchical"}
        self.assertEqual(len(runs), 475)
        self.assertEqual(len(flat), 75)
        self.assertEqual(len(hierarchical), 400)
        flat_repetitions = {
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            16,
            32,
            64,
            128,
            256,
            512,
            1024,
        }
        hierarchical_repetitions = {8, 16, 32, 64, 128, 256, 512, 1024}
        seeds = {7, 20, 42, 1312, 2026}
        projections = {1, 2, 4, 8, 16, 32, 48, 64, 96, 128}
        self.assertEqual(
            flat,
            {
                ("flat", repetition, seed, None)
                for repetition in flat_repetitions
                for seed in seeds
            },
        )
        self.assertEqual(
            hierarchical,
            {
                ("hierarchical", repetition, seed, projection)
                for repetition in hierarchical_repetitions
                for seed in seeds
                for projection in projections
            },
        )

    def create_inputs(self, root: Path) -> tuple[Path, Path, Path]:
        benchmark = root / "fake_benchmark.py"
        benchmark.write_text(FAKE_BENCHMARK, encoding="utf-8")
        benchmark.chmod(0o755)
        report = root / "hierarchy.json"
        setup = root / "setup.tsv"
        setup.write_text(
            "ultrahigh_ann_benchmark_setup_v1\n"
            "dataset\tdataset\n"
            f"output\t{report.name}\n"
            "distance\tl2\n"
            "max_queries\t0\n"
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
                summary["shared_preprocessing"]["sampling_probabilities"]["build_ms"],
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


if __name__ == "__main__":
    unittest.main()
