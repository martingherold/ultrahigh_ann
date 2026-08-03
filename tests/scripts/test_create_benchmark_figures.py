#!/usr/bin/env python3
"""Tests for schema-version 4 benchmark figure generation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "create_benchmark_figures.py"


def approximation(agreement: float) -> dict[str, object]:
    margin_rates = (0.4, 0.1, 0.02, 0.0, 0.0)
    bounds: tuple[tuple[float, float | None], ...] = (
        (1.0, 1.01),
        (1.01, 1.05),
        (1.05, 1.10),
        (1.10, 1.25),
        (1.25, None),
    )
    return {
        "optimal_representative_rate": agreement,
        "distance_ratio": {
            "distribution": {
                "count": 10,
                "mean": 1.01,
                "median": 1.0,
                "percentile_95": 1.05,
                "percentile_99": 1.08,
                "maximum": 1.1,
            }
        },
        "by_multiplicative_margin": [
            {
                "lower_inclusive": lower,
                "upper_exclusive": upper,
                "query_count": 2,
                "non_optimal_count": round(2 * rate),
                "non_optimal_rate": rate,
            }
            for (lower, upper), rate in zip(bounds, margin_rates)
        ],
        "approximation_guarantee_failures": [
            {
                "epsilon": epsilon,
                "violation_count": round(10 * rate),
                "violation_rate": rate,
            }
            for epsilon, rate in (
                (0.01, 0.08),
                (0.05, 0.02),
                (0.10, 0.0),
                (0.20, 0.0),
            )
        ],
    }


def run(
    name: str,
    method: str,
    repetitions: int,
    seed: int,
    query_microseconds: float,
    agreement: float,
    projection_dimension: int | None = None,
) -> dict[str, object]:
    settings: dict[str, int] = {
        "repetitions": repetitions,
        "seed": seed,
    }
    if projection_dimension is not None:
        settings["projection_dimension"] = projection_dimension
    return {
        "name": name,
        "method": method,
        "settings": settings,
        "result": {
            "microseconds_per_query": query_microseconds,
            "accuracy": agreement - 0.05,
            "agreement_with_exact": agreement,
            "approximation": approximation(agreement),
        },
    }


def fixture_report() -> dict[str, object]:
    bounds: tuple[tuple[float, float | None], ...] = (
        (1.0, 1.01),
        (1.01, 1.05),
        (1.05, 1.10),
        (1.10, 1.25),
        (1.25, None),
    )
    return {
        "schema_version": 4,
        "dataset": {
            "directory": "/tmp/fixture_dataset",
            "representative_count": 33,
            "dimension": 100,
        },
        "settings": {"distance": "l2"},
        "exact": {
            "microseconds_per_query": 1000.0,
            "accuracy": 0.9,
        },
        "diagnostics": {
            "query_geometry": {
                "query_count": 10,
                "multiplicative_margin": {
                    "buckets": [
                        {
                            "lower_inclusive": lower,
                            "upper_exclusive": upper,
                            "query_count": 2,
                        }
                        for lower, upper in bounds
                    ]
                },
            }
        },
        "runs": [
            run("flat_64_7", "flat", 64, 7, 50.0, 0.95),
            run("flat_64_42", "flat", 64, 42, 55.0, 0.94),
            run("flat_128_7", "flat", 128, 7, 90.0, 0.98),
            run("flat_128_42", "flat", 128, 42, 95.0, 0.97),
            run("uniform_64_7", "uniform", 64, 7, 45.0, 0.80),
            run("uniform_64_42", "uniform", 64, 42, 47.0, 0.82),
            run("hier_64_4", "hierarchical", 64, 7, 20.0, 0.70, 4),
            run("hier_64_8", "hierarchical", 64, 7, 30.0, 0.85, 8),
        ],
    }


class CreateBenchmarkFiguresTest(unittest.TestCase):
    def test_creates_all_png_figures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            report = root / "report.json"
            output = root / "figures"
            report.write_text(
                json.dumps(fixture_report()),
                encoding="utf-8",
            )
            environment = dict(os.environ)
            environment["MPLCONFIGDIR"] = str(root / "matplotlib")
            completed = subprocess.run(
                (
                    sys.executable,
                    str(SCRIPT),
                    "--report",
                    str(report),
                    "--output-dir",
                    str(output),
                    "--formats",
                    "png",
                    "--dpi",
                    "80",
                ),
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            expected = (
                "report_margin_error.png",
                "report_approximation_failures.png",
                "report_latency_fidelity_pareto.png",
            )
            for filename in expected:
                figure = output / filename
                self.assertTrue(figure.is_file(), filename)
                self.assertGreater(figure.stat().st_size, 1000, filename)

    def test_rejects_an_old_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            report = root / "report.json"
            report.write_text(
                json.dumps({"schema_version": 3}),
                encoding="utf-8",
            )
            environment = dict(os.environ)
            environment["MPLCONFIGDIR"] = str(root / "matplotlib")
            completed = subprocess.run(
                (
                    sys.executable,
                    str(SCRIPT),
                    "--report",
                    str(report),
                    "--formats",
                    "png",
                ),
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("schema-version 4", completed.stderr)

    def test_selects_flat_and_uniform_line_curves(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            report = root / "report.json"
            output = root / "figures"
            report.write_text(
                json.dumps(fixture_report()),
                encoding="utf-8",
            )
            environment = dict(os.environ)
            environment["MPLCONFIGDIR"] = str(root / "matplotlib")
            completed = subprocess.run(
                (
                    sys.executable,
                    str(SCRIPT),
                    "--report",
                    str(report),
                    "--output-dir",
                    str(output),
                    "--formats",
                    "png",
                    "--figures",
                    "failure",
                    "--line-method",
                    "flat-uniform",
                    "--max-lines",
                    "3",
                ),
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(
                (output / "report_approximation_failures.png").is_file()
            )


if __name__ == "__main__":
    unittest.main()
