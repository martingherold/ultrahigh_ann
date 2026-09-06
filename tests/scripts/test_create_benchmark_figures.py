#!/usr/bin/env python3
"""Tests for schema-version 5 benchmark figure generation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "reporting" / "create_benchmark_figures.py"


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
        "distance_optimal_count": round(10 * agreement),
        "distance_optimal_rate": agreement,
        "non_optimal_count": round(10 * (1.0 - agreement)),
        "non_optimal_rate": 1.0 - agreement,
        "reference_improvement_count": 0,
        "distance_ratio": {
            "count": 10,
            "mean": 1.01,
            "median": 1.0,
            "percentile_95": 1.05,
            "percentile_99": 1.08,
            "maximum": 1.1,
        },
        "zero_optimum_query_count": 0,
        "zero_optimum_non_optimal_count": 0,
        "non_optimal_ratio_count": round(10 * (1.0 - agreement)),
        "non_optimal_distance_ratio_mean": 1.05,
        "non_optimal_relative_excess_mean": 0.05,
        "by_multiplicative_margin": [
            {
                "lower_inclusive": lower,
                "upper_exclusive": upper,
                "query_count": 2,
                "non_optimal_count": round(2 * rate),
                "ratio_eligible_count": 2,
                "mean_distance_ratio": 1.0 + rate,
            }
            for (lower, upper), rate in zip(bounds, margin_rates)
        ],
        "returned_representative_rank": None,
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
    backend: str = "cpu",
    strategy: str = "sequential",
) -> dict[str, object]:
    projection = projection_dimension or 0
    return {
        "name": name,
        "index": method,
        "backend": backend,
        "strategy": strategy,
        "reference": False,
        "repetitions": repetitions,
        "seed": seed,
        "projection_dimension": projection,
        "warmups": 0,
        "trials": 1,
        "build_ms": 1.0,
        "space": {
            "index_payload_bytes": 100,
            "query_workspace_payload_bytes": 0,
            "unique_coordinates": 10,
            "sampled_multiplicity": 10,
        },
        "device": None,
        "measurements": [
            {
                "batch_size": 1,
                "workspace_payload_bytes": 0,
                "trial_ms": [query_microseconds / 100.0],
                "median_ms": query_microseconds / 100.0,
                "median_microseconds_per_query": query_microseconds,
                "median_queries_per_second": 1_000_000 / query_microseconds,
                "exact_choice_agreement_count": round(agreement * 10),
                "exact_choice_agreement": agreement,
                "correct": round((agreement - 0.05) * 10),
                "accuracy": agreement - 0.05,
                "label_agreement_with_exact": agreement,
                "approximation": approximation(agreement),
            }
        ],
    }


def exact_run() -> dict[str, object]:
    return {
        "name": "exact",
        "index": "exact",
        "backend": "cpu",
        "strategy": "sequential",
        "reference": True,
        "repetitions": 0,
        "seed": 0,
        "projection_dimension": 0,
        "warmups": 0,
        "trials": 1,
        "build_ms": 1.0,
        "space": {
            "index_payload_bytes": 400,
            "query_workspace_payload_bytes": 0,
            "unique_coordinates": 100,
            "sampled_multiplicity": 100,
        },
        "device": None,
        "measurements": [
            {
                "batch_size": 1,
                "workspace_payload_bytes": 0,
                "trial_ms": [10.0],
                "median_ms": 10.0,
                "median_microseconds_per_query": 1000.0,
                "median_queries_per_second": 1000.0,
                "exact_choice_agreement_count": 10,
                "exact_choice_agreement": 1.0,
                "correct": 9,
                "accuracy": 0.9,
                "label_agreement_with_exact": 1.0,
                "approximation": None,
            }
        ],
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
        "schema_version": 5,
        "dataset": {
            "directory": "/tmp/fixture_dataset",
            "representative_count": 33,
            "query_count_available": 10,
            "query_count_run": 10,
            "dimension": 100,
            "labels_available": True,
        },
        "settings": {
            "distance": "l2",
            "reference_run": "exact",
            "max_queries": 0,
            "device": 0,
            "diagnostics": "full_distance_table",
        },
        "diagnostic_execution": {
            "query_geometry": {
                "query_count": 10,
                "margin_buckets": [
                    {
                        "lower_inclusive": lower,
                        "upper_exclusive": upper,
                        "query_count": 2,
                    }
                    for lower, upper in bounds
                ],
            }
        },
        "runs": [
            exact_run(),
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
                json.dumps({"schema_version": 4}),
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
            self.assertIn("schema-version 5", completed.stderr)

    def test_creates_latency_distance_excess_figure(self) -> None:
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
                    "distance",
                    "--distance-statistic",
                    "p95",
                    "--dpi",
                    "80",
                ),
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            figure = output / "report_latency_distance_excess_p95.png"
            self.assertTrue(figure.is_file())
            self.assertGreater(figure.stat().st_size, 1000)

    def test_uses_a_single_reference_measurement_as_a_fixed_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            report = root / "report.json"
            output = root / "figures"
            contents = fixture_report()
            exact_candidate = run(
                "exact_cuda_direct",
                "exact",
                0,
                0,
                25.0,
                0.99,
            )
            exact_candidate["backend"] = "cuda"
            exact_candidate["strategy"] = "direct"
            contents["runs"].append(exact_candidate)
            for raw_run in contents["runs"]:
                if not raw_run["reference"]:
                    raw_run["measurements"][0]["batch_size"] = 128
            report.write_text(json.dumps(contents), encoding="utf-8")
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
                    "distance",
                    "--batch-size",
                    "128",
                    "--dpi",
                    "80",
                ),
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(
                (output / "report_latency_distance_excess_mean.png").is_file()
            )

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
            self.assertTrue((output / "report_approximation_failures.png").is_file())

    def test_separates_direct_and_gemm_plot_series(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            report = root / "report.json"
            output = root / "figures"
            contents = fixture_report()
            contents["diagnostic_execution"]["query_geometry"] = None
            contents["runs"] = [
                exact_run(),
                run(
                    "flat_direct",
                    "flat",
                    32,
                    42,
                    50.0,
                    0.95,
                    backend="cuda",
                    strategy="direct",
                ),
                run(
                    "flat_gemm",
                    "flat",
                    32,
                    42,
                    40.0,
                    0.95,
                    backend="cuda",
                    strategy="gemm",
                ),
                run(
                    "hier_direct",
                    "hierarchical",
                    32,
                    42,
                    30.0,
                    0.92,
                    32,
                    backend="cuda",
                    strategy="direct",
                ),
                run(
                    "hier_gemm",
                    "hierarchical",
                    32,
                    42,
                    25.0,
                    0.92,
                    32,
                    backend="cuda",
                    strategy="gemm",
                ),
            ]
            report.write_text(json.dumps(contents), encoding="utf-8")
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
                    "svg",
                    "--figures",
                    "distance",
                ),
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            svg = (
                output / "report_latency_distance_excess_mean.svg"
            ).read_text(encoding="utf-8")
            for label in (
                "Flat, cuda/direct",
                "Flat, cuda/gemm",
                "Hierarchy, cuda/direct",
                "Hierarchy, cuda/gemm",
            ):
                self.assertIn(label, svg)
            self.assertIn(
                "fill: #0072b2; stroke: #0072b2",
                svg,
            )
            self.assertIn(
                "fill-opacity: 0; stroke: #0072b2",
                svg,
            )
            self.assertIn(
                "fill: #6a3d9a; fill-opacity: 0.78; stroke: #6a3d9a",
                svg,
            )
            self.assertIn(
                "fill-opacity: 0; stroke: #6a3d9a",
                svg,
            )

    def test_compact_distance_plot_keeps_points_without_point_labels(self) -> None:
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
                    "svg",
                    "--figures",
                    "distance",
                    "--compact",
                ),
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            svg = (
                output / "report_latency_distance_excess_mean.svg"
            ).read_text(encoding="utf-8")
            self.assertIn("Hierarchy size: T=64", svg)
            self.assertIn("Projection dimension p", svg)
            self.assertNotIn("T=128", svg)


if __name__ == "__main__":
    unittest.main()
