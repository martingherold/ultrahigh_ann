#!/usr/bin/env python3
"""Regression tests for the repository-wide benchmark format contract."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "benchmark"))

from benchmark_reporting import parse_setup, validate_report_provenance  # noqa: E402


EXPECTED_ASSET_SOURCE_COMMIT = "2acd11cb1b95d94afbf99a2468a4714783573c62"
EXPECTED_REPORTS = {
    "tcga_pancancer_r33_l2_cpu_sparse_sweep.json",
    "tcga_pancancer_r33_l2_cpu_uniform_vs_flat.json",
    "tcga_pancancer_r7028_l2_cuda_crossover.json",
}
EXPECTED_FIGURES = {
    "tcga_pancancer_r33_l2_cpu_sparse_sweep_latency_agreement_pareto.png",
    "tcga_pancancer_r33_l2_cpu_uniform_vs_flat_latency_agreement_pareto.png",
    "tcga_pancancer_r7028_l2_cuda_crossover_latency_distance_excess_mean.png",
}


def valid_provenance_report() -> dict[str, object]:
    reference_vectors_sha256 = "a" * 64
    return {
        "provenance": {
            "source": {
                "project_version": "0.2.0",
                "git": {"available": True, "commit": "b" * 40, "dirty": False},
            },
            "build": {
                "cmake_version": "3.29.3",
                "build_type": "Release",
                "cxx": {
                    "standard": 20,
                    "compiler_id": "GNU",
                    "compiler_version": "12.2.0",
                    "flags": "-O3 -DNDEBUG",
                },
                "cuda": {
                    "enabled": False,
                    "compiler_id": None,
                    "compiler_version": None,
                    "architectures": None,
                    "flags": None,
                },
            },
            "host": {
                "operating_system": {
                    "name": "Linux",
                    "version": "6.1",
                    "architecture": "x86_64",
                },
                "processor_model": "fixture CPU",
                "physical_memory_bytes": 1_000_000,
            },
            "cuda_runtime": None,
            "invocation": {
                "arguments": ["--setup", "experiments/fixture.tsv"],
                "setup_sha256": "c" * 64,
            },
        },
        "dataset": {
            "reference_vectors_sha256": reference_vectors_sha256,
            "reference_labels_sha256": "d" * 64,
            "queries_sha256": "e" * 64,
            "query_labels_sha256": "f" * 64,
            "labels_available": True,
        },
        "sampling_probabilities": {
            "required": True,
            "source_file": None,
            "source_sha256": None,
            "reference_vectors_sha256": reference_vectors_sha256,
        },
    }


class BenchmarkSchemaTest(unittest.TestCase):
    def test_valid_provenance_contract_is_accepted(self) -> None:
        validate_report_provenance(valid_provenance_report())

    def test_joined_provenance_requires_source_checksums(self) -> None:
        report = valid_provenance_report()
        invocation = {
            "tool": "scripts/reporting/join_benchmark_reports.py",
            "source_reports": [{"path": "first.json", "sha256": "a" * 64}],
        }
        report["provenance"]["invocation"] = invocation
        validate_report_provenance(report)

        del invocation["source_reports"][0]["sha256"]
        with self.assertRaisesRegex(RuntimeError, "SHA-256"):
            validate_report_provenance(report)

        invocation["source_reports"] = []
        with self.assertRaisesRegex(RuntimeError, "source reports"):
            validate_report_provenance(report)

    def test_missing_provenance_is_rejected(self) -> None:
        report = valid_provenance_report()
        del report["provenance"]
        with self.assertRaisesRegex(RuntimeError, "provenance"):
            validate_report_provenance(report)

    def test_malformed_checksum_is_rejected(self) -> None:
        report = valid_provenance_report()
        dataset = report["dataset"]
        assert isinstance(dataset, dict)
        dataset["queries_sha256"] = "not-a-digest"
        with self.assertRaisesRegex(RuntimeError, "SHA-256"):
            validate_report_provenance(report)

    def test_wrong_probability_binding_is_rejected(self) -> None:
        report = copy.deepcopy(valid_provenance_report())
        probabilities = report["sampling_probabilities"]
        assert isinstance(probabilities, dict)
        probabilities["reference_vectors_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "not bound"):
            validate_report_provenance(report)

    def test_every_setup_uses_v2_and_parses(self) -> None:
        setups = sorted((PROJECT_ROOT / "experiments").rglob("*.tsv"))
        self.assertTrue(setups)
        for setup in setups:
            with self.subTest(setup=setup.relative_to(PROJECT_ROOT)):
                first = next(
                    line.strip()
                    for line in setup.read_text(encoding="utf-8").splitlines()
                    if line.strip() and not line.lstrip().startswith("#")
                )
                self.assertEqual(first, "ultrahigh_ann_benchmark_setup_v2")
                _, runs = parse_setup(setup)
                references = [run for run in runs if run["reference"]]
                self.assertEqual(len(references), 1)
                self.assertEqual(references[0]["index"], "exact")

    def test_tracked_reports_use_schema_6_only(self) -> None:
        reports = sorted((PROJECT_ROOT / "assets").glob("*.json"))
        self.assertEqual({path.name for path in reports}, EXPECTED_REPORTS)
        figures = {path.name for path in (PROJECT_ROOT / "assets").glob("*.png")}
        self.assertEqual(figures, EXPECTED_FIGURES)
        for path in reports:
            with self.subTest(report=path.name):
                report = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(report["schema_version"], 6)
                validate_report_provenance(report)
                source = report["provenance"]["source"]
                self.assertEqual(source["project_version"], "0.2.0")
                self.assertEqual(
                    source["git"],
                    {
                        "available": True,
                        "commit": EXPECTED_ASSET_SOURCE_COMMIT,
                        "dirty": False,
                    },
                )
                setup = PROJECT_ROOT / report["setup_file"]
                self.assertTrue(setup.is_file(), setup)
                outputs = report["outputs"]
                self.assertEqual(Path(outputs["json"]).name, path.name)
                self.assertEqual(
                    Path(outputs["csv"]).stem,
                    path.stem,
                )
                self.assertNotIn("exact", report)
                self.assertNotIn("results", report)
                references = [
                    run for run in report["runs"] if run.get("reference") is True
                ]
                self.assertEqual(len(references), 1)
                self.assertEqual(references[0]["index"], "exact")
                self.assertTrue(all(run.get("measurements") for run in report["runs"]))
                dataset = report["dataset"]
                labels_available = dataset["labels_available"]
                self.assertEqual(
                    dataset["reference_labels_file"] is not None,
                    labels_available,
                )
                self.assertEqual(
                    dataset["query_labels_file"] is not None,
                    labels_available,
                )
                for run in report["runs"]:
                    for measurement in run["measurements"]:
                        self.assertEqual(
                            measurement["correct"] is not None,
                            labels_available,
                        )
                        self.assertEqual(
                            measurement["accuracy"] is not None,
                            labels_available,
                        )
                        self.assertEqual(
                            measurement["label_agreement_with_exact"]
                            is not None,
                            labels_available,
                        )

    def test_v1_setup_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            setup = Path(temporary_directory) / "legacy.tsv"
            setup.write_text(
                "ultrahigh_ann_benchmark_setup_v1\n"
                "dataset\tdataset\n"
                "output\treport.json\n"
                "run\tflat\tflat\trepetitions=1\tseed=1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "setup_v2"):
                parse_setup(setup)


if __name__ == "__main__":
    unittest.main()
