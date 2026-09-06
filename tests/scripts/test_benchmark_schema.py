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


def valid_provenance_report() -> dict[str, object]:
    representatives_sha256 = "a" * 64
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
            "representatives_sha256": representatives_sha256,
            "representative_labels_sha256": "d" * 64,
            "queries_sha256": "e" * 64,
            "query_labels_sha256": "f" * 64,
            "labels_available": True,
        },
        "sampling_probabilities": {
            "required": True,
            "source_file": None,
            "source_sha256": None,
            "representatives_sha256": representatives_sha256,
        },
    }


class BenchmarkSchemaTest(unittest.TestCase):
    def test_valid_provenance_contract_is_accepted(self) -> None:
        validate_report_provenance(valid_provenance_report())

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
        probabilities["representatives_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "not bound"):
            validate_report_provenance(report)

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
