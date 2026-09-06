#!/usr/bin/env python3
"""Tests for joining compatible schema-version 5 checkpoint reports."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "reporting" / "join_benchmark_reports.py"
CSV_FIELDS = (
    "run_name",
    "reference",
    "batch_size",
    "warmups",
    "trial",
    "build_ms",
    "elapsed_ms",
)


def exact_run(elapsed_ms: float, build_ms: float) -> dict[str, object]:
    query_count = 10
    return {
        "name": "exact_cuda_direct",
        "index": "exact",
        "backend": "cuda",
        "strategy": "direct",
        "reference": True,
        "repetitions": 0,
        "seed": 0,
        "projection_dimension": 0,
        "warmups": 1,
        "trials": 1,
        "build_ms": build_ms,
        "space": {"index_payload_bytes": 400},
        "device": {"ordinal": 0, "name": "fixture GPU"},
        "measurements": [
            {
                "batch_size": 128,
                "workspace_payload_bytes": 100,
                "trial_ms": [elapsed_ms],
                "median_ms": elapsed_ms,
                "median_microseconds_per_query": (
                    elapsed_ms * 1000.0 / query_count
                ),
                "median_queries_per_second": (
                    query_count * 1000.0 / elapsed_ms
                ),
                "exact_choice_agreement_count": query_count,
                "exact_choice_agreement": 1.0,
                "correct": 9,
                "accuracy": 0.9,
                "label_agreement_with_exact": 1.0,
                "approximation": None,
            }
        ],
    }


def write_checkpoint(
    root: Path,
    stem: str,
    approximate_name: str,
    elapsed_ms: float,
    build_ms: float,
    dimension: int = 100,
) -> Path:
    report_path = root / f"{stem}.json"
    csv_path = root / f"{stem}.csv"
    representatives_sha256 = "a" * 64
    report = {
        "schema_version": 5,
        "generated_at_utc": "2026-09-06T00:00:00Z",
        "setup_file": f"experiments/{stem}.tsv",
        "provenance": {
            "source": {
                "project_version": "0.2.0",
                "git": {"available": True, "commit": "a" * 40, "dirty": False},
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
                    "enabled": True,
                    "compiler_id": "NVIDIA",
                    "compiler_version": "13.3.73",
                    "architectures": "75",
                    "flags": "-O3 -DNDEBUG",
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
            "cuda_runtime": {
                "device": 0,
                "device_name": "fixture GPU",
                "compute_capability": "7.5",
                "total_global_memory_bytes": 4_000_000_000,
                "compiled_runtime": {"encoded": 13030, "version": "13.3"},
                "runtime": {"encoded": 13030, "version": "13.3"},
                "driver": {"encoded": 13030, "version": "13.3"},
            },
            "invocation": {
                "arguments": ["--setup", f"experiments/{stem}.tsv"],
                "setup_sha256": hashlib.sha256(stem.encode()).hexdigest(),
            },
        },
        "outputs": {
            "json": str(report_path),
            "csv": str(csv_path),
        },
        "dataset": {
            "directory": "/tmp/fixture_dataset",
            "representatives_sha256": representatives_sha256,
            "representative_labels_sha256": "b" * 64,
            "queries_sha256": "c" * 64,
            "query_labels_sha256": "d" * 64,
            "representative_count": 33,
            "query_count_available": 10,
            "query_count_run": 10,
            "dimension": dimension,
            "labels_available": True,
            "hash_ms": 0.5,
            "load_ms": 2.0,
        },
        "settings": {
            "distance": "l2",
            "reference_run": "exact_cuda_direct",
            "device": 0,
            "diagnostics": "selected_distances",
        },
        "sampling_probabilities": {
            "required": True,
            "policy": "load",
            "source_file": "/tmp/probabilities.txt",
            "source_sha256": "e" * 64,
            "representatives_sha256": representatives_sha256,
            "coordinate_count": dimension,
            "sampling_mass": 1.0,
            "build_ms": 3.0,
            "load_ms": 4.0,
        },
        "diagnostic_execution": {
            "build_ms": 5.0,
            "evaluation_ms": 6.0,
            "payload_bytes": 800,
            "query_geometry": None,
        },
        "runs": [
            exact_run(elapsed_ms, build_ms),
            {
                "name": approximate_name,
                "index": "flat",
                "reference": False,
            },
        ],
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "run_name": "exact_cuda_direct",
                "reference": "1",
                "batch_size": "128",
                "warmups": "1",
                "trial": "0",
                "build_ms": str(build_ms),
                "elapsed_ms": str(elapsed_ms),
            }
        )
        writer.writerow(
            {
                "run_name": approximate_name,
                "reference": "0",
                "batch_size": "128",
                "warmups": "1",
                "trial": "0",
                "build_ms": "7.0",
                "elapsed_ms": "8.0",
            }
        )
    return report_path


class JoinBenchmarkReportsTest(unittest.TestCase):
    def invoke(
        self,
        reports: list[Path],
        output: Path,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            (
                sys.executable,
                str(SCRIPT),
                "--reports",
                *(str(report) for report in reports),
                "--output",
                str(output),
            ),
            check=False,
            capture_output=True,
            text=True,
        )

    def test_joins_runs_reference_trials_and_csv_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = write_checkpoint(root, "first", "flat_a", 10.0, 1.0)
            second = write_checkpoint(root, "second", "flat_b", 14.0, 3.0)
            output = root / "joined.json"

            completed = self.invoke([second, first], output)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            joined = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(
                [run["name"] for run in joined["runs"]],
                ["exact_cuda_direct", "flat_b", "flat_a"],
            )
            reference = joined["runs"][0]
            self.assertEqual(reference["warmups"], 2)
            self.assertEqual(reference["trials"], 2)
            self.assertEqual(reference["build_ms"], 2.0)
            measurement = reference["measurements"][0]
            self.assertEqual(measurement["trial_ms"], [14.0, 10.0])
            self.assertEqual(measurement["median_ms"], 12.0)
            self.assertEqual(
                measurement["median_microseconds_per_query"],
                1200.0,
            )
            self.assertEqual(joined["dataset"]["load_ms"], 4.0)
            self.assertEqual(joined["dataset"]["hash_ms"], 1.0)
            self.assertEqual(
                joined["provenance"]["invocation"]["tool"],
                "scripts/reporting/join_benchmark_reports.py",
            )
            self.assertEqual(
                len(joined["joined_checkpoints"]["source_reports"]),
                2,
            )

            with output.with_suffix(".csv").open(
                encoding="utf-8", newline=""
            ) as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), 4)
            self.assertEqual(
                [row["run_name"] for row in rows],
                ["exact_cuda_direct", "exact_cuda_direct", "flat_b", "flat_a"],
            )
            self.assertEqual([row["trial"] for row in rows[:2]], ["0", "1"])
            self.assertTrue(all(row["warmups"] == "2" for row in rows[:2]))
            self.assertTrue(all(row["build_ms"] == "2" for row in rows[:2]))

    def test_rejects_incompatible_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = write_checkpoint(root, "first", "flat_a", 10.0, 1.0)
            second = write_checkpoint(
                root,
                "second",
                "flat_b",
                14.0,
                3.0,
                dimension=101,
            )

            completed = self.invoke([first, second], root / "joined.json")

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("different dataset", completed.stderr)

    def test_rejects_duplicate_approximate_run_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = write_checkpoint(root, "first", "flat", 10.0, 1.0)
            second = write_checkpoint(root, "second", "flat", 14.0, 3.0)

            completed = self.invoke([first, second], root / "joined.json")

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("duplicate non-reference run name", completed.stderr)

    def test_rejects_incompatible_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = write_checkpoint(root, "first", "flat_a", 10.0, 1.0)
            second = write_checkpoint(root, "second", "flat_b", 14.0, 3.0)
            report = json.loads(second.read_text(encoding="utf-8"))
            report["provenance"]["source"]["git"]["commit"] = "b" * 40
            second.write_text(json.dumps(report), encoding="utf-8")

            completed = self.invoke([first, second], root / "joined.json")

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("different provenance", completed.stderr)


if __name__ == "__main__":
    unittest.main()
