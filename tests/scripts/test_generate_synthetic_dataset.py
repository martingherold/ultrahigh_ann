#!/usr/bin/env python3
"""Integration tests for the synthetic quickstart generator."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "data" / "generate_synthetic_dataset.py"


class GenerateSyntheticDatasetTest(unittest.TestCase):
    def run_generator(
        self,
        output: Path,
        *extra_args: str,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            (
                sys.executable,
                str(SCRIPT),
                "--output-dir",
                str(output),
                "--representatives",
                "6",
                "--queries",
                "24",
                "--dimensions",
                "96",
                "--informative-dimensions",
                "12",
                "--background-scale",
                "0.05",
                "--query-noise",
                "0.01",
                "--seed",
                "7",
                *extra_args,
            ),
            check=False,
            capture_output=True,
            text=True,
        )

    def test_generates_deterministic_loader_compatible_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = root / "first"
            second = root / "second"
            first_result = self.run_generator(first)
            second_result = self.run_generator(second)
            self.assertEqual(first_result.returncode, 0, first_result.stderr)
            self.assertEqual(second_result.returncode, 0, second_result.stderr)

            representatives = np.load(
                first / "representatives.npy", allow_pickle=False
            )
            representative_labels = np.load(
                first / "representative_labels.npy", allow_pickle=False
            )
            queries = np.load(first / "queries.npy", allow_pickle=False)
            query_labels = np.load(
                first / "query_labels.npy", allow_pickle=False
            )
            self.assertEqual(representatives.shape, (6, 96))
            self.assertEqual(queries.shape, (24, 96))
            self.assertEqual(representatives.dtype, np.dtype("<f4"))
            self.assertEqual(queries.dtype, np.dtype("<f4"))
            self.assertEqual(representative_labels.dtype, np.dtype("<u2"))
            self.assertEqual(query_labels.dtype, np.dtype("<u2"))
            np.testing.assert_array_equal(
                representative_labels,
                np.arange(6, dtype="<u2"),
            )

            squared_distances = np.sum(
                (
                    queries.astype(np.float64)[:, None, :]
                    - representatives.astype(np.float64)[None, :, :]
                )
                ** 2,
                axis=2,
            )
            np.testing.assert_array_equal(
                np.argmin(squared_distances, axis=1),
                query_labels,
            )
            for name in (
                "representatives.npy",
                "representative_labels.npy",
                "queries.npy",
                "query_labels.npy",
            ):
                np.testing.assert_array_equal(
                    np.load(first / name, allow_pickle=False),
                    np.load(second / name, allow_pickle=False),
                )

            metadata = json.loads(
                (first / "dataset.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["distance"], "l2")
            self.assertEqual(metadata["representatives"]["shape"], [6, 96])
            self.assertEqual(metadata["queries"]["shape"], [24, 96])
            self.assertEqual(metadata["queries"]["exact_target_agreement"], 1.0)
            self.assertEqual(
                metadata["model"]["informative_dimension_count"],
                12,
            )
            self.assertEqual(sum(metadata["queries"]["difficulty_counts"].values()), 24)

    def test_refuses_overwrite_and_rejects_invalid_interpolation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "dataset"
            first = self.run_generator(output)
            self.assertEqual(first.returncode, 0, first.stderr)
            second = self.run_generator(output)
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("Refusing to overwrite", second.stderr)
            forced = self.run_generator(output, "--force")
            self.assertEqual(forced.returncode, 0, forced.stderr)

            invalid = self.run_generator(
                Path(temporary_directory) / "invalid",
                "--interpolations",
                "0.1,0.5",
            )
            self.assertNotEqual(invalid.returncode, 0)
            self.assertIn("in [0, 0.5)", invalid.stderr)


if __name__ == "__main__":
    unittest.main()
