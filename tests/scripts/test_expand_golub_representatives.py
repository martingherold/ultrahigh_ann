#!/usr/bin/env python3
"""Integration tests for Golub representative expansion."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "data" / "expand_golub_representatives.py"


class ExpandGolubRepresentativesTest(unittest.TestCase):
    def create_fixture(self, directory: Path) -> None:
        directory.mkdir()
        class_zero = np.asarray(
            (
                (0.0, 1.0, 2.0, 3.0),
                (1.0, 2.0, 3.0, 4.0),
                (10.0, 11.0, 12.0, 13.0),
                (11.0, 12.0, 13.0, 14.0),
            ),
            dtype="<f4",
        )
        class_one = class_zero + np.float32(100.0)
        pool = np.vstack(
            (
                class_zero[[0, 2]],
                class_one[[0, 2]],
                class_zero[[1, 3]],
                class_one[[1, 3]],
            )
        )
        pool_labels = np.asarray((0, 0, 1, 1, 0, 0, 1, 1), dtype="<u2")
        queries = np.asarray(
            ((0.25, 1.25, 2.25, 3.25), (110.25, 111.25, 112.25, 113.25)),
            dtype="<f4",
        )
        np.save(directory / "representative_pool.npy", pool, allow_pickle=False)
        np.save(
            directory / "representative_pool_labels.npy",
            pool_labels,
            allow_pickle=False,
        )
        np.save(directory / "queries.npy", queries, allow_pickle=False)
        np.save(
            directory / "query_labels.npy",
            np.asarray((0, 1), dtype="<u2"),
            allow_pickle=False,
        )
        np.save(
            directory / "normalization_mean.npy",
            np.zeros(4, dtype="<f4"),
            allow_pickle=False,
        )
        np.save(
            directory / "normalization_scale.npy",
            np.ones(4, dtype="<f4"),
            allow_pickle=False,
        )
        with (directory / "representatives.csv").open(
            "w", encoding="utf-8", newline=""
        ) as output:
            writer = csv.DictWriter(
                output,
                fieldnames=(
                    "representative_row",
                    "label",
                    "diagnosis",
                    "construction",
                    "source_sample_count",
                ),
            )
            writer.writeheader()
            writer.writerow(
                {
                    "representative_row": 0,
                    "label": 0,
                    "diagnosis": "ALL",
                    "construction": "arithmetic_mean_centroid",
                    "source_sample_count": 4,
                }
            )
            writer.writerow(
                {
                    "representative_row": 1,
                    "label": 1,
                    "diagnosis": "AML",
                    "construction": "arithmetic_mean_centroid",
                    "source_sample_count": 4,
                }
            )
        for name, contents in (
            ("features.csv", "feature_index,affymetrix_probe_id\n0,probe_0\n"),
            ("samples.csv", "sample_id,label\n1,0\n"),
            ("source_manifest.json", "{}\n"),
            ("dataset.json", '{"fixture": true}\n'),
        ):
            (directory / name).write_text(contents, encoding="utf-8")

    def run_script(
        self, source: Path, output: Path, total: int
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            (
                sys.executable,
                str(SCRIPT),
                "--input-dir",
                str(source),
                "--output-dir",
                str(output),
                "--total-representatives",
                str(total),
                "--clustering-features",
                "2",
                "--feature-block-size",
                "2",
                "--seed",
                "7",
            ),
            check=False,
            capture_output=True,
            text=True,
        )

    def test_learns_even_full_dimensional_subcentroids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            output = root / "expanded"
            self.create_fixture(source)

            result = self.run_script(source, output, total=4)
            self.assertEqual(result.returncode, 0, result.stderr)

            representatives = np.load(
                output / "representatives.npy", allow_pickle=False
            )
            labels = np.load(
                output / "representative_labels.npy", allow_pickle=False
            )
            self.assertEqual(representatives.shape, (4, 4))
            np.testing.assert_array_equal(labels, (0, 0, 1, 1))
            metadata = json.loads(
                (output / "dataset.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["representatives"]["total_count"], 4)
            self.assertEqual(metadata["representatives"]["count_per_class"], 2)
            self.assertIn("published training rows", metadata["leakage_control"])
            self.assertTrue((output / "source_dataset.json").is_file())
            self.assertFalse((output / "representative_pool.npy").exists())

            second = self.run_script(source, output, total=4)
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("Refusing to overwrite", second.stderr)

    def test_rejects_more_subcentroids_than_smallest_class(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            self.create_fixture(source)
            result = self.run_script(source, root / "expanded", total=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("smallest training class has only 4 samples", result.stderr)


if __name__ == "__main__":
    unittest.main()
