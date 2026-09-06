#!/usr/bin/env python3
"""Integration tests for GSE2034 representative expansion."""

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
SCRIPT = PROJECT_ROOT / "scripts" / "data" / "expand_gse2034_centroids.py"


class ExpandGse2034RepresentativesTest(unittest.TestCase):
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
        query_labels = np.asarray((0, 1), dtype="<u2")
        np.save(directory / "training_pool.npy", pool, allow_pickle=False)
        np.save(
            directory / "training_pool_labels.npy",
            pool_labels,
            allow_pickle=False,
        )
        np.save(directory / "queries.npy", queries, allow_pickle=False)
        np.save(directory / "query_labels.npy", query_labels, allow_pickle=False)
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
        with (directory / "reference_vectors.csv").open(
            "w",
            encoding="utf-8",
            newline="",
        ) as output:
            writer = csv.DictWriter(
                output,
                fieldnames=(
                    "reference_vector_row",
                    "label",
                    "outcome",
                    "construction",
                    "source_sample_count",
                ),
            )
            writer.writeheader()
            writer.writerow(
                {
                    "reference_vector_row": 0,
                    "label": 0,
                    "outcome": "relapse_free",
                    "construction": "arithmetic_mean_centroid",
                    "source_sample_count": 4,
                }
            )
            writer.writerow(
                {
                    "reference_vector_row": 1,
                    "label": 1,
                    "outcome": "distant_metastasis",
                    "construction": "arithmetic_mean_centroid",
                    "source_sample_count": 4,
                }
            )
        for name, contents in (
            ("features.csv", "feature_index,probe_set_id\n0,probe_0\n"),
            ("samples.csv", "geo_accession,label\nGSM1,0\n"),
            ("source_manifest.json", "{}\n"),
            ("dataset.json", '{"fixture": true}\n'),
        ):
            (directory / name).write_text(contents, encoding="utf-8")

    def run_script(
        self,
        source: Path,
        output: Path,
        *extra_args: str,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            (
                sys.executable,
                str(SCRIPT),
                "--input-dir",
                str(source),
                "--output-dir",
                str(output),
                "--centroids-per-class",
                "2",
                "--clustering-features",
                "2",
                "--feature-block-size",
                "2",
                "--seed",
                "7",
                *extra_args,
            ),
            check=False,
            capture_output=True,
            text=True,
        )

    def test_learns_full_dimensional_subcentroids_and_label_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            output = root / "expanded"
            self.create_fixture(source)

            result = self.run_script(source, output)
            self.assertEqual(result.returncode, 0, result.stderr)

            reference_vectors = np.load(
                output / "reference_vectors.npy",
                allow_pickle=False,
            )
            labels = np.load(
                output / "reference_labels.npy",
                allow_pickle=False,
            )
            self.assertEqual(reference_vectors.shape, (4, 4))
            np.testing.assert_array_equal(labels, (0, 0, 1, 1))
            expected = {
                0: np.asarray(
                    ((0.5, 1.5, 2.5, 3.5), (10.5, 11.5, 12.5, 13.5))
                ),
                1: np.asarray(
                    (
                        (100.5, 101.5, 102.5, 103.5),
                        (110.5, 111.5, 112.5, 113.5),
                    )
                ),
            }
            for label, expected_centroids in expected.items():
                actual = reference_vectors[labels == label]
                actual = actual[np.argsort(actual[:, 0])]
                np.testing.assert_allclose(actual, expected_centroids)

            with (output / "reference_vectors.csv").open(
                encoding="utf-8",
                newline="",
            ) as source_file:
                records = list(csv.DictReader(source_file))
            self.assertEqual(len(records), 4)
            self.assertEqual(
                [int(record["cluster_sample_count"]) for record in records],
                [2, 2, 2, 2],
            )
            metadata = json.loads(
                (output / "dataset.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["reference_vectors"]["shape"], [4, 4])
            self.assertEqual(metadata["reference_vectors"]["count_per_class"], 2)
            self.assertIn(
                "only training_pool.npy",
                metadata["leakage_control"],
            )
            self.assertTrue((output / "source_dataset.json").is_file())
            self.assertFalse((output / "training_pool.npy").exists())
            np.testing.assert_array_equal(
                np.load(output / "queries.npy", allow_pickle=False),
                np.load(source / "queries.npy", allow_pickle=False),
            )

            second_result = self.run_script(source, output)
            self.assertNotEqual(second_result.returncode, 0)
            self.assertIn("Refusing to overwrite", second_result.stderr)

    def test_rejects_more_clusters_than_smallest_class(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            self.create_fixture(source)
            result = subprocess.run(
                (
                    sys.executable,
                    str(SCRIPT),
                    "--input-dir",
                    str(source),
                    "--output-dir",
                    str(root / "expanded"),
                    "--centroids-per-class",
                    "5",
                ),
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("smallest pool has only 4 samples", result.stderr)


if __name__ == "__main__":
    unittest.main()
