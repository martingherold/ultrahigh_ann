#!/usr/bin/env python3
"""Integration test for scripts/data/transform_tcga_kidney.py."""

from __future__ import annotations

import csv
import gzip
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "data" / "transform_tcga_kidney.py"


class TransformTcgaTest(unittest.TestCase):
    def write_matrix(
        self,
        path: Path,
        sample_ids: tuple[str, ...],
        rows: tuple[tuple[str, tuple[float, ...]], ...],
    ) -> None:
        if path.suffix == ".gz":
            output_context = gzip.open(
                path, mode="wt", encoding="utf-8", newline=""
            )
        else:
            output_context = path.open(
                mode="w", encoding="utf-8", newline=""
            )
        with output_context as output:
            writer = csv.writer(output, delimiter="\t", lineterminator="\n")
            writer.writerow(("Ensembl_ID", *sample_ids))
            for feature_id, values in rows:
                writer.writerow((feature_id, *values))

    def test_transforms_filters_and_groups_participants(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw = root / "raw"
            output = root / "processed"
            raw.mkdir()

            rows_by_cohort = (
                (
                    "TCGA-KICH.star_fpkm.tsv",
                    (
                        "TCGA-AA-0001-01A",
                        "TCGA-AA-0001-11A",
                        "TCGA-AA-0002-01A",
                    ),
                    (
                        ("ENSG1.1", (1.0, 101.0, 4.0)),
                        ("ENSG2.2", (2.0, 102.0, 5.0)),
                        ("ENSG3.3", (3.0, 103.0, 6.0)),
                    ),
                ),
                (
                    "TCGA-KIRC.star_fpkm.tsv.gz",
                    (
                        "TCGA-BB-0001-01A",
                        "TCGA-BB-0001-01B",
                        "TCGA-BB-0002-01A",
                        "TCGA-BB-0003-11A",
                    ),
                    (
                        ("ENSG1.1", (10.0, 11.0, 12.0, 110.0)),
                        ("ENSG2.2", (20.0, 21.0, 22.0, 120.0)),
                        ("ENSG3.3", (30.0, 31.0, 32.0, 130.0)),
                    ),
                ),
                (
                    "TCGA-KIRP.star_fpkm.tsv",
                    (
                        "TCGA-CC-0001-11A",
                        "TCGA-CC-0002-01A",
                        "TCGA-CC-0003-01A",
                    ),
                    (
                        ("ENSG1.1", (201.0, 40.0, 70.0)),
                        ("ENSG2.2", (202.0, 50.0, 80.0)),
                        ("ENSG3.3", (203.0, 60.0, 90.0)),
                    ),
                ),
            )
            for name, sample_ids, rows in rows_by_cohort:
                self.write_matrix(raw / name, sample_ids, rows)

            source_manifest = {"fixture": True}
            (raw / "manifest.json").write_text(
                json.dumps(source_manifest), encoding="utf-8"
            )

            command = (
                sys.executable,
                str(SCRIPT),
                "--input-dir",
                str(raw),
                "--output-dir",
                str(output),
                "--expected-features",
                "3",
            )
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

            with (output / "samples.csv").open(
                encoding="utf-8", newline=""
            ) as source:
                samples = list(csv.DictReader(source))
            expected_vectors = {
                "TCGA-AA-0001-01A": (1.0, 2.0, 3.0),
                "TCGA-AA-0002-01A": (4.0, 5.0, 6.0),
                "TCGA-BB-0001-01A": (10.0, 20.0, 30.0),
                "TCGA-BB-0001-01B": (11.0, 21.0, 31.0),
                "TCGA-BB-0002-01A": (12.0, 22.0, 32.0),
                "TCGA-CC-0002-01A": (40.0, 50.0, 60.0),
                "TCGA-CC-0003-01A": (70.0, 80.0, 90.0),
            }
            self.assertEqual(
                {sample["sample_id"] for sample in samples},
                set(expected_vectors),
            )
            self.assertTrue(
                all(sample["sample_type_code"] == "01" for sample in samples)
            )

            matrices = {
                "representative_pool": np.load(
                    output / "representative_pool.npy", allow_pickle=False
                ),
                "query": np.load(output / "queries.npy", allow_pickle=False),
            }
            labels = {
                "representative": np.load(
                    output / "representative_labels.npy",
                    allow_pickle=False,
                ),
                "representative_pool": np.load(
                    output / "representative_pool_labels.npy",
                    allow_pickle=False,
                ),
                "query": np.load(
                    output / "query_labels.npy", allow_pickle=False
                ),
            }
            self.assertEqual(labels["representative"].dtype, np.dtype("<u2"))
            np.testing.assert_array_equal(
                labels["representative"],
                np.asarray((0, 1, 2), dtype=np.dtype("<u2")),
            )
            self.assertTrue(
                set(int(label) for label in np.unique(labels["query"])).issubset(
                    set(int(label) for label in labels["representative"])
                )
            )
            for matrix in matrices.values():
                self.assertEqual(matrix.dtype, np.dtype("<f4"))
                self.assertTrue(matrix.flags.c_contiguous)
            for sample in samples:
                role = sample["role"]
                role_row_index = int(sample["role_row_index"])
                np.testing.assert_array_equal(
                    matrices[role][role_row_index],
                    np.asarray(expected_vectors[sample["sample_id"]]),
                )
                self.assertEqual(
                    labels[role][role_row_index], int(sample["label"])
                )

            bb_samples = [
                sample
                for sample in samples
                if sample["participant_id"] == "TCGA-BB-0001"
            ]
            self.assertEqual(len(bb_samples), 2)
            self.assertEqual(bb_samples[0]["role"], bb_samples[1]["role"])

            representatives = np.load(
                output / "representatives.npy", allow_pickle=False
            )
            for label in range(3):
                pool_vectors = [
                    expected_vectors[sample["sample_id"]]
                    for sample in samples
                    if sample["role"] == "representative_pool"
                    and int(sample["label"]) == label
                ]
                np.testing.assert_array_equal(
                    representatives[label],
                    np.mean(np.asarray(pool_vectors), axis=0),
                )

            with (output / "features.csv").open(
                encoding="utf-8", newline=""
            ) as source:
                features = list(csv.DictReader(source))
            self.assertEqual(
                [feature["ensembl_id"] for feature in features],
                ["ENSG1.1", "ENSG2.2", "ENSG3.3"],
            )

            metadata = json.loads(
                (output / "dataset.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                metadata["matrices"]["representatives"]["shape"], [3, 3]
            )
            self.assertEqual(
                metadata["matrices"]["queries"]["shape"][0],
                len(matrices["query"]),
            )
            self.assertEqual(
                metadata["labels"]["representative_file"],
                "representative_labels.npy",
            )
            self.assertIn(
                "representative_labels.npy",
                metadata["generated_files"],
            )
            self.assertEqual(
                metadata["samples"]["counts_by_project"],
                {"TCGA-KICH": 2, "TCGA-KIRC": 3, "TCGA-KIRP": 2},
            )
            self.assertEqual(
                json.loads(
                    (output / "source_manifest.json").read_text(
                        encoding="utf-8"
                    )
                ),
                source_manifest,
            )

            second_result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(second_result.returncode, 0)
            self.assertIn("Refusing to overwrite", second_result.stderr)


if __name__ == "__main__":
    unittest.main()
