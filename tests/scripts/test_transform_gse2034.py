#!/usr/bin/env python3
"""Integration tests for scripts/data/transform_gse2034.py."""

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
TRANSFORM_SCRIPT = PROJECT_ROOT / "scripts" / "data" / "transform_gse2034.py"
CLINICAL_COLUMNS = (
    "PID",
    "GEO asscession number",
    "lymph node status",
    "time to relapse or last follow-up (months)",
    "relapse (1=True)",
    "ER Status",
    "Brain relapses (1=yes, 0=no)",
)


class TransformGse2034Test(unittest.TestCase):
    def create_fixture(self, raw: Path) -> np.ndarray:
        accessions = tuple(f"GSM{index + 1}" for index in range(6))
        values = np.asarray(
            (
                (1, 2, 4, 8),
                (2, 3, 6, 7),
                (3, 5, 5, 9),
                (8, 7, 3, 2),
                (9, 6, 2, 3),
                (7, 8, 1, 4),
            ),
            dtype=np.float32,
        )
        raw.mkdir(parents=True)
        with gzip.open(
            raw / "GSE2034_series_matrix.txt.gz",
            "wt",
            encoding="utf-8",
            newline="",
        ) as output:
            writer = csv.writer(output, delimiter="\t", quoting=csv.QUOTE_ALL)
            writer.writerow(("!Series_geo_accession", "GSE2034"))
            writer.writerow(("!Sample_geo_accession", *accessions))
            writer.writerow(("!series_matrix_table_begin",))
            writer.writerow(("ID_REF", *accessions))
            for feature, feature_values in enumerate(values.T):
                writer.writerow((f"probe_{feature}", *feature_values))
            writer.writerow(("!series_matrix_table_end",))

        with (raw / "GSE2034_clinical.tsv").open(
            "w",
            encoding="utf-8",
            newline="",
        ) as output:
            output.write("# clinical test fixture\n")
            writer = csv.writer(output, delimiter="\t")
            writer.writerow(CLINICAL_COLUMNS)
            for index, accession in enumerate(accessions):
                writer.writerow(
                    (
                        index + 1,
                        accession,
                        "negative",
                        10 + index,
                        int(index >= 3),
                        "ER+" if index % 2 else "ER-",
                        0,
                    )
                )
        (raw / "manifest.json").write_text(
            json.dumps({"fixture": True}),
            encoding="utf-8",
        )
        return values

    def run_transform(
        self,
        raw: Path,
        output: Path,
        *extra_args: str,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            (
                sys.executable,
                str(TRANSFORM_SCRIPT),
                "--input-dir",
                str(raw),
                "--output-dir",
                str(output),
                "--expected-samples",
                "6",
                "--expected-features",
                "4",
                "--expected-relapse-free",
                "3",
                "--expected-relapse",
                "3",
                "--representative-fraction",
                "0.67",
                "--seed",
                "42",
                *extra_args,
            ),
            check=False,
            capture_output=True,
            text=True,
        )

    def test_writes_training_only_normalization_centroids_and_queries(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw = root / "raw"
            output = root / "processed"
            source_values = self.create_fixture(raw)

            result = self.run_transform(raw, output)
            self.assertEqual(result.returncode, 0, result.stderr)

            representatives = np.load(
                output / "representatives.npy",
                allow_pickle=False,
            )
            representative_labels = np.load(
                output / "representative_labels.npy",
                allow_pickle=False,
            )
            pool = np.load(output / "representative_pool.npy", allow_pickle=False)
            pool_labels = np.load(
                output / "representative_pool_labels.npy",
                allow_pickle=False,
            )
            queries = np.load(output / "queries.npy", allow_pickle=False)
            query_labels = np.load(
                output / "query_labels.npy",
                allow_pickle=False,
            )
            self.assertEqual(representatives.shape, (2, 4))
            self.assertEqual(pool.shape, (4, 4))
            self.assertEqual(queries.shape, (2, 4))
            self.assertEqual(representatives.dtype, np.dtype("<f4"))
            self.assertEqual(query_labels.dtype, np.dtype("<u2"))
            np.testing.assert_array_equal(representative_labels, (0, 1))
            self.assertEqual(sorted(query_labels.tolist()), [0, 1])
            for label in (0, 1):
                np.testing.assert_allclose(
                    representatives[label],
                    pool[pool_labels == label].mean(axis=0),
                    rtol=0.0,
                    atol=1e-6,
                )

            with (output / "samples.csv").open(
                encoding="utf-8",
                newline="",
            ) as source:
                sample_rows = list(csv.DictReader(source))
            pool_source_columns = [
                int(row["source_column"])
                for row in sample_rows
                if row["role"] == "representative_pool"
            ]
            logged = np.log2(source_values + 1.0)
            expected_mean = logged[pool_source_columns].mean(
                axis=0,
                dtype=np.float64,
            )
            expected_scale = logged[pool_source_columns].std(
                axis=0,
                dtype=np.float64,
            )
            np.testing.assert_allclose(
                np.load(output / "normalization_mean.npy", allow_pickle=False),
                expected_mean,
                rtol=0.0,
                atol=1e-6,
            )
            np.testing.assert_allclose(
                np.load(output / "normalization_scale.npy", allow_pickle=False),
                expected_scale,
                rtol=0.0,
                atol=1e-6,
            )

            metadata = json.loads(
                (output / "dataset.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["split"]["representative_pool_count"], 4)
            self.assertEqual(metadata["split"]["query_count"], 2)
            self.assertEqual(
                metadata["normalization"]["fit_rows"],
                "representative_pool only",
            )
            self.assertEqual(
                json.loads(
                    (output / "source_manifest.json").read_text(encoding="utf-8")
                ),
                {"fixture": True},
            )

            second_result = self.run_transform(raw, output)
            self.assertNotEqual(second_result.returncode, 0)
            self.assertIn("Refusing to overwrite", second_result.stderr)


if __name__ == "__main__":
    unittest.main()
