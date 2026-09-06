#!/usr/bin/env python3
"""Integration tests for scripts/data/transform_tcga_pancancer.py."""

from __future__ import annotations

import csv
import gzip
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOWNLOAD_SCRIPT = PROJECT_ROOT / "scripts" / "data" / "download_tcga_pancancer.py"
TRANSFORM_SCRIPT = PROJECT_ROOT / "scripts" / "data" / "transform_tcga_pancancer.py"
SPEC = importlib.util.spec_from_file_location(
    "download_tcga_pancancer_for_transform_test",
    DOWNLOAD_SCRIPT,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {DOWNLOAD_SCRIPT}")
DOWNLOAD_MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = DOWNLOAD_MODULE
SPEC.loader.exec_module(DOWNLOAD_MODULE)


class TransformTcgaPancancerTest(unittest.TestCase):
    def write_matrix(
        self,
        path: Path,
        sample_ids: tuple[str, ...],
        rows: tuple[tuple[str, tuple[float, ...]], ...],
    ) -> None:
        with gzip.open(
            path,
            mode="wt",
            encoding="utf-8",
            newline="",
        ) as output:
            writer = csv.writer(output, delimiter="\t", lineterminator="\n")
            writer.writerow(("Ensembl_ID", *sample_ids))
            for feature_id, values in rows:
                writer.writerow((feature_id, *values))

    def create_fixture(self, raw: Path) -> dict[str, tuple[float, ...]]:
        expected_vectors: dict[str, tuple[float, ...]] = {}
        for cohort in DOWNLOAD_MODULE.COHORTS:
            retained_code = "03" if cohort.project_id == "TCGA-LAML" else "01"
            excluded_code = "01" if cohort.project_id == "TCGA-LAML" else "11"
            tissue_source_site = f"{cohort.label:02d}"
            sample_ids = (
                f"TCGA-{tissue_source_site}-0001-{retained_code}A",
                f"TCGA-{tissue_source_site}-0002-{retained_code}A",
                f"TCGA-{tissue_source_site}-0003-{excluded_code}A",
            )
            rows: list[tuple[str, tuple[float, ...]]] = []
            for feature_index in range(3):
                base = float(cohort.label * 100 + feature_index)
                rows.append(
                    (
                        f"ENSG{feature_index + 1}.{feature_index + 1}",
                        (base + 10.0, base + 20.0, base + 90.0),
                    )
                )
            self.write_matrix(
                raw / cohort.file_name,
                sample_ids,
                tuple(rows),
            )
            expected_vectors[sample_ids[0]] = tuple(
                float(cohort.label * 100 + index + 10)
                for index in range(3)
            )
            expected_vectors[sample_ids[1]] = tuple(
                float(cohort.label * 100 + index + 20)
                for index in range(3)
            )

        (raw / "manifest.json").write_text(
            json.dumps({"fixture": True}),
            encoding="utf-8",
        )
        return expected_vectors

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
                "--expected-features",
                "3",
                *extra_args,
            ),
            check=False,
            capture_output=True,
            text=True,
        )

    def test_builds_33_centroids_and_project_aware_queries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw = root / "raw"
            output = root / "processed"
            raw.mkdir()
            expected_vectors = self.create_fixture(raw)

            result = self.run_transform(raw, output)
            self.assertEqual(result.returncode, 0, result.stderr)

            reference_vectors = np.load(
                output / "reference_vectors.npy",
                allow_pickle=False,
            )
            queries = np.load(output / "queries.npy", allow_pickle=False)
            reference_labels = np.load(
                output / "reference_labels.npy",
                allow_pickle=False,
            )
            query_labels = np.load(
                output / "query_labels.npy",
                allow_pickle=False,
            )
            self.assertEqual(reference_vectors.shape, (33, 3))
            self.assertEqual(queries.shape, (33, 3))
            self.assertEqual(reference_labels.shape, (33,))
            self.assertEqual(query_labels.shape, (33,))
            self.assertEqual(reference_vectors.dtype, np.dtype("<f4"))
            self.assertEqual(queries.dtype, np.dtype("<f4"))
            self.assertEqual(reference_labels.dtype, np.dtype("<u2"))
            self.assertEqual(query_labels.dtype, np.dtype("<u2"))
            np.testing.assert_array_equal(
                reference_labels,
                np.asarray(
                    [cohort.label for cohort in DOWNLOAD_MODULE.COHORTS],
                    dtype=np.dtype("<u2"),
                ),
            )
            self.assertTrue(
                set(int(label) for label in np.unique(query_labels)).issubset(
                    set(int(label) for label in reference_labels)
                )
            )
            self.assertTrue(reference_vectors.flags.c_contiguous)
            self.assertTrue(queries.flags.c_contiguous)
            self.assertFalse((output / "training_pool.npy").exists())

            with (output / "samples.csv").open(
                encoding="utf-8",
                newline="",
            ) as source:
                samples = list(csv.DictReader(source))
            self.assertEqual(len(samples), 66)
            laml_samples = [
                sample
                for sample in samples
                if sample["project_id"] == "TCGA-LAML"
            ]
            self.assertEqual(
                {sample["sample_type_code"] for sample in laml_samples},
                {"03"},
            )
            self.assertTrue(
                all(
                    sample["sample_type_code"] == "01"
                    for sample in samples
                    if sample["project_id"] != "TCGA-LAML"
                )
            )

            for sample in samples:
                expected = np.asarray(expected_vectors[sample["sample_id"]])
                label = int(sample["label"])
                role_row = int(sample["role_row_index"])
                if sample["role"] == "query":
                    np.testing.assert_array_equal(queries[role_row], expected)
                    self.assertEqual(query_labels[role_row], label)
                else:
                    np.testing.assert_array_equal(reference_vectors[label], expected)

            metadata = json.loads(
                (output / "dataset.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                metadata["matrices"]["reference_vectors"]["shape"],
                [33, 3],
            )
            self.assertEqual(
                metadata["matrices"]["queries"]["shape"],
                [33, 3],
            )
            self.assertEqual(
                metadata["labels"]["reference_file"],
                "reference_labels.npy",
            )
            self.assertIn(
                "reference_labels.npy",
                metadata["generated_files"],
            )
            self.assertEqual(
                metadata["sample_filter"]
                ["retained_sample_type_codes_by_project"]["TCGA-LAML"],
                ["03"],
            )
            self.assertFalse(
                metadata["reference_vector_construction"]
                ["pool_matrix_materialized"]
            )
            self.assertEqual(
                json.loads(
                    (output / "source_manifest.json").read_text(
                        encoding="utf-8"
                    )
                ),
                {"fixture": True},
            )

            second_result = self.run_transform(raw, output)
            self.assertNotEqual(second_result.returncode, 0)
            self.assertIn("Refusing to overwrite", second_result.stderr)

    def test_optionally_materializes_training_pool(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw = root / "raw"
            output = root / "processed"
            raw.mkdir()
            expected_vectors = self.create_fixture(raw)

            result = self.run_transform(
                raw,
                output,
                "--write-training-pool",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            pool = np.load(
                output / "training_pool.npy",
                allow_pickle=False,
            )
            pool_labels = np.load(
                output / "training_pool_labels.npy",
                allow_pickle=False,
            )
            self.assertEqual(pool.shape, (33, 3))
            self.assertEqual(pool_labels.shape, (33,))

            with (output / "samples.csv").open(
                encoding="utf-8",
                newline="",
            ) as source:
                samples = list(csv.DictReader(source))
            for sample in samples:
                if sample["role"] != "training_pool":
                    continue
                row = int(sample["role_row_index"])
                np.testing.assert_array_equal(
                    pool[row],
                    np.asarray(expected_vectors[sample["sample_id"]]),
                )
                self.assertEqual(pool_labels[row], int(sample["label"]))


if __name__ == "__main__":
    unittest.main()
