#!/usr/bin/env python3
"""Tests for scripts/download_tcga_pancancer.py without network access."""

from __future__ import annotations

import csv
import gzip
import hashlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "download_tcga_pancancer.py"
SPEC = importlib.util.spec_from_file_location(
    "download_tcga_pancancer",
    SCRIPT,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {SCRIPT}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class DownloadTcgaPancancerTest(unittest.TestCase):
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

    def test_catalog_contains_33_stably_labeled_projects(self) -> None:
        self.assertEqual(len(MODULE.COHORTS), 33)
        self.assertEqual(
            [cohort.label for cohort in MODULE.COHORTS],
            list(range(33)),
        )
        project_ids = [cohort.project_id for cohort in MODULE.COHORTS]
        self.assertEqual(project_ids, sorted(project_ids))
        self.assertEqual(len(set(project_ids)), 33)
        self.assertIn("TCGA-LAML", project_ids)
        self.assertIn("TCGA-UVM", project_ids)

    def test_download_once_streams_and_hashes_a_local_url(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            payload = b"local download fixture\n" * 100
            source = root / "source.bin"
            destination = root / "destination.bin"
            source.write_bytes(payload)

            record = MODULE.download_once(source.as_uri(), destination)

            self.assertEqual(destination.read_bytes(), payload)
            self.assertEqual(record["bytes"], len(payload))
            self.assertEqual(
                record["sha256"], hashlib.sha256(payload).hexdigest()
            )

    def test_validates_aligned_matrices_and_observes_sample_counts(self) -> None:
        cohorts = (
            MODULE.Cohort(0, "TCGA-AAA", "First Cancer"),
            MODULE.Cohort(1, "TCGA-BBB", "Second Cancer"),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            paths = [root / cohort.file_name for cohort in cohorts]
            self.write_matrix(
                paths[0],
                ("TCGA-AA-0001-01A", "TCGA-AA-0002-01A"),
                (
                    ("ENSG1.1", (1.0, 2.0)),
                    ("ENSG2.2", (3.0, 4.0)),
                    ("ENSG3.3", (5.0, 6.0)),
                ),
            )
            self.write_matrix(
                paths[1],
                ("TCGA-BB-0001-01A",),
                (
                    ("ENSG1.1", (7.0,)),
                    ("ENSG2.2", (8.0,)),
                    ("ENSG3.3", (9.0,)),
                ),
            )

            samples, sample_counts, feature_count = MODULE.validate_matrices(
                cohorts,
                paths,
                expected_features=3,
            )

            self.assertEqual(feature_count, 3)
            self.assertEqual(
                sample_counts,
                {"TCGA-AAA": 2, "TCGA-BBB": 1},
            )
            self.assertEqual([sample["label"] for sample in samples], [0, 0, 1])

    def test_rejects_different_feature_order(self) -> None:
        cohorts = (
            MODULE.Cohort(0, "TCGA-AAA", "First Cancer"),
            MODULE.Cohort(1, "TCGA-BBB", "Second Cancer"),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            paths = [root / cohort.file_name for cohort in cohorts]
            self.write_matrix(
                paths[0],
                ("TCGA-AA-0001-01A",),
                (("ENSG1.1", (1.0,)), ("ENSG2.2", (2.0,))),
            )
            self.write_matrix(
                paths[1],
                ("TCGA-BB-0001-01A",),
                (("ENSG2.2", (3.0,)), ("ENSG1.1", (4.0,))),
            )

            with self.assertRaisesRegex(RuntimeError, "matching Ensembl"):
                MODULE.validate_matrices(
                    cohorts,
                    paths,
                    expected_features=2,
                )


if __name__ == "__main__":
    unittest.main()
