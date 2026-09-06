#!/usr/bin/env python3
"""Tests for scripts/data/download_gse2034.py without network access."""

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
SCRIPT = PROJECT_ROOT / "scripts" / "data" / "download_gse2034.py"
SPEC = importlib.util.spec_from_file_location("download_gse2034", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {SCRIPT}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_matrix(path: Path) -> tuple[str, ...]:
    accessions = ("GSM1", "GSM2", "GSM3", "GSM4")
    with gzip.open(path, "wt", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, delimiter="\t", quoting=csv.QUOTE_ALL)
        writer.writerow(("!Series_geo_accession", "GSE2034"))
        writer.writerow(("!Series_platform_id", "GPL96"))
        writer.writerow(("!Series_pubmed_id", "15721472"))
        writer.writerow(("!Sample_geo_accession", *accessions))
        writer.writerow(("!series_matrix_table_begin",))
        writer.writerow(("ID_REF", *accessions))
        writer.writerow(("probe_a", 1, 2, 3, 4))
        writer.writerow(("probe_b", 5, 6, 7, 8))
        writer.writerow(("!series_matrix_table_end",))
    return accessions


def write_clinical(path: Path, accessions: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        output.write("#PID = \n")
        writer = csv.writer(output, delimiter="\t")
        writer.writerow(MODULE.CLINICAL_COLUMNS)
        for index, accession in enumerate(accessions):
            label = int(index >= 2)
            writer.writerow(
                (
                    index + 1,
                    accession,
                    "negative",
                    12 + index,
                    label,
                    "ER+" if index % 2 else "ER-",
                    0,
                )
            )


class DownloadGse2034Test(unittest.TestCase):
    def test_download_once_streams_and_hashes_a_local_url(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.txt"
            destination = root / "destination.txt"
            payload = b"GSE2034 local download fixture\n" * 20
            source.write_bytes(payload)

            record = MODULE.download_once(source.as_uri(), destination)

            self.assertEqual(destination.read_bytes(), payload)
            self.assertEqual(record["bytes"], len(payload))
            self.assertEqual(
                record["sha256"],
                hashlib.sha256(payload).hexdigest(),
            )

    def test_discovers_raw_clinical_table_url(self) -> None:
        page = (
            "window.open('/geo/query/acc.cgi?view=data&amp;acc=GSE2034"
            "&amp;id=40089&amp;db=GeoDb_blob26', '_blank')"
        )
        url = MODULE.discover_clinical_url(
            page,
            "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE2034",
        )
        self.assertEqual(
            url,
            "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?"
            "mode=raw&is_datatable=true&acc=GSE2034&id=40089&db=GeoDb_blob26",
        )

    def test_validates_matrix_and_clinical_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            matrix = root / "matrix.txt.gz"
            clinical = root / "clinical.tsv"
            accessions = write_matrix(matrix)
            write_clinical(clinical, accessions)

            matrix_record = MODULE.validate_series_matrix(
                matrix,
                expected_samples=4,
                expected_features=2,
            )
            clinical_record = MODULE.validate_clinical_table(
                clinical,
                matrix_record["sample_accessions"],
                expected_counts={0: 2, 1: 2},
            )

            self.assertEqual(matrix_record["samples"], 4)
            self.assertEqual(matrix_record["features"], 2)
            self.assertEqual(matrix_record["minimum_value"], 1.0)
            self.assertEqual(matrix_record["maximum_value"], 8.0)
            self.assertEqual(
                clinical_record["relapse_counts"],
                {"relapse_free_0": 2, "distant_metastasis_1": 2},
            )

    def test_rejects_clinical_table_with_a_missing_matrix_sample(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            clinical = Path(temporary_directory) / "clinical.tsv"
            write_clinical(clinical, ("GSM1", "GSM2", "GSM3", "GSM4"))
            with self.assertRaisesRegex(RuntimeError, "samples differ"):
                MODULE.validate_clinical_table(
                    clinical,
                    ("GSM1", "GSM2", "GSM3", "GSM999"),
                    expected_counts={0: 2, 1: 2},
                )


if __name__ == "__main__":
    unittest.main()
