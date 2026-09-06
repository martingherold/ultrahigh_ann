#!/usr/bin/env python3
"""Integration tests for scripts/data/transform_golub.py."""

from __future__ import annotations

import csv
import io
import json
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRANSFORM_SCRIPT = PROJECT_ROOT / "scripts" / "data" / "transform_golub.py"
PACKAGE_NAME = "golubEsets_1.54.0.tar.gz"


def add_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(payload)
    archive.addfile(member, io.BytesIO(payload))


class TransformGolubTest(unittest.TestCase):
    def create_fixture(self, root: Path) -> tuple[Path, Path, np.ndarray]:
        raw = root / "raw"
        raw.mkdir()
        with tarfile.open(raw / PACKAGE_NAME, mode="w:gz") as archive:
            add_bytes(
                archive,
                "golubEsets/data/Golub_Train.rda",
                b"placeholder train R data",
            )
            add_bytes(
                archive,
                "golubEsets/data/Golub_Test.rda",
                b"placeholder test R data",
            )
        (raw / "manifest.json").write_text(
            json.dumps({"fixture": True}), encoding="utf-8"
        )

        training = np.asarray(
            (
                (-4, 0, 2, 8),
                (-2, 2, 4, 10),
                (0, 4, 6, 12),
                (8, 8, -2, 0),
                (10, 10, 0, 2),
                (12, 12, 2, 4),
            ),
            dtype=np.float32,
        )
        test = np.asarray(
            (
                (-3, 1, 3, 9),
                (-1, 3, 5, 11),
                (9, 9, -1, 1),
                (11, 11, 1, 3),
            ),
            dtype=np.float32,
        )
        exporter = root / "fake_rscript.py"
        exporter.write_text(
            textwrap.dedent(
                f"""\
                #!{sys.executable}
                import csv
                import pathlib
                import sys

                output = pathlib.Path(sys.argv[-1])
                output.mkdir(parents=True)
                feature_ids = ["probe_a", "probe_b", "probe_c", "probe_d"]
                splits = {{
                    "train": (
                        {training.tolist()!r},
                        ["1", "2", "3", "4", "5", "6"],
                        ["ALL", "ALL", "ALL", "AML", "AML", "AML"],
                    ),
                    "test": (
                        {test.tolist()!r},
                        ["7", "8", "9", "10"],
                        ["ALL", "ALL", "AML", "AML"],
                    ),
                }}
                phenotype_columns = [
                    "Samples", "ALL.AML", "BM.PB", "T.B.cell", "FAB",
                    "Date", "Gender", "pctBlasts", "Treatment", "PS", "Source"
                ]
                for split, (rows, sample_ids, labels) in splits.items():
                    with (output / f"{{split}}_expression.tsv").open(
                        "w", newline="", encoding="utf-8"
                    ) as destination:
                        writer = csv.writer(destination, delimiter="\\t")
                        writer.writerow(["", *sample_ids])
                        for feature_index, feature_id in enumerate(feature_ids):
                            writer.writerow(
                                [feature_id, *[row[feature_index] for row in rows]]
                            )
                    with (output / f"{{split}}_samples.tsv").open(
                        "w", newline="", encoding="utf-8"
                    ) as destination:
                        writer = csv.writer(destination, delimiter="\\t")
                        writer.writerow(phenotype_columns)
                        for sample_id, label in zip(sample_ids, labels):
                            writer.writerow(
                                [sample_id, label, "BM", "", "", "", "", "", "", "", "fixture"]
                            )
                """
            ),
            encoding="utf-8",
        )
        exporter.chmod(0o755)
        return raw, exporter, training

    def run_transform(
        self,
        raw: Path,
        output: Path,
        exporter: Path,
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
                "--rscript",
                str(exporter),
                "--expected-features",
                "4",
                "--expected-train-samples",
                "6",
                "--expected-test-samples",
                "4",
                "--expected-train-all",
                "3",
                "--expected-train-aml",
                "3",
                "--expected-test-all",
                "2",
                "--expected-test-aml",
                "2",
                *extra_args,
            ),
            check=False,
            capture_output=True,
            text=True,
        )

    def test_preserves_published_split_and_uses_training_only_zscores(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw, exporter, source_training = self.create_fixture(root)
            output = root / "processed"

            result = self.run_transform(raw, output, exporter)
            self.assertEqual(result.returncode, 0, result.stderr)

            representatives = np.load(
                output / "representatives.npy", allow_pickle=False
            )
            representative_labels = np.load(
                output / "representative_labels.npy", allow_pickle=False
            )
            pool = np.load(output / "representative_pool.npy", allow_pickle=False)
            pool_labels = np.load(
                output / "representative_pool_labels.npy", allow_pickle=False
            )
            queries = np.load(output / "queries.npy", allow_pickle=False)
            query_labels = np.load(output / "query_labels.npy", allow_pickle=False)
            self.assertEqual(representatives.shape, (2, 4))
            self.assertEqual(pool.shape, (6, 4))
            self.assertEqual(queries.shape, (4, 4))
            self.assertEqual(representatives.dtype, np.dtype("<f4"))
            np.testing.assert_array_equal(representative_labels, (0, 1))
            np.testing.assert_array_equal(pool_labels, (0, 0, 0, 1, 1, 1))
            np.testing.assert_array_equal(query_labels, (0, 0, 1, 1))
            for label in (0, 1):
                np.testing.assert_allclose(
                    representatives[label],
                    pool[pool_labels == label].mean(axis=0),
                    rtol=0.0,
                    atol=1e-6,
                )

            expected_mean = source_training.mean(axis=0, dtype=np.float64)
            expected_scale = source_training.std(axis=0, dtype=np.float64)
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

            with (output / "samples.csv").open(
                encoding="utf-8", newline=""
            ) as source:
                sample_rows = list(csv.DictReader(source))
            self.assertEqual(
                [row["source_split"] for row in sample_rows],
                ["published_train"] * 6 + ["published_test"] * 4,
            )

            metadata = json.loads(
                (output / "dataset.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                metadata["split"]["policy"],
                "original Golub published train/test split",
            )
            self.assertEqual(
                metadata["normalization"]["fit_rows"],
                "published training split only",
            )
            self.assertEqual(
                json.loads(
                    (output / "source_manifest.json").read_text(encoding="utf-8")
                ),
                {"fixture": True},
            )

            second = self.run_transform(raw, output, exporter)
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("Refusing to overwrite", second.stderr)


if __name__ == "__main__":
    unittest.main()
