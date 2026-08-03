#!/usr/bin/env python3
"""Integration tests for scripts/transform_coil100.py."""

from __future__ import annotations

import csv
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRANSFORM_SCRIPT = PROJECT_ROOT / "scripts" / "transform_coil100.py"


def write_ppm(
    path: Path,
    width: int,
    height: int,
    values: tuple[int, ...],
) -> None:
    if len(values) != width * height * 3:
        raise ValueError("fixture raster has the wrong size")
    header = (
        b"P6\n# COIL-100 test fixture\n"
        + f"{width} {height}\n".encode("ascii")
        + b"65535 65535 65535\n"
    )
    path.write_bytes(header + struct.pack(f">{len(values)}H", *values))


class TransformCoil100Test(unittest.TestCase):
    def create_fixture(
        self,
        raw: Path,
    ) -> dict[tuple[int, int], np.ndarray]:
        images = raw / "images"
        images.mkdir(parents=True)
        expected: dict[tuple[int, int], np.ndarray] = {}
        for object_id in (1, 2):
            for angle in (0, 90, 180, 270):
                values = tuple(
                    object_id * 1_000 + angle + index
                    for index in range(6)
                )
                write_ppm(
                    images / f"obj{object_id}__{angle}.ppm",
                    2,
                    1,
                    values,
                )
                expected[(object_id, angle)] = (
                    np.asarray(values, dtype=np.float32) / 65_535.0
                )
        (raw / "manifest.json").write_text(
            json.dumps({"fixture": True}),
            encoding="utf-8",
        )
        return expected

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
                "--expected-object-count",
                "2",
                "--angle-step",
                "90",
                "--image-width",
                "2",
                "--image-height",
                "1",
                "--query-start-angle",
                "270",
                "--query-view-count",
                "2",
                *extra_args,
            ),
            check=False,
            capture_output=True,
            text=True,
        )

    def test_writes_training_only_centroids_and_wrapped_contiguous_queries(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw = root / "raw"
            output = root / "processed"
            raw.mkdir()
            expected = self.create_fixture(raw)

            result = self.run_transform(raw, output)
            self.assertEqual(result.returncode, 0, result.stderr)

            representatives = np.load(
                output / "representatives.npy",
                allow_pickle=False,
            )
            queries = np.load(output / "queries.npy", allow_pickle=False)
            labels = np.load(output / "query_labels.npy", allow_pickle=False)
            self.assertEqual(representatives.shape, (2, 6))
            self.assertEqual(queries.shape, (4, 6))
            self.assertEqual(labels.shape, (4,))
            self.assertEqual(representatives.dtype, np.dtype("<f4"))
            self.assertEqual(queries.dtype, np.dtype("<f4"))
            self.assertEqual(labels.dtype, np.dtype("<u2"))
            self.assertTrue(representatives.flags.c_contiguous)
            self.assertTrue(queries.flags.c_contiguous)
            self.assertFalse((output / "representative_pool.npy").exists())

            for label, object_id in enumerate((1, 2)):
                np.testing.assert_allclose(
                    representatives[label],
                    (expected[(object_id, 90)] + expected[(object_id, 180)])
                    / 2.0,
                    rtol=0.0,
                    atol=1e-7,
                )
            query_keys = ((1, 270), (1, 0), (2, 270), (2, 0))
            for row, key in enumerate(query_keys):
                np.testing.assert_array_equal(queries[row], expected[key])
            np.testing.assert_array_equal(labels, np.asarray((0, 0, 1, 1)))

            with (output / "samples.csv").open(
                encoding="utf-8",
                newline="",
            ) as source:
                samples = list(csv.DictReader(source))
            self.assertEqual(len(samples), 8)
            self.assertEqual(
                {
                    int(sample["angle_degrees"])
                    for sample in samples
                    if sample["role"] == "query"
                },
                {0, 270},
            )

            metadata = json.loads(
                (output / "dataset.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                metadata["split"]["query_angles_degrees_in_row_order"],
                [270, 0],
            )
            self.assertEqual(
                metadata["matrices"]["representatives"]["shape"],
                [2, 6],
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

    def test_optionally_materializes_representative_pool(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw = root / "raw"
            output = root / "processed"
            raw.mkdir()
            expected = self.create_fixture(raw)

            result = self.run_transform(
                raw,
                output,
                "--write-representative-pool",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            pool = np.load(
                output / "representative_pool.npy",
                allow_pickle=False,
            )
            labels = np.load(
                output / "representative_pool_labels.npy",
                allow_pickle=False,
            )
            self.assertEqual(pool.shape, (4, 6))
            pool_keys = ((1, 90), (1, 180), (2, 90), (2, 180))
            for row, key in enumerate(pool_keys):
                np.testing.assert_array_equal(pool[row], expected[key])
            np.testing.assert_array_equal(labels, np.asarray((0, 0, 1, 1)))


if __name__ == "__main__":
    unittest.main()
