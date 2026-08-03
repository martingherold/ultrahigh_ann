#!/usr/bin/env python3
"""Tests for scripts/download_coil100.py without network access."""

from __future__ import annotations

import hashlib
import importlib.util
import struct
import sys
import tarfile
import tempfile
import unittest
from io import BytesIO
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "download_coil100.py"
SPEC = importlib.util.spec_from_file_location("download_coil100", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {SCRIPT}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def ppm_bytes(
    width: int,
    height: int,
    values: tuple[int, ...],
) -> bytes:
    if len(values) != width * height * 3:
        raise ValueError("fixture raster has the wrong size")
    header = (
        b"P6\n# COIL-100 test fixture\n"
        + f"{width} {height}\n".encode("ascii")
        + b"65535 65535 65535\n"
    )
    return header + struct.pack(f">{len(values)}H", *values)


class DownloadCoil100Test(unittest.TestCase):
    def test_catalog_covers_100_objects_and_72_views(self) -> None:
        self.assertEqual(MODULE.EXPECTED_OBJECT_IDS, tuple(range(1, 101)))
        self.assertEqual(MODULE.EXPECTED_ANGLES, tuple(range(0, 360, 5)))
        self.assertEqual(
            len(MODULE.expected_image_keys()),
            7_200,
        )
        self.assertEqual(MODULE.image_key_from_name("obj100__355.ppm"), (100, 355))

    def test_download_once_streams_and_hashes_a_local_url(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            payload = b"local COIL-100 download fixture\n" * 100
            source = root / "source.tar.gz"
            destination = root / "destination.tar.gz"
            source.write_bytes(payload)

            record = MODULE.download_once(source.as_uri(), destination)

            self.assertEqual(destination.read_bytes(), payload)
            self.assertEqual(record["bytes"], len(payload))
            self.assertEqual(
                record["sha256"],
                hashlib.sha256(payload).hexdigest(),
            )

    def test_extracts_normalized_inventory_and_validates_ppm_headers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            archive = root / "coil.tar.gz"
            images = root / "images"
            with tarfile.open(archive, "w:gz") as output:
                for object_id in (1, 2):
                    for angle in (0, 180):
                        values = tuple(
                            object_id * 100 + angle + index
                            for index in range(6)
                        )
                        payload = ppm_bytes(2, 1, values)
                        member = tarfile.TarInfo(
                            f"coil-100/obj{object_id}__{angle}.ppm"
                        )
                        member.size = len(payload)
                        output.addfile(member, BytesIO(payload))

            record = MODULE.extract_and_validate_archive(
                archive,
                images,
                object_ids=(1, 2),
                angles=(0, 180),
                expected_width=2,
                expected_height=1,
            )

            self.assertEqual(record["count"], 4)
            self.assertEqual(record["object_count"], 2)
            self.assertEqual(record["views_per_object"], 2)
            inventory = MODULE.collect_image_paths(
                images,
                object_ids=(1, 2),
                angles=(0, 180),
            )
            self.assertEqual(set(inventory), {(1, 0), (1, 180), (2, 0), (2, 180)})
            header = MODULE.read_ppm_header(inventory[(2, 180)])
            self.assertEqual((header.width, header.height), (2, 1))
            self.assertEqual(header.channel_maxima, (65_535, 65_535, 65_535))
            self.assertEqual(header.bytes_per_sample, 2)

    def test_rejects_a_traversal_member_before_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            archive = root / "unsafe.tar.gz"
            with tarfile.open(archive, "w:gz") as output:
                payload = ppm_bytes(1, 1, (1, 2, 3))
                member = tarfile.TarInfo("../obj1__0.ppm")
                member.size = len(payload)
                output.addfile(member, BytesIO(payload))

            with self.assertRaisesRegex(RuntimeError, "unsafe tar member"):
                MODULE.extract_and_validate_archive(
                    archive,
                    root / "images",
                    object_ids=(1,),
                    angles=(0,),
                    expected_width=1,
                    expected_height=1,
                )
            self.assertEqual(list((root / "images").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
