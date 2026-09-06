#!/usr/bin/env python3
"""Tests for scripts/data/download_golub.py without network access."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "data" / "download_golub.py"
SPEC = importlib.util.spec_from_file_location("download_golub", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {SCRIPT}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def add_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(payload)
    archive.addfile(member, io.BytesIO(payload))


def write_package(path: Path) -> None:
    description = (
        "Package: golubEsets\n"
        "Version: 1.54.0\n"
        "Description: first line\n"
        "  continued description\n"
        "Repository: Bioconductor 3.23\n"
        "License: LGPL\n"
        "git_url: https://git.bioconductor.org/packages/golubEsets\n"
        "git_last_commit: fixture\n"
        "git_last_commit_date: 2026-01-01\n"
    ).encode()
    with tarfile.open(path, mode="w:gz") as archive:
        add_bytes(archive, "golubEsets/DESCRIPTION", description)
        add_bytes(archive, "golubEsets/data/Golub_Train.rda", b"train fixture")
        add_bytes(archive, "golubEsets/data/Golub_Test.rda", b"test fixture")


class DownloadGolubTest(unittest.TestCase):
    def test_download_once_streams_and_hashes_local_package(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.tar.gz"
            destination = root / "destination.tar.gz"
            write_package(source)

            record = MODULE.download_once(source.as_uri(), destination)

            self.assertEqual(destination.read_bytes(), source.read_bytes())
            self.assertEqual(record["bytes"], source.stat().st_size)
            self.assertEqual(
                record["sha256"], hashlib.sha256(source.read_bytes()).hexdigest()
            )

    def test_validates_package_identity_and_required_members(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            package = Path(temporary_directory) / "golub.tar.gz"
            write_package(package)

            validation = MODULE.validate_archive(package, expected_sha256=None)

            self.assertEqual(validation["package"], "golubEsets")
            self.assertEqual(validation["package_version"], "1.54.0")
            self.assertEqual(validation["bioconductor_release"], "3.23")
            self.assertEqual(
                validation["archive_members"]["train"]["bytes"],
                len(b"train fixture"),
            )

    def test_rejects_a_checksum_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            package = Path(temporary_directory) / "golub.tar.gz"
            write_package(package)
            with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                MODULE.validate_archive(package, expected_sha256="0" * 64)


if __name__ == "__main__":
    unittest.main()
