#!/usr/bin/env python3
"""Download and validate the pinned Bioconductor Golub leukemia package."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "golub"
BIOCONDUCTOR_RELEASE = "3.23"
PACKAGE = "golubEsets"
PACKAGE_VERSION = "1.54.0"
PACKAGE_NAME = f"{PACKAGE}_{PACKAGE_VERSION}.tar.gz"
PACKAGE_URL = (
    f"https://bioconductor.org/packages/{BIOCONDUCTOR_RELEASE}/data/"
    f"experiment/src/contrib/{PACKAGE_NAME}"
)
PACKAGE_PAGE_URL = (
    "https://bioconductor.org/packages/release/data/experiment/html/"
    "golubEsets.html"
)
EXPECTED_PACKAGE_SHA256 = (
    "47c6bbf49322f32c80e63aaf50fab9db16a0a9c42951dc433dc8e84c3bf01f22"
)
EXPECTED_MEMBERS = {
    "description": "golubEsets/DESCRIPTION",
    "train": "golubEsets/data/Golub_Train.rda",
    "test": "golubEsets/data/Golub_Test.rda",
}
EXPECTED_FEATURES = 7_129
EXPECTED_TRAIN_SAMPLES = 38
EXPECTED_TEST_SAMPLES = 34
EXPECTED_TRAIN_COUNTS = {"ALL": 27, "AML": 11}
EXPECTED_TEST_COUNTS = {"ALL": 20, "AML": 14}
DOWNLOAD_BLOCK_SIZE = 1024 * 1024
PROGRESS_INTERVAL = 4 * 1024 * 1024
MANAGED_OUTPUT_NAMES = {PACKAGE_NAME, "manifest.json"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the pinned official Bioconductor golubEsets source "
            "package, validate its archive contents and checksum, and record "
            "provenance for the ALL-versus-AML benchmark."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"destination directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--package-url",
        default=PACKAGE_URL,
        help=f"source-package URL (default: {PACKAGE_URL})",
    )
    parser.add_argument(
        "--expected-sha256",
        default=EXPECTED_PACKAGE_SHA256,
        help="expected source-package SHA-256 (use 'none' to skip pinning)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="download attempts (default: 3)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=90.0,
        help="per-operation network timeout in seconds (default: 90)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace a previously validated Golub download",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(DOWNLOAD_BLOCK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def stream_download(
    source: BinaryIO,
    destination: BinaryIO,
    expected_bytes: int | None,
) -> int:
    downloaded = 0
    next_report = PROGRESS_INTERVAL
    while block := source.read(DOWNLOAD_BLOCK_SIZE):
        destination.write(block)
        downloaded += len(block)
        if downloaded >= next_report:
            message = f"  {downloaded / (1024 * 1024):.0f} MiB"
            if expected_bytes:
                message += f" ({100.0 * downloaded / expected_bytes:.0f}%)"
            print(message, flush=True)
            next_report += PROGRESS_INTERVAL
    return downloaded


def download_once(
    url: str,
    destination: Path,
    timeout: float = 90.0,
) -> dict[str, object]:
    request = Request(url, headers={"User-Agent": "ultrahigh-ann/0.1"})
    with urlopen(request, timeout=timeout) as response:
        length_header = response.headers.get("Content-Length")
        expected_bytes = int(length_header) if length_header else None
        with destination.open("wb") as output:
            downloaded_bytes = stream_download(response, output, expected_bytes)
        if expected_bytes is not None and downloaded_bytes != expected_bytes:
            raise RuntimeError(
                f"incomplete download: received {downloaded_bytes} of "
                f"{expected_bytes} bytes"
            )
        return {
            "requested_url": url,
            "resolved_url": response.geturl(),
            "bytes": downloaded_bytes,
            "etag": response.headers.get("ETag"),
            "last_modified": response.headers.get("Last-Modified"),
            "sha256": sha256(destination),
        }


def download(
    url: str,
    destination: Path,
    attempts: int,
    timeout: float,
) -> dict[str, object]:
    for attempt in range(1, attempts + 1):
        try:
            return download_once(url, destination, timeout)
        except (
            HTTPError,
            URLError,
            TimeoutError,
            OSError,
            RuntimeError,
        ) as error:
            if destination.exists():
                destination.unlink()
            if attempt == attempts:
                raise RuntimeError(
                    f"download failed after {attempts} attempts: {error}"
                ) from error
            delay = min(2 ** (attempt - 1), 8)
            print(
                f"  attempt {attempt} failed: {error}; retrying in "
                f"{delay} seconds...",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
    raise RuntimeError("unreachable download retry state")


def parse_description(payload: str) -> dict[str, str]:
    """Parse the simple Debian-control-style DESCRIPTION metadata."""
    fields: dict[str, str] = {}
    current: str | None = None
    for line in payload.splitlines():
        if line[:1].isspace():
            if current is None:
                raise RuntimeError("DESCRIPTION begins with a continuation line")
            fields[current] += " " + line.strip()
            continue
        if ":" not in line:
            if line.strip():
                raise RuntimeError(f"invalid DESCRIPTION line: {line!r}")
            current = None
            continue
        current, value = line.split(":", 1)
        current = current.strip()
        if not current or current in fields:
            raise RuntimeError(f"invalid or duplicate DESCRIPTION field: {current!r}")
        fields[current] = value.strip()
    return fields


def member_record(archive: tarfile.TarFile, member: tarfile.TarInfo) -> dict[str, object]:
    source = archive.extractfile(member)
    if source is None:
        raise RuntimeError(f"cannot read archive member {member.name}")
    digest = hashlib.sha256()
    size = 0
    with source:
        while block := source.read(DOWNLOAD_BLOCK_SIZE):
            digest.update(block)
            size += len(block)
    if size != member.size:
        raise RuntimeError(f"archive member {member.name} is truncated")
    return {"archive_path": member.name, "bytes": size, "sha256": digest.hexdigest()}


def validate_archive(
    path: Path,
    expected_sha256: str | None = EXPECTED_PACKAGE_SHA256,
) -> dict[str, object]:
    actual_sha256 = sha256(path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"source-package SHA-256 is {actual_sha256}; expected {expected_sha256}"
        )
    try:
        with tarfile.open(path, mode="r:gz") as archive:
            members = {member.name: member for member in archive.getmembers()}
            required: dict[str, tarfile.TarInfo] = {}
            for role, name in EXPECTED_MEMBERS.items():
                member = members.get(name)
                if member is None or not member.isfile():
                    raise RuntimeError(f"source package is missing regular file {name}")
                if member.size <= 0:
                    raise RuntimeError(f"source package contains empty file {name}")
                required[role] = member

            description_source = archive.extractfile(required["description"])
            if description_source is None:
                raise RuntimeError("cannot read source-package DESCRIPTION")
            try:
                description_payload = description_source.read().decode("utf-8")
            except UnicodeDecodeError as error:
                raise RuntimeError("source-package DESCRIPTION is not UTF-8") from error
            finally:
                description_source.close()
            description = parse_description(description_payload)
            expected_fields = {
                "Package": PACKAGE,
                "Version": PACKAGE_VERSION,
                "Repository": f"Bioconductor {BIOCONDUCTOR_RELEASE}",
            }
            for field, expected in expected_fields.items():
                if description.get(field) != expected:
                    raise RuntimeError(
                        f"DESCRIPTION {field} is {description.get(field)!r}; "
                        f"expected {expected!r}"
                    )

            member_records = {
                role: member_record(archive, member)
                for role, member in required.items()
                if role != "description"
            }
    except (tarfile.TarError, OSError) as error:
        raise RuntimeError(f"cannot parse {path.name}: {error}") from error

    return {
        "archive_sha256": actual_sha256,
        "package": PACKAGE,
        "package_version": PACKAGE_VERSION,
        "bioconductor_release": BIOCONDUCTOR_RELEASE,
        "license": description.get("License"),
        "git_url": description.get("git_url"),
        "git_commit": description.get("git_last_commit"),
        "git_commit_date": description.get("git_last_commit_date"),
        "archive_members": member_records,
    }


def install_directory(staged: Path, destination: Path) -> None:
    if destination.is_symlink():
        raise RuntimeError(f"refusing to replace symbolic-link output: {destination}")
    if not destination.exists():
        staged.replace(destination)
        return
    backup = staged.parent / "previous-download"
    destination.replace(backup)
    try:
        staged.replace(destination)
    except OSError:
        backup.replace(destination)
        raise
    shutil.rmtree(backup)


def validate_args(args: argparse.Namespace) -> str | None:
    if args.retries < 1:
        raise ValueError("--retries must be at least one")
    if args.timeout <= 0.0:
        raise ValueError("--timeout must be positive")
    expected = args.expected_sha256.strip().lower()
    if expected == "none":
        return None
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise ValueError("--expected-sha256 must be 64 hexadecimal characters or 'none'")
    return expected


def main() -> int:
    args = parse_args()
    try:
        expected_sha256 = validate_args(args)
        requested_output_dir = args.output_dir.expanduser()
        if requested_output_dir.is_symlink():
            raise RuntimeError(
                f"refusing to replace symbolic-link output: {requested_output_dir}"
            )
        output_dir = requested_output_dir.resolve()
        if output_dir.exists() and not output_dir.is_dir():
            raise RuntimeError(f"output path is not a directory: {output_dir}")
        existing: list[str] = []
        if output_dir.is_dir():
            existing = sorted(
                path.name for path in output_dir.iterdir() if path.name != ".gitkeep"
            )
        unexpected = sorted(set(existing) - MANAGED_OUTPUT_NAMES)
        if unexpected:
            raise RuntimeError(
                f"refusing to replace unrelated entries in {output_dir}: "
                f"{', '.join(unexpected)}"
            )
        if existing and not args.force:
            print(
                f"Refusing to overwrite existing files in {output_dir}: "
                f"{', '.join(existing)}\nRun again with --force to replace them.",
                file=sys.stderr,
            )
            return 1

        output_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".golub-download-",
            dir=output_dir.parent,
        ) as temporary_directory:
            staging_dir = Path(temporary_directory) / "golub"
            staging_dir.mkdir()
            package_path = staging_dir / PACKAGE_NAME

            print(
                f"Downloading official Bioconductor {PACKAGE} {PACKAGE_VERSION}...",
                flush=True,
            )
            download_record = download(
                args.package_url,
                package_path,
                args.retries,
                args.timeout,
            )
            print("Validating package identity, checksum, and data members...", flush=True)
            validation = validate_archive(package_path, expected_sha256)

            manifest = {
                "dataset_name": "Golub ALL-versus-AML leukemia expression data",
                "format_version": 1,
                "source": "Bioconductor experiment-data package",
                "source_page_url": PACKAGE_PAGE_URL,
                "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_package": {
                    "file": PACKAGE_NAME,
                    **download_record,
                    "validation": validation,
                },
                "expected_dataset": {
                    "features": EXPECTED_FEATURES,
                    "train_samples": EXPECTED_TRAIN_SAMPLES,
                    "test_samples": EXPECTED_TEST_SAMPLES,
                    "train_class_counts": EXPECTED_TRAIN_COUNTS,
                    "test_class_counts": EXPECTED_TEST_COUNTS,
                    "platform": "Affymetrix Hgu6800",
                },
                "provenance_note": (
                    "The package documentation says that the original Golub data "
                    "were 'transformed slightly' and that some covariate provenance "
                    "is unknown. The benchmark preserves the packaged expression "
                    "values and published train/test split."
                ),
            }
            with (staging_dir / "manifest.json").open(
                "w", encoding="utf-8"
            ) as output:
                json.dump(manifest, output, indent=2, sort_keys=True)
                output.write("\n")

            install_directory(staging_dir, output_dir)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Saved the validated source package to {output_dir}.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
