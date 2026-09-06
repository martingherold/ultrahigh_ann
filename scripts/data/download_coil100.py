#!/usr/bin/env python3
"""Download and validate the processed Columbia COIL-100 image archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import BinaryIO
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "coil100"
SOURCE_PAGE_URL = "https://cave.cs.columbia.edu/repository/COIL-100"
ARCHIVE_URL = (
    "https://www.cs.columbia.edu/CAVE/databases/"
    "SLAM_coil-20_coil-100/coil-100/coil-100.tar.gz"
)
ARCHIVE_NAME = "coil-100.tar.gz"
IMAGE_PATTERN = re.compile(r"obj([1-9][0-9]{0,2})__([0-9]{1,3})\.ppm")
EXPECTED_OBJECT_IDS = tuple(range(1, 101))
EXPECTED_ANGLES = tuple(range(0, 360, 5))
EXPECTED_WIDTH = 128
EXPECTED_HEIGHT = 128
EXPECTED_CHANNEL_MAXIMA = (65_535, 65_535, 65_535)
DOWNLOAD_BLOCK_SIZE = 1024 * 1024
PROGRESS_INTERVAL = 16 * 1024 * 1024
MAX_IMAGE_BYTES = 256 * 1024
MAX_EXTRACTED_BYTES = 1024 * 1024 * 1024
MANAGED_OUTPUT_NAMES = {ARCHIVE_NAME, "images", "manifest.json"}


@dataclass(frozen=True)
class PpmHeader:
    width: int
    height: int
    channel_maxima: tuple[int, int, int]
    bytes_per_sample: int
    raster_offset: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the 250 MiB processed COIL-100 PPM tarball from Columbia, "
            "safely extract its 7,200 images, and record checksums/provenance."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"destination directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--url",
        default=ARCHIVE_URL,
        help=f"archive URL (default: {ARCHIVE_URL})",
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
        default=60.0,
        help="per-operation network timeout in seconds (default: 60)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an earlier COIL-100 download after validation",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(DOWNLOAD_BLOCK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_with_progress(
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
            progress = f"  {downloaded / (1024 * 1024):.0f} MiB"
            if expected_bytes:
                progress += f" ({100.0 * downloaded / expected_bytes:.0f}%)"
            print(progress, flush=True)
            next_report += PROGRESS_INTERVAL
    return downloaded


def download_once(
    url: str,
    destination: Path,
    timeout: float = 60.0,
) -> dict[str, object]:
    request = Request(url, headers={"User-Agent": "ultrahigh-ann/0.2"})
    with urlopen(request, timeout=timeout) as response:
        length_header = response.headers.get("Content-Length")
        expected_bytes = int(length_header) if length_header else None
        with destination.open("wb") as output:
            downloaded_bytes = copy_with_progress(
                response,
                output,
                expected_bytes,
            )

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


def _content_line(stream: BinaryIO, path: Path, field: str) -> bytes:
    while True:
        line = stream.readline(4097)
        if not line:
            raise RuntimeError(f"{path.name} ends before its {field}")
        if len(line) > 4096:
            raise RuntimeError(f"{path.name} has an unreasonable PPM header")
        content = line.partition(b"#")[0].strip()
        if content:
            return content


def read_ppm_header(path: Path) -> PpmHeader:
    """Read the COIL variant of P6, which stores one maximum per channel."""
    with path.open("rb") as source:
        if source.readline(4097).strip() != b"P6":
            raise RuntimeError(f"{path.name} is not a binary P6 PPM image")

        dimensions = _content_line(source, path, "dimensions").split()
        if len(dimensions) != 2:
            raise RuntimeError(f"{path.name} has malformed PPM dimensions")
        try:
            width, height = (int(value) for value in dimensions)
        except ValueError as error:
            raise RuntimeError(
                f"{path.name} has non-integer PPM dimensions"
            ) from error

        maximum_fields = _content_line(source, path, "channel maximum").split()
        if len(maximum_fields) not in (1, 3):
            raise RuntimeError(
                f"{path.name} must contain one or three PPM channel maxima"
            )
        try:
            maxima = tuple(int(value) for value in maximum_fields)
        except ValueError as error:
            raise RuntimeError(
                f"{path.name} has a non-integer PPM channel maximum"
            ) from error
        if len(maxima) == 1:
            maxima = maxima * 3
        if width < 1 or height < 1 or any(value < 1 for value in maxima):
            raise RuntimeError(f"{path.name} has an invalid PPM header value")
        if any(value > 65_535 for value in maxima):
            raise RuntimeError(
                f"{path.name} uses unsupported PPM samples above 16 bits"
            )
        bytes_per_sample = 1 if max(maxima) <= 255 else 2
        raster_offset = source.tell()

    expected_size = (
        raster_offset + width * height * 3 * bytes_per_sample
    )
    observed_size = path.stat().st_size
    if observed_size != expected_size:
        raise RuntimeError(
            f"{path.name} contains {observed_size - raster_offset} raster "
            f"bytes; expected {expected_size - raster_offset}"
        )
    return PpmHeader(
        width=width,
        height=height,
        channel_maxima=(maxima[0], maxima[1], maxima[2]),
        bytes_per_sample=bytes_per_sample,
        raster_offset=raster_offset,
    )


def image_key_from_name(name: str) -> tuple[int, int] | None:
    match = IMAGE_PATTERN.fullmatch(name)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def expected_image_keys(
    object_ids: tuple[int, ...] = EXPECTED_OBJECT_IDS,
    angles: tuple[int, ...] = EXPECTED_ANGLES,
) -> set[tuple[int, int]]:
    return {(object_id, angle) for object_id in object_ids for angle in angles}


def collect_image_paths(
    images_dir: Path,
    object_ids: tuple[int, ...] = EXPECTED_OBJECT_IDS,
    angles: tuple[int, ...] = EXPECTED_ANGLES,
) -> dict[tuple[int, int], Path]:
    if not images_dir.is_dir():
        raise RuntimeError(f"image directory does not exist: {images_dir}")
    inventory: dict[tuple[int, int], Path] = {}
    for path in images_dir.iterdir():
        if not path.is_file():
            continue
        key = image_key_from_name(path.name)
        if key is None:
            continue
        if key in inventory:
            raise RuntimeError(f"duplicate COIL-100 image key: {key}")
        inventory[key] = path

    expected = expected_image_keys(object_ids, angles)
    missing = sorted(expected - inventory.keys())
    extra = sorted(inventory.keys() - expected)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing {len(missing)} images (first: {missing[0]})")
        if extra:
            details.append(f"found {len(extra)} unexpected images (first: {extra[0]})")
        raise RuntimeError("invalid COIL-100 image inventory: " + "; ".join(details))
    return inventory


def _safe_member_path(member: tarfile.TarInfo) -> PurePosixPath:
    if "\\" in member.name or "\x00" in member.name:
        raise RuntimeError(f"unsafe tar member name: {member.name!r}")
    path = PurePosixPath(member.name)
    if path.is_absolute() or ".." in path.parts:
        raise RuntimeError(f"unsafe tar member path: {member.name!r}")
    if not member.isfile() and not member.isdir():
        raise RuntimeError(
            f"tar contains a non-regular member: {member.name!r}"
        )
    return path


def extract_and_validate_archive(
    archive_path: Path,
    images_dir: Path,
    object_ids: tuple[int, ...] = EXPECTED_OBJECT_IDS,
    angles: tuple[int, ...] = EXPECTED_ANGLES,
    expected_width: int = EXPECTED_WIDTH,
    expected_height: int = EXPECTED_HEIGHT,
    expected_maxima: tuple[int, int, int] = EXPECTED_CHANNEL_MAXIMA,
) -> dict[str, object]:
    expected = expected_image_keys(object_ids, angles)
    observed: set[tuple[int, int]] = set()
    total_bytes = 0

    try:
        archive = tarfile.open(archive_path, mode="r:gz")
    except (tarfile.TarError, EOFError) as error:
        raise RuntimeError(
            f"download is not a valid gzip-compressed tar archive: {error}"
        ) from error

    with archive:
        images_dir.mkdir(parents=True)
        aggregate_digest = hashlib.sha256()
        extracted_bytes = 0
        for member in archive:
            member_path = _safe_member_path(member)
            if member.isdir():
                continue
            key = image_key_from_name(member_path.name)
            if key is None:
                continue
            if key not in expected:
                raise RuntimeError(
                    f"tar contains unexpected COIL-100 image key: {key}"
                )
            if key in observed:
                raise RuntimeError(f"tar contains duplicate image key: {key}")
            if member.size > MAX_IMAGE_BYTES:
                raise RuntimeError(
                    f"tar image {member_path.name} is unexpectedly large"
                )
            total_bytes += member.size
            if total_bytes > MAX_EXTRACTED_BYTES:
                raise RuntimeError("tar image payload exceeds the safety limit")

            name = f"obj{key[0]}__{key[1]}.ppm"
            path = images_dir / name
            aggregate_digest.update(name.encode("ascii"))
            aggregate_digest.update(b"\0")
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"cannot read tar member {member.name}")
            try:
                with source, path.open("wb") as output:
                    while block := source.read(DOWNLOAD_BLOCK_SIZE):
                        output.write(block)
                        aggregate_digest.update(block)
                        extracted_bytes += len(block)
            except (OSError, RuntimeError, tarfile.TarError) as error:
                raise RuntimeError(
                    f"failed while extracting {member.name}: {error}"
                ) from error
            if path.stat().st_size != member.size:
                raise RuntimeError(f"incomplete extraction of {member.name}")

            header = read_ppm_header(path)
            if (header.width, header.height) != (
                expected_width,
                expected_height,
            ):
                raise RuntimeError(
                    f"{name} is {header.width}x{header.height}; expected "
                    f"{expected_width}x{expected_height}"
                )
            if header.channel_maxima != expected_maxima:
                raise RuntimeError(
                    f"{name} has channel maxima {header.channel_maxima}; "
                    f"expected {expected_maxima}"
                )
            observed.add(key)
            if len(observed) % 720 == 0 or len(observed) == len(expected):
                print(
                    f"  validated {len(observed):,} / {len(expected):,} images",
                    flush=True,
                )

        missing = sorted(expected - observed)
        if missing:
            raise RuntimeError(
                f"invalid COIL-100 archive inventory: missing {len(missing)} "
                f"images (first: {missing[0]})"
            )

    return {
        "directory": "images",
        "count": len(observed),
        "object_count": len(object_ids),
        "views_per_object": len(angles),
        "angles_degrees": list(angles),
        "dimensions": [expected_height, expected_width, 3],
        "source_dtype": "big-endian uint16",
        "channel_maxima": list(expected_maxima),
        "extracted_bytes": extracted_bytes,
        "aggregate_sha256": aggregate_digest.hexdigest(),
        "aggregate_hash_order": "tar member order; normalized filename NUL payload",
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


def validate_args(args: argparse.Namespace) -> None:
    if args.retries < 1:
        raise ValueError("--retries must be at least one")
    if args.timeout <= 0.0:
        raise ValueError("--timeout must be positive")


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
        requested_output_dir = args.output_dir.expanduser()
        if requested_output_dir.is_symlink():
            raise RuntimeError(
                f"refusing to replace symbolic-link output: {requested_output_dir}"
            )
        output_dir = requested_output_dir.resolve()
        if output_dir.exists() and not output_dir.is_dir():
            raise RuntimeError(f"output path is not a directory: {output_dir}")
        existing = []
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
            prefix=".coil100-download-",
            dir=output_dir.parent,
        ) as temporary_directory:
            staging_dir = Path(temporary_directory) / "coil100"
            staging_dir.mkdir()
            archive_path = staging_dir / ARCHIVE_NAME

            print("Downloading the official COIL-100 PPM tarball...", flush=True)
            archive_record = download(
                args.url,
                archive_path,
                args.retries,
                args.timeout,
            )
            print("Safely extracting and validating COIL-100...", flush=True)
            image_record = extract_and_validate_archive(
                archive_path,
                staging_dir / "images",
            )

            manifest = {
                "dataset_name": "Columbia Object Image Library COIL-100",
                "format_version": 1,
                "source": "Columbia University CAVE",
                "source_page_url": SOURCE_PAGE_URL,
                "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                "archive": {"file": ARCHIVE_NAME, **archive_record},
                "images": image_record,
                "filename_pattern": "obj<object_id>__<angle_degrees>.ppm",
                "label_definition": "physical object identity",
            }
            with (staging_dir / "manifest.json").open(
                "w",
                encoding="utf-8",
            ) as output:
                json.dump(manifest, output, indent=2, sort_keys=True)
                output.write("\n")

            install_directory(staging_dir, output_dir)

    except (
        OSError,
        RuntimeError,
        ValueError,
        tarfile.TarError,
    ) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(
        f"Saved {len(EXPECTED_OBJECT_IDS):,} objects and "
        f"{len(EXPECTED_OBJECT_IDS) * len(EXPECTED_ANGLES):,} images to "
        f"{output_dir}.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
