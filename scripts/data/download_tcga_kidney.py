#!/usr/bin/env python3
"""Download the three TCGA Kidney Cancers matrices from the GDC-backed Xena hub."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
import tempfile
from contextlib import ExitStack
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path
from typing import BinaryIO
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


UCI_REFERENCE_URL = (
    "https://archive.ics.uci.edu/dataset/892/tcga%2Bkidney%2Bcancers"
)
XENA_HUB_URL = "https://gdc.xenahubs.net"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "tcga"
EXPECTED_FEATURES = 60_660
DOWNLOAD_BLOCK_SIZE = 1024 * 1024

COHORTS = (
    {
        "project_id": "TCGA-KICH",
        "subtype": "Kidney Chromophobe",
        "expected_samples": 91,
        "file_name": "TCGA-KICH.star_fpkm.tsv.gz",
    },
    {
        "project_id": "TCGA-KIRC",
        "subtype": "Kidney Renal Clear Cell Carcinoma",
        "expected_samples": 610,
        "file_name": "TCGA-KIRC.star_fpkm.tsv.gz",
    },
    {
        "project_id": "TCGA-KIRP",
        "subtype": "Kidney Renal Papillary Cell Carcinoma",
        "expected_samples": 323,
        "file_name": "TCGA-KIRP.star_fpkm.tsv.gz",
    },
)

LEGACY_OUTPUT_NAMES = {
    "features.csv.gz",
    "ids.csv",
    "metadata.json",
    "variables.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the three GDC TCGA kidney-cancer STAR-FPKM matrices "
            "published by UCSC Xena."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"destination directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace files from an earlier download",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(DOWNLOAD_BLOCK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_with_progress(
    source: BinaryIO, destination: BinaryIO, expected_bytes: int | None
) -> int:
    downloaded = 0
    next_report = 25 * 1024 * 1024

    while block := source.read(DOWNLOAD_BLOCK_SIZE):
        destination.write(block)
        downloaded += len(block)
        if downloaded >= next_report:
            if expected_bytes:
                percent = 100.0 * downloaded / expected_bytes
                print(
                    f"  {downloaded / (1024 * 1024):.0f} MiB "
                    f"({percent:.0f}%)",
                    flush=True,
                )
            else:
                print(
                    f"  {downloaded / (1024 * 1024):.0f} MiB",
                    flush=True,
                )
            next_report += 25 * 1024 * 1024

    return downloaded


def download(url: str, destination: Path) -> dict[str, object]:
    request = Request(url, headers={"User-Agent": "ultrahigh-ann/0.1"})

    try:
        with urlopen(request, timeout=60) as response:
            length_header = response.headers.get("Content-Length")
            expected_bytes = int(length_header) if length_header else None
            with destination.open("wb") as output:
                downloaded_bytes = copy_with_progress(
                    response, output, expected_bytes
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
    except (HTTPError, URLError, TimeoutError) as error:
        raise RuntimeError(f"download failed for {url}: {error}") from error


def validate_matrices(
    matrix_paths: list[Path],
) -> tuple[list[dict[str, str]], int]:
    samples: list[dict[str, str]] = []
    seen_samples: set[str] = set()

    with ExitStack() as stack:
        readers = []
        widths = []

        for cohort, path in zip(COHORTS, matrix_paths, strict=True):
            stream = stack.enter_context(
                gzip.open(path, mode="rt", encoding="utf-8", newline="")
            )
            reader = csv.reader(stream, delimiter="\t")
            header = next(reader, None)
            if not header or header[0] != "Ensembl_ID":
                raise RuntimeError(f"{path.name} has an unexpected header")

            sample_ids = header[1:]
            expected_samples = cohort["expected_samples"]
            if len(sample_ids) != expected_samples:
                raise RuntimeError(
                    f"{path.name} contains {len(sample_ids)} samples; "
                    f"expected {expected_samples}"
                )

            duplicates = seen_samples.intersection(sample_ids)
            if duplicates:
                example = min(duplicates)
                raise RuntimeError(f"duplicate sample ID across cohorts: {example}")
            seen_samples.update(sample_ids)

            for sample_id in sample_ids:
                samples.append(
                    {
                        "sample_id": sample_id,
                        "project_id": str(cohort["project_id"]),
                        "cancer_subtype": str(cohort["subtype"]),
                    }
                )

            readers.append(reader)
            widths.append(len(header))

        feature_count = 0
        for rows in zip_longest(*readers):
            if any(row is None for row in rows):
                raise RuntimeError("cohort matrices have different feature counts")

            assert all(row is not None for row in rows)
            feature_ids = [row[0] for row in rows]
            if len(set(feature_ids)) != 1:
                raise RuntimeError(
                    "cohort matrices do not have matching Ensembl gene order"
                )

            for path, row, width in zip(
                matrix_paths, rows, widths, strict=True
            ):
                if len(row) != width:
                    raise RuntimeError(
                        f"{path.name} has a malformed row for {row[0]}"
                    )

            feature_count += 1

    if feature_count != EXPECTED_FEATURES:
        raise RuntimeError(
            f"matrices contain {feature_count} features; "
            f"expected {EXPECTED_FEATURES}"
        )

    return samples, feature_count


def write_targets(path: Path, samples: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=("sample_id", "project_id", "cancer_subtype"),
        )
        writer.writeheader()
        writer.writerows(samples)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    output_names = {
        *(str(cohort["file_name"]) for cohort in COHORTS),
        "targets.csv",
        "manifest.json",
    }
    managed_names = output_names | LEGACY_OUTPUT_NAMES
    existing = sorted(
        path.name for path in output_dir.iterdir() if path.name in managed_names
    )
    if existing and not args.force:
        print(
            f"Refusing to overwrite existing files in {output_dir}: "
            f"{', '.join(existing)}\nRun again with --force to replace them.",
            file=sys.stderr,
        )
        return 1

    try:
        with tempfile.TemporaryDirectory(
            prefix=".tcga-download-", dir=output_dir
        ) as temp:
            staging_dir = Path(temp)
            matrix_paths = []
            source_files = {}

            for cohort in COHORTS:
                file_name = str(cohort["file_name"])
                url = f"{XENA_HUB_URL}/download/{file_name}"
                path = staging_dir / file_name
                print(f"Downloading {cohort['project_id']} from UCSC Xena...")
                source_files[file_name] = download(url, path)
                matrix_paths.append(path)

            print("Validating sample IDs and feature alignment...")
            samples, feature_count = validate_matrices(matrix_paths)
            targets_path = staging_dir / "targets.csv"
            write_targets(targets_path, samples)

            manifest = {
                "dataset_name": "TCGA Kidney Cancers",
                "uci_dataset_id": 892,
                "uci_reference_url": UCI_REFERENCE_URL,
                "source": "UCSC Xena GDC Hub",
                "source_hub_url": XENA_HUB_URL,
                "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                "matrix_orientation": "features_by_samples",
                "feature_identifier": "versioned Ensembl gene ID",
                "value_unit": "log2(FPKM+1)",
                "feature_count": feature_count,
                "sample_count": len(samples),
                "cohorts": [
                    {
                        "project_id": cohort["project_id"],
                        "cancer_subtype": cohort["subtype"],
                        "sample_count": cohort["expected_samples"],
                    }
                    for cohort in COHORTS
                ],
                "files": {
                    **source_files,
                    "targets.csv": {
                        "bytes": targets_path.stat().st_size,
                        "sha256": sha256(targets_path),
                    },
                },
            }
            with (staging_dir / "manifest.json").open(
                "w", encoding="utf-8"
            ) as output:
                json.dump(manifest, output, indent=2, sort_keys=True)
                output.write("\n")

            staged_names = {path.name for path in staging_dir.iterdir()}
            for name in managed_names - staged_names:
                old_path = output_dir / name
                if old_path.exists():
                    old_path.unlink()

            for path in staging_dir.iterdir():
                path.replace(output_dir / path.name)

    except (OSError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(
        f"Saved {len(samples)} samples across {feature_count} features to "
        f"{output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
