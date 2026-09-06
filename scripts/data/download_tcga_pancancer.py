#!/usr/bin/env python3
"""Download all 33 TCGA STAR-FPKM matrices from the GDC Xena hub."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
import tempfile
import time
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path
from typing import BinaryIO, TextIO
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


XENA_HUB_URL = "https://gdc.xenahubs.net"
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[2] / "data" / "raw" / "tcga_pancancer"
)
EXPECTED_FEATURES = 60_660
DOWNLOAD_BLOCK_SIZE = 1024 * 1024
PROGRESS_INTERVAL = 50 * 1024 * 1024
APPROXIMATE_DOWNLOAD_GIB = 2.8


@dataclass(frozen=True)
class Cohort:
    label: int
    project_id: str
    cancer_type: str

    @property
    def file_name(self) -> str:
        return f"{self.project_id}.star_fpkm.tsv.gz"


COHORT_DEFINITIONS = (
    ("TCGA-ACC", "Adrenocortical Carcinoma"),
    ("TCGA-BLCA", "Bladder Urothelial Carcinoma"),
    ("TCGA-BRCA", "Breast Invasive Carcinoma"),
    (
        "TCGA-CESC",
        "Cervical Squamous Cell Carcinoma and Endocervical Adenocarcinoma",
    ),
    ("TCGA-CHOL", "Cholangiocarcinoma"),
    ("TCGA-COAD", "Colon Adenocarcinoma"),
    (
        "TCGA-DLBC",
        "Lymphoid Neoplasm Diffuse Large B-cell Lymphoma",
    ),
    ("TCGA-ESCA", "Esophageal Carcinoma"),
    ("TCGA-GBM", "Glioblastoma Multiforme"),
    ("TCGA-HNSC", "Head and Neck Squamous Cell Carcinoma"),
    ("TCGA-KICH", "Kidney Chromophobe"),
    ("TCGA-KIRC", "Kidney Renal Clear Cell Carcinoma"),
    ("TCGA-KIRP", "Kidney Renal Papillary Cell Carcinoma"),
    ("TCGA-LAML", "Acute Myeloid Leukemia"),
    ("TCGA-LGG", "Brain Lower Grade Glioma"),
    ("TCGA-LIHC", "Liver Hepatocellular Carcinoma"),
    ("TCGA-LUAD", "Lung Adenocarcinoma"),
    ("TCGA-LUSC", "Lung Squamous Cell Carcinoma"),
    ("TCGA-MESO", "Mesothelioma"),
    ("TCGA-OV", "Ovarian Serous Cystadenocarcinoma"),
    ("TCGA-PAAD", "Pancreatic Adenocarcinoma"),
    ("TCGA-PCPG", "Pheochromocytoma and Paraganglioma"),
    ("TCGA-PRAD", "Prostate Adenocarcinoma"),
    ("TCGA-READ", "Rectum Adenocarcinoma"),
    ("TCGA-SARC", "Sarcoma"),
    ("TCGA-SKCM", "Skin Cutaneous Melanoma"),
    ("TCGA-STAD", "Stomach Adenocarcinoma"),
    ("TCGA-TGCT", "Testicular Germ Cell Tumors"),
    ("TCGA-THCA", "Thyroid Carcinoma"),
    ("TCGA-THYM", "Thymoma"),
    ("TCGA-UCEC", "Uterine Corpus Endometrial Carcinoma"),
    ("TCGA-UCS", "Uterine Carcinosarcoma"),
    ("TCGA-UVM", "Uveal Melanoma"),
)

COHORTS = tuple(
    Cohort(label, project_id, cancer_type)
    for label, (project_id, cancer_type) in enumerate(COHORT_DEFINITIONS)
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download consistent GDC STAR-FPKM matrices for all 33 TCGA "
            f"projects from UCSC Xena (approximately {APPROXIMATE_DOWNLOAD_GIB} "
            "GiB compressed)."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"destination directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="download attempts per file (default: 3)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace files from an earlier download after validation",
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


def download_once(url: str, destination: Path) -> dict[str, object]:
    request = Request(url, headers={"User-Agent": "ultrahigh-ann/0.2"})
    with urlopen(request, timeout=60) as response:
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
) -> dict[str, object]:
    for attempt in range(1, attempts + 1):
        try:
            return download_once(url, destination)
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
                    f"download failed for {url} after {attempts} attempts: "
                    f"{error}"
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


def open_matrix(path: Path) -> TextIO:
    return gzip.open(path, mode="rt", encoding="utf-8", newline="")


def read_header(
    stream: TextIO,
    path: Path,
) -> tuple[str, ...]:
    header = stream.readline()
    if not header:
        raise RuntimeError(f"{path.name} is empty")
    fields = header.rstrip("\r\n").split("\t")
    if len(fields) < 2 or fields[0] != "Ensembl_ID":
        raise RuntimeError(f"{path.name} has an unexpected header")
    sample_ids = tuple(fields[1:])
    if len(set(sample_ids)) != len(sample_ids):
        raise RuntimeError(f"{path.name} contains duplicate sample IDs")
    if any(not sample_id.startswith("TCGA-") for sample_id in sample_ids):
        raise RuntimeError(f"{path.name} contains an unexpected sample ID")
    return sample_ids


def feature_id_from_line(
    line: str,
    path: Path,
    expected_tabs: int,
) -> str:
    feature_id, separator, _ = line.partition("\t")
    if not separator or not feature_id:
        raise RuntimeError(f"{path.name} contains a malformed feature row")
    if line.count("\t") != expected_tabs:
        raise RuntimeError(
            f"{path.name} feature {feature_id!r} has an unexpected width"
        )
    return feature_id


def validate_matrices(
    cohorts: tuple[Cohort, ...],
    matrix_paths: list[Path],
    expected_features: int = EXPECTED_FEATURES,
) -> tuple[list[dict[str, object]], dict[str, int], int]:
    if len(cohorts) != len(matrix_paths):
        raise ValueError("cohort and matrix path counts do not match")

    samples: list[dict[str, object]] = []
    sample_counts: dict[str, int] = {}
    seen_samples: set[str] = set()

    with ExitStack() as stack:
        streams = [stack.enter_context(open_matrix(path)) for path in matrix_paths]
        sample_ids_by_cohort: list[tuple[str, ...]] = []
        for cohort, path, stream in zip(
            cohorts,
            matrix_paths,
            streams,
            strict=True,
        ):
            sample_ids = read_header(stream, path)
            duplicates = seen_samples.intersection(sample_ids)
            if duplicates:
                raise RuntimeError(
                    f"duplicate sample ID across cohorts: {min(duplicates)}"
                )
            seen_samples.update(sample_ids)
            sample_counts[cohort.project_id] = len(sample_ids)
            sample_ids_by_cohort.append(sample_ids)
            samples.extend(
                {
                    "sample_id": sample_id,
                    "project_id": cohort.project_id,
                    "cancer_type": cohort.cancer_type,
                    "label": cohort.label,
                }
                for sample_id in sample_ids
            )

        feature_count = 0
        for lines in zip_longest(*streams):
            if any(line is None for line in lines):
                raise RuntimeError("cohort matrices have different feature counts")

            feature_ids = []
            for line, path, sample_ids in zip(
                lines,
                matrix_paths,
                sample_ids_by_cohort,
                strict=True,
            ):
                assert line is not None
                feature_ids.append(
                    feature_id_from_line(line, path, len(sample_ids))
                )
            if len(set(feature_ids)) != 1:
                raise RuntimeError(
                    "cohort matrices do not have matching Ensembl gene order"
                )
            feature_count += 1
            if feature_count % 10_000 == 0:
                print(
                    f"  validated {feature_count:,} / "
                    f"{expected_features:,} features",
                    flush=True,
                )

    if feature_count != expected_features:
        raise RuntimeError(
            f"matrices contain {feature_count} features; "
            f"expected {expected_features}"
        )
    return samples, sample_counts, feature_count


def write_targets(path: Path, samples: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=(
                "sample_id",
                "project_id",
                "cancer_type",
                "label",
            ),
        )
        writer.writeheader()
        writer.writerows(samples)


def validate_args(args: argparse.Namespace) -> None:
    if args.retries < 1:
        raise ValueError("--retries must be at least one")


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        output_names = {
            *(cohort.file_name for cohort in COHORTS),
            "targets.csv",
            "manifest.json",
        }
        existing = sorted(
            path.name for path in output_dir.iterdir() if path.name in output_names
        )
        if existing and not args.force:
            print(
                f"Refusing to overwrite existing files in {output_dir}: "
                f"{', '.join(existing)}\nRun again with --force to replace them.",
                file=sys.stderr,
            )
            return 1

        with tempfile.TemporaryDirectory(
            prefix=".tcga-pancancer-download-",
            dir=output_dir,
        ) as temporary_directory:
            staging_dir = Path(temporary_directory)
            matrix_paths: list[Path] = []
            source_files: dict[str, dict[str, object]] = {}

            for index, cohort in enumerate(COHORTS, start=1):
                url = f"{XENA_HUB_URL}/download/{cohort.file_name}"
                path = staging_dir / cohort.file_name
                print(
                    f"[{index:02}/{len(COHORTS)}] Downloading "
                    f"{cohort.project_id} from UCSC Xena...",
                    flush=True,
                )
                source_files[cohort.file_name] = download(
                    url,
                    path,
                    args.retries,
                )
                matrix_paths.append(path)

            print(
                "Validating sample IDs, row widths, and feature alignment...",
                flush=True,
            )
            samples, sample_counts, feature_count = validate_matrices(
                COHORTS,
                matrix_paths,
            )
            targets_path = staging_dir / "targets.csv"
            write_targets(targets_path, samples)

            manifest = {
                "dataset_name": "TCGA Pan-Cancer STAR-FPKM",
                "source": "UCSC Xena GDC Hub",
                "source_hub_url": XENA_HUB_URL,
                "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                "matrix_orientation": "features_by_samples",
                "feature_identifier": "versioned Ensembl gene ID",
                "value_unit": "log2(FPKM+1)",
                "feature_count": feature_count,
                "sample_count": len(samples),
                "cohort_count": len(COHORTS),
                "label_order": "alphabetical_by_project_id",
                "cohorts": [
                    {
                        "label": cohort.label,
                        "project_id": cohort.project_id,
                        "cancer_type": cohort.cancer_type,
                        "sample_count": sample_counts[cohort.project_id],
                        "file": cohort.file_name,
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
                "w",
                encoding="utf-8",
            ) as output:
                json.dump(manifest, output, indent=2, sort_keys=True)
                output.write("\n")

            for name in output_names:
                staged_path = staging_dir / name
                destination = output_dir / name
                if not staged_path.exists():
                    continue
                staged_path.replace(destination)

    except (OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(
        f"Saved {len(samples):,} samples from {len(COHORTS)} projects across "
        f"{feature_count:,} features to {output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
