#!/usr/bin/env python3
"""Download and validate the official GSE2034 expression and clinical tables."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import html
import json
import math
import re
import shutil
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "gse2034"
ACCESSION = "GSE2034"
PLATFORM = "GPL96"
PUBMED_ID = "15721472"
ACCESSION_URL = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE2034"
MATRIX_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE2nnn/GSE2034/"
    "matrix/GSE2034_series_matrix.txt.gz"
)
MATRIX_NAME = "GSE2034_series_matrix.txt.gz"
CLINICAL_NAME = "GSE2034_clinical.tsv"
EXPECTED_SAMPLES = 286
EXPECTED_FEATURES = 22_283
EXPECTED_RELAPSE_COUNTS = {0: 179, 1: 107}
DOWNLOAD_BLOCK_SIZE = 1024 * 1024
PROGRESS_INTERVAL = 8 * 1024 * 1024
MANAGED_OUTPUT_NAMES = {MATRIX_NAME, CLINICAL_NAME, "manifest.json"}

CLINICAL_COLUMNS = (
    "PID",
    "GEO asscession number",
    "lymph node status",
    "time to relapse or last follow-up (months)",
    "relapse (1=True)",
    "ER Status",
    "Brain relapses (1=yes, 0=no)",
)


@dataclass(frozen=True)
class ClinicalSample:
    patient_id: str
    geo_accession: str
    lymph_node_status: str
    follow_up_months: float
    relapse: int
    er_status: str
    brain_relapse: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the official GEO series matrix and clinical outcome "
            "table for GSE2034, validate their alignment, and record provenance."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"destination directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--matrix-url",
        default=MATRIX_URL,
        help=f"series-matrix URL (default: {MATRIX_URL})",
    )
    parser.add_argument(
        "--accession-url",
        default=ACCESSION_URL,
        help=(
            "GEO accession page used to discover the current clinical-table "
            f"endpoint (default: {ACCESSION_URL})"
        ),
    )
    parser.add_argument(
        "--clinical-url",
        default=None,
        help=(
            "raw clinical-table URL override; by default it is discovered "
            "from the GEO accession page"
        ),
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="download attempts per file (default: 3)",
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
        help="replace a previously validated GSE2034 download",
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
    timeout: float = 90.0,
) -> dict[str, object]:
    request = Request(url, headers={"User-Agent": "ultrahigh-ann/0.1"})
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


def discover_clinical_url(accession_page: str, page_url: str) -> str:
    """Discover GEO's internal blob identifier instead of hard-coding it."""
    decoded = html.unescape(accession_page)
    pattern = re.compile(
        r"['\"]([^'\"]*/geo/query/acc\.cgi\?view=data&acc=GSE2034"
        r"&id=[0-9]+&db=GeoDb_blob[0-9]+)['\"]"
    )
    matches = pattern.findall(decoded)
    if len(matches) != 1:
        raise RuntimeError(
            "could not uniquely discover the GSE2034 clinical table from "
            f"the accession page (found {len(matches)} candidates)"
        )

    view_url = urljoin(page_url, matches[0])
    parsed = urlsplit(view_url)
    parameters = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key != "view"
    ]
    parameters[0:0] = [("mode", "raw"), ("is_datatable", "true")]
    return urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, urlencode(parameters), "")
    )


def _single_metadata_value(
    metadata: dict[str, list[list[str]]],
    key: str,
) -> str:
    rows = metadata.get(key, [])
    if len(rows) != 1 or len(rows[0]) != 1:
        raise RuntimeError(f"matrix has an unexpected {key} declaration")
    return rows[0][0]


def validate_series_matrix(
    path: Path,
    expected_samples: int = EXPECTED_SAMPLES,
    expected_features: int = EXPECTED_FEATURES,
) -> dict[str, object]:
    metadata: dict[str, list[list[str]]] = {}
    table_started = False
    table_ended = False
    sample_accessions: tuple[str, ...] | None = None
    feature_ids: set[str] = set()
    feature_count = 0
    minimum = math.inf
    maximum = -math.inf

    try:
        source_context = gzip.open(
            path,
            mode="rt",
            encoding="utf-8",
            newline="",
        )
        with source_context as source:
            rows = csv.reader(source, delimiter="\t")
            for row_number, row in enumerate(rows, start=1):
                if not row:
                    continue
                key = row[0]
                if not table_started:
                    if key == "!series_matrix_table_begin":
                        table_started = True
                        try:
                            header = next(rows)
                        except StopIteration as error:
                            raise RuntimeError(
                                "series matrix ends before its table header"
                            ) from error
                        if not header or header[0] != "ID_REF":
                            raise RuntimeError(
                                "series matrix has an unexpected table header"
                            )
                        sample_accessions = tuple(header[1:])
                        if len(sample_accessions) != expected_samples:
                            raise RuntimeError(
                                f"series matrix has {len(sample_accessions)} "
                                f"samples; expected {expected_samples}"
                            )
                        if len(set(sample_accessions)) != len(sample_accessions):
                            raise RuntimeError(
                                "series matrix contains duplicate sample accessions"
                            )
                        declared_rows = metadata.get(
                            "!Sample_geo_accession",
                            [],
                        )
                        if len(declared_rows) != 1:
                            raise RuntimeError(
                                "matrix has an unexpected sample-accession declaration"
                            )
                        if tuple(declared_rows[0]) != sample_accessions:
                            raise RuntimeError(
                                "matrix table columns do not match its sample metadata"
                            )
                        continue
                    if key.startswith("!"):
                        metadata.setdefault(key, []).append(row[1:])
                    continue

                if key == "!series_matrix_table_end":
                    table_ended = True
                    break
                if sample_accessions is None:
                    raise RuntimeError("series matrix table is missing its header")
                if len(row) != len(sample_accessions) + 1:
                    raise RuntimeError(
                        f"matrix row {row_number} has {len(row) - 1} values; "
                        f"expected {len(sample_accessions)}"
                    )
                feature_id = row[0]
                if not feature_id or feature_id in feature_ids:
                    raise RuntimeError(
                        f"invalid or duplicate feature ID at row {row_number}: "
                        f"{feature_id!r}"
                    )
                feature_ids.add(feature_id)
                try:
                    values = [float(value) for value in row[1:]]
                except ValueError as error:
                    raise RuntimeError(
                        f"matrix row {row_number} contains a non-numeric value"
                    ) from error
                if not all(math.isfinite(value) for value in values):
                    raise RuntimeError(
                        f"matrix row {row_number} contains a non-finite value"
                    )
                minimum = min(minimum, min(values))
                maximum = max(maximum, max(values))
                feature_count += 1
    except (gzip.BadGzipFile, UnicodeDecodeError, csv.Error) as error:
        raise RuntimeError(f"cannot parse {path.name}: {error}") from error

    if not table_started or not table_ended or sample_accessions is None:
        raise RuntimeError("series matrix does not contain a complete data table")
    if feature_count != expected_features:
        raise RuntimeError(
            f"series matrix has {feature_count} features; "
            f"expected {expected_features}"
        )
    if _single_metadata_value(metadata, "!Series_geo_accession") != ACCESSION:
        raise RuntimeError(f"series matrix is not for {ACCESSION}")
    if _single_metadata_value(metadata, "!Series_platform_id") != PLATFORM:
        raise RuntimeError(f"series matrix is not on platform {PLATFORM}")
    if _single_metadata_value(metadata, "!Series_pubmed_id") != PUBMED_ID:
        raise RuntimeError(f"series matrix has an unexpected PubMed ID")

    return {
        "samples": len(sample_accessions),
        "features": feature_count,
        "platform": PLATFORM,
        "pubmed_id": PUBMED_ID,
        "minimum_value": minimum,
        "maximum_value": maximum,
        "sample_accessions": list(sample_accessions),
    }


def read_clinical_samples(path: Path) -> list[ClinicalSample]:
    try:
        with path.open(encoding="utf-8", newline="") as source:
            data_lines = (line for line in source if not line.startswith("#"))
            reader = csv.DictReader(data_lines, delimiter="\t")
            if tuple(reader.fieldnames or ()) != CLINICAL_COLUMNS:
                raise RuntimeError(
                    f"{path.name} has an unexpected clinical-table header"
                )
            samples: list[ClinicalSample] = []
            for row_number, row in enumerate(reader, start=2):
                try:
                    sample = ClinicalSample(
                        patient_id=row[CLINICAL_COLUMNS[0]],
                        geo_accession=row[CLINICAL_COLUMNS[1]],
                        lymph_node_status=row[CLINICAL_COLUMNS[2]],
                        follow_up_months=float(row[CLINICAL_COLUMNS[3]]),
                        relapse=int(row[CLINICAL_COLUMNS[4]]),
                        er_status=row[CLINICAL_COLUMNS[5]],
                        brain_relapse=int(row[CLINICAL_COLUMNS[6]]),
                    )
                except (KeyError, TypeError, ValueError) as error:
                    raise RuntimeError(
                        f"invalid clinical value at data row {row_number}"
                    ) from error
                samples.append(sample)
    except (UnicodeDecodeError, csv.Error) as error:
        raise RuntimeError(f"cannot parse {path.name}: {error}") from error
    return samples


def validate_clinical_table(
    path: Path,
    matrix_accessions: list[str] | tuple[str, ...],
    expected_counts: dict[int, int] | None = EXPECTED_RELAPSE_COUNTS,
) -> dict[str, object]:
    samples = read_clinical_samples(path)
    accessions = [sample.geo_accession for sample in samples]
    if len(samples) != len(matrix_accessions):
        raise RuntimeError(
            f"clinical table has {len(samples)} samples; "
            f"matrix has {len(matrix_accessions)}"
        )
    if len(set(accessions)) != len(accessions):
        raise RuntimeError("clinical table contains duplicate GEO accessions")
    if set(accessions) != set(matrix_accessions):
        missing = sorted(set(matrix_accessions) - set(accessions))
        extra = sorted(set(accessions) - set(matrix_accessions))
        raise RuntimeError(
            "clinical and expression samples differ "
            f"(missing={missing[:1]}, extra={extra[:1]})"
        )
    if any(sample.lymph_node_status != "negative" for sample in samples):
        raise RuntimeError("clinical table contains a non-negative lymph-node case")
    if any(sample.follow_up_months < 0.0 for sample in samples):
        raise RuntimeError("clinical table contains negative follow-up time")
    if any(sample.relapse not in (0, 1) for sample in samples):
        raise RuntimeError("clinical table contains a non-binary relapse label")
    if any(sample.brain_relapse not in (0, 1) for sample in samples):
        raise RuntimeError("clinical table contains a non-binary brain-relapse flag")
    if any(sample.er_status not in ("ER-", "ER+") for sample in samples):
        raise RuntimeError("clinical table contains an unknown ER status")

    counts = Counter(sample.relapse for sample in samples)
    if expected_counts is not None and dict(counts) != expected_counts:
        raise RuntimeError(
            f"clinical relapse counts are {dict(counts)}; "
            f"expected {expected_counts}"
        )
    return {
        "samples": len(samples),
        "relapse_counts": {
            "relapse_free_0": counts[0],
            "distant_metastasis_1": counts[1],
        },
        "er_status_counts": dict(sorted(Counter(
            sample.er_status for sample in samples
        ).items())),
        "brain_relapse_counts": {
            str(label): count
            for label, count in sorted(Counter(
                sample.brain_relapse for sample in samples
            ).items())
        },
        "alignment": "joined to expression columns by GEO sample accession",
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
        existing: list[str] = []
        if output_dir.is_dir():
            existing = sorted(
                path.name
                for path in output_dir.iterdir()
                if path.name != ".gitkeep"
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
            prefix=".gse2034-download-",
            dir=output_dir.parent,
        ) as temporary_directory:
            staging_dir = Path(temporary_directory) / "gse2034"
            staging_dir.mkdir()
            matrix_path = staging_dir / MATRIX_NAME
            clinical_path = staging_dir / CLINICAL_NAME

            print("Downloading the official GSE2034 series matrix...", flush=True)
            matrix_record = download(
                args.matrix_url,
                matrix_path,
                args.retries,
                args.timeout,
            )

            accession_record: dict[str, object] | None = None
            if args.clinical_url:
                clinical_url = args.clinical_url
            else:
                accession_path = Path(temporary_directory) / "accession.html"
                print("Discovering the official clinical table...", flush=True)
                accession_record = download(
                    args.accession_url,
                    accession_path,
                    args.retries,
                    args.timeout,
                )
                accession_page = accession_path.read_text(encoding="utf-8")
                clinical_url = discover_clinical_url(
                    accession_page,
                    str(accession_record["resolved_url"]),
                )

            print("Downloading the official clinical outcome table...", flush=True)
            clinical_record = download(
                clinical_url,
                clinical_path,
                args.retries,
                args.timeout,
            )

            print("Validating expression values and clinical labels...", flush=True)
            matrix_validation = validate_series_matrix(matrix_path)
            clinical_validation = validate_clinical_table(
                clinical_path,
                matrix_validation["sample_accessions"],
            )
            del matrix_validation["sample_accessions"]

            manifest = {
                "dataset_name": "GSE2034 breast cancer relapse-free survival",
                "format_version": 1,
                "accession": ACCESSION,
                "platform": PLATFORM,
                "pubmed_id": PUBMED_ID,
                "source": "NCBI Gene Expression Omnibus",
                "source_page_url": ACCESSION_URL,
                "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                "series_matrix": {
                    "file": MATRIX_NAME,
                    **matrix_record,
                    "validation": matrix_validation,
                },
                "clinical_table": {
                    "file": CLINICAL_NAME,
                    **clinical_record,
                    "discovery_page": accession_record,
                    "validation": clinical_validation,
                },
                "label_definition": {
                    "0": "relapse-free at last follow-up",
                    "1": "developed distant metastasis",
                    "source_column": "relapse (1=True)",
                },
                "source_discrepancy": (
                    "The current GEO clinical table contains 179 label-0 and "
                    "107 label-1 cases. The older series summary says 180 and "
                    "106; this pipeline uses the patient-level clinical table."
                ),
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
        json.JSONDecodeError,
    ) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(
        f"Saved {EXPECTED_SAMPLES} samples and {EXPECTED_FEATURES:,} probe sets "
        f"to {output_dir}.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
