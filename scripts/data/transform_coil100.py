#!/usr/bin/env python3
"""Transform COIL-100 into held-out queries and object centroids."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from download_coil100 import (
    EXPECTED_ANGLES,
    EXPECTED_HEIGHT,
    EXPECTED_OBJECT_IDS,
    EXPECTED_WIDTH,
    PpmHeader,
    collect_image_paths,
    read_ppm_header,
)

try:
    import numpy as np
except ModuleNotFoundError as error:
    print(
        "transform_coil100.py requires NumPy; install it with "
        "'python3 -m pip install numpy'.",
        file=sys.stderr,
    )
    raise SystemExit(2) from error


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "coil100"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "coil100_v1"
DEFAULT_QUERY_START_ANGLE = 0
DEFAULT_QUERY_VIEW_COUNT = 18
CHANNELS = ("red", "green", "blue")
HASH_BLOCK_SIZE = 1024 * 1024
REQUIRED_OUTPUT_NAMES = {
    "reference_vectors.npy",
    "reference_labels.npy",
    "reference_vectors.csv",
    "queries.npy",
    "query_labels.npy",
    "samples.csv",
    "features.csv",
    "dataset.json",
    "source_manifest.json",
}
OPTIONAL_OUTPUT_NAMES = {
    "training_pool.npy",
    "training_pool_labels.npy",
}
MANAGED_OUTPUT_NAMES = REQUIRED_OUTPUT_NAMES | OPTIONAL_OUTPUT_NAMES


@dataclass(frozen=True)
class TransformLayout:
    object_ids: tuple[int, ...]
    all_angles: tuple[int, ...]
    query_angles: tuple[int, ...]
    pool_angles: tuple[int, ...]
    width: int
    height: int

    @property
    def dimensions(self) -> int:
        return self.width * self.height * len(CHANNELS)

    @property
    def query_count(self) -> int:
        return len(self.object_ids) * len(self.query_angles)

    @property
    def pool_count(self) -> int:
        return len(self.object_ids) * len(self.pool_angles)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decode the 16-bit COIL-100 images, hold out contiguous azimuth "
            "ranges as queries, and construct one training-only centroid per "
            "physical object."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"download directory (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"destination directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--query-start-angle",
        type=int,
        default=DEFAULT_QUERY_START_ANGLE,
        help=(
            "first angle in the cyclic held-out block, in degrees "
            f"(default: {DEFAULT_QUERY_START_ANGLE})"
        ),
    )
    parser.add_argument(
        "--query-view-count",
        type=int,
        default=DEFAULT_QUERY_VIEW_COUNT,
        help=(
            "number of consecutive 5-degree views held out per object "
            f"(default: {DEFAULT_QUERY_VIEW_COUNT})"
        ),
    )
    parser.add_argument(
        "--write-training-pool",
        action="store_true",
        help=(
            "also materialize all non-query views and labels (about 1.0 GiB "
            "with the default split)"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace managed files from an earlier transformation",
    )
    parser.add_argument(
        "--expected-object-count",
        type=int,
        default=len(EXPECTED_OBJECT_IDS),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--angle-step",
        type=int,
        default=EXPECTED_ANGLES[1] - EXPECTED_ANGLES[0],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=EXPECTED_WIDTH,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=EXPECTED_HEIGHT,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def build_layout(args: argparse.Namespace) -> TransformLayout:
    if args.expected_object_count < 1 or args.expected_object_count > 65_535:
        raise ValueError("--expected-object-count must be between 1 and 65535")
    if args.angle_step < 1 or 360 % args.angle_step != 0:
        raise ValueError("--angle-step must be a positive divisor of 360")
    if args.image_width < 1 or args.image_height < 1:
        raise ValueError("image dimensions must be positive")
    if not 0 <= args.query_start_angle < 360:
        raise ValueError("--query-start-angle must be in [0, 360)")
    if args.query_start_angle % args.angle_step != 0:
        raise ValueError(
            "--query-start-angle must be aligned to the source angle step"
        )

    all_angles = tuple(range(0, 360, args.angle_step))
    if not 1 <= args.query_view_count < len(all_angles):
        raise ValueError(
            f"--query-view-count must be between 1 and {len(all_angles) - 1}"
        )
    query_angles = tuple(
        (args.query_start_angle + index * args.angle_step) % 360
        for index in range(args.query_view_count)
    )
    query_set = set(query_angles)
    pool_angles = tuple(angle for angle in all_angles if angle not in query_set)
    return TransformLayout(
        object_ids=tuple(range(1, args.expected_object_count + 1)),
        all_angles=all_angles,
        query_angles=query_angles,
        pool_angles=pool_angles,
        width=args.image_width,
        height=args.image_height,
    )


def decode_ppm(path: Path, header: PpmHeader | None = None) -> np.ndarray:
    if header is None:
        header = read_ppm_header(path)
    dtype = np.dtype("u1") if header.bytes_per_sample == 1 else np.dtype(">u2")
    samples = np.fromfile(
        path,
        dtype=dtype,
        count=header.width * header.height * len(CHANNELS),
        offset=header.raster_offset,
    )
    expected_samples = header.width * header.height * len(CHANNELS)
    if samples.size != expected_samples:
        raise RuntimeError(
            f"{path.name} contains {samples.size} samples; expected "
            f"{expected_samples}"
        )

    pixels = samples.reshape(-1, len(CHANNELS)).astype(
        np.dtype("<f4"),
        copy=False,
    )
    pixels /= np.asarray(header.channel_maxima, dtype=np.dtype("<f4"))
    if not np.isfinite(pixels).all() or np.any(pixels < 0.0) or np.any(pixels > 1.0):
        raise RuntimeError(f"{path.name} produced an invalid normalized pixel")
    return pixels.reshape(-1)


def validate_header(path: Path, layout: TransformLayout) -> PpmHeader:
    header = read_ppm_header(path)
    if (header.width, header.height) != (layout.width, layout.height):
        raise RuntimeError(
            f"{path.name} is {header.width}x{header.height}; expected "
            f"{layout.width}x{layout.height}"
        )
    return header


def write_matrices(
    inventory: dict[tuple[int, int], Path],
    layout: TransformLayout,
    staging_dir: Path,
    write_training_pool: bool,
) -> None:
    reference_vectors = np.lib.format.open_memmap(
        staging_dir / "reference_vectors.npy",
        mode="w+",
        dtype=np.dtype("<f4"),
        shape=(len(layout.object_ids), layout.dimensions),
        fortran_order=False,
    )
    queries = np.lib.format.open_memmap(
        staging_dir / "queries.npy",
        mode="w+",
        dtype=np.dtype("<f4"),
        shape=(layout.query_count, layout.dimensions),
        fortran_order=False,
    )
    training_pool = None
    if write_training_pool:
        training_pool = np.lib.format.open_memmap(
            staging_dir / "training_pool.npy",
            mode="w+",
            dtype=np.dtype("<f4"),
            shape=(layout.pool_count, layout.dimensions),
            fortran_order=False,
        )

    try:
        query_row = 0
        pool_row = 0
        for object_index, object_id in enumerate(layout.object_ids):
            centroid_sum = np.zeros(layout.dimensions, dtype=np.float64)
            for angle in layout.pool_angles:
                path = inventory[(object_id, angle)]
                pixels = decode_ppm(path, validate_header(path, layout))
                centroid_sum += pixels
                if training_pool is not None:
                    training_pool[pool_row, :] = pixels
                pool_row += 1

            reference_vectors[object_index, :] = (
                centroid_sum / len(layout.pool_angles)
            )
            for angle in layout.query_angles:
                path = inventory[(object_id, angle)]
                queries[query_row, :] = decode_ppm(
                    path,
                    validate_header(path, layout),
                )
                query_row += 1

            if (object_index + 1) % 10 == 0 or object_index + 1 == len(
                layout.object_ids
            ):
                print(
                    f"  converted {object_index + 1:,} / "
                    f"{len(layout.object_ids):,} objects",
                    flush=True,
                )

        if query_row != layout.query_count or pool_row != layout.pool_count:
            raise RuntimeError("internal COIL-100 output row count mismatch")
        reference_vectors.flush()
        queries.flush()
        if training_pool is not None:
            training_pool.flush()
    finally:
        del reference_vectors
        del queries
        if training_pool is not None:
            del training_pool


def write_labels(
    path: Path,
    object_ids: tuple[int, ...],
    rows_per_object: int,
) -> None:
    labels = np.repeat(
        np.arange(len(object_ids), dtype=np.dtype("<u2")),
        rows_per_object,
    )
    np.save(path, labels, allow_pickle=False)


def role_row_maps(
    layout: TransformLayout,
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], int]]:
    pool_rows: dict[tuple[int, int], int] = {}
    query_rows: dict[tuple[int, int], int] = {}
    for object_index, object_id in enumerate(layout.object_ids):
        for local_row, angle in enumerate(layout.pool_angles):
            pool_rows[(object_id, angle)] = (
                object_index * len(layout.pool_angles) + local_row
            )
        for local_row, angle in enumerate(layout.query_angles):
            query_rows[(object_id, angle)] = (
                object_index * len(layout.query_angles) + local_row
            )
    return pool_rows, query_rows


def write_samples(path: Path, layout: TransformLayout) -> None:
    pool_rows, query_rows = role_row_maps(layout)
    with path.open("w", encoding="utf-8", newline="") as output:
        fieldnames = (
            "sample_index",
            "role",
            "role_row_index",
            "file",
            "object_id",
            "label",
            "angle_degrees",
        )
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        sample_index = 0
        for label, object_id in enumerate(layout.object_ids):
            for angle in layout.all_angles:
                key = (object_id, angle)
                if key in query_rows:
                    role = "query"
                    role_row = query_rows[key]
                else:
                    role = "training_pool"
                    role_row = pool_rows[key]
                writer.writerow(
                    {
                        "sample_index": sample_index,
                        "role": role,
                        "role_row_index": role_row,
                        "file": f"images/obj{object_id}__{angle}.ppm",
                        "object_id": object_id,
                        "label": label,
                        "angle_degrees": angle,
                    }
                )
                sample_index += 1


def write_reference_vectors(path: Path, layout: TransformLayout) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        fieldnames = (
            "row_index",
            "object_id",
            "label",
            "construction",
            "pool_view_count",
            "pool_angles_degrees",
        )
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for label, object_id in enumerate(layout.object_ids):
            writer.writerow(
                {
                    "row_index": label,
                    "object_id": object_id,
                    "label": label,
                    "construction": "arithmetic_mean_centroid",
                    "pool_view_count": len(layout.pool_angles),
                    "pool_angles_degrees": ";".join(
                        str(angle) for angle in layout.pool_angles
                    ),
                }
            )


def write_features(path: Path, layout: TransformLayout) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(
            ("column_index", "pixel_row", "pixel_column", "channel")
        )
        column_index = 0
        for pixel_row in range(layout.height):
            for pixel_column in range(layout.width):
                for channel in CHANNELS:
                    writer.writerow(
                        (column_index, pixel_row, pixel_column, channel)
                    )
                    column_index += 1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(HASH_BLOCK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, object]:
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def read_source_manifest(input_dir: Path) -> dict[str, object] | None:
    path = input_dir / "manifest.json"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as source:
        manifest = json.load(source)
    if not isinstance(manifest, dict):
        raise RuntimeError("source manifest must contain a JSON object")
    return manifest


def build_dataset_metadata(
    layout: TransformLayout,
    args: argparse.Namespace,
    staging_dir: Path,
    source_manifest: dict[str, object] | None,
) -> dict[str, object]:
    generated_files = {
        path.name: file_record(path)
        for path in sorted(staging_dir.iterdir())
        if path.is_file() and path.name != "dataset.json"
    }
    matrices: dict[str, object] = {
        "reference_vectors": {
            "file": "reference_vectors.npy",
            "shape": [len(layout.object_ids), layout.dimensions],
            "purpose": "one training-view centroid per physical object",
        },
        "queries": {
            "file": "queries.npy",
            "shape": [layout.query_count, layout.dimensions],
            "purpose": "held-out contiguous-azimuth evaluation views",
        },
        "common_layout": {
            "orientation": "rows_by_features",
            "feature_order": "pixel_row, pixel_column, RGB_channel",
            "dtype": "float32",
            "byte_order": "little-endian",
            "memory_order": "C",
            "value_range": [0.0, 1.0],
        },
    }
    if args.write_training_pool:
        matrices["training_pool"] = {
            "file": "training_pool.npy",
            "shape": [layout.pool_count, layout.dimensions],
            "purpose": "non-query views used to construct centroids",
        }

    return {
        "dataset_name": "COIL-100 contiguous-azimuth split",
        "format_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "matrices": matrices,
        "labels": {
            "reference_file": "reference_labels.npy",
            "query_file": "query_labels.npy",
            "training_pool_file": (
                "training_pool_labels.npy"
                if args.write_training_pool
                else None
            ),
            "dtype": "uint16",
            "definition": "zero-based physical object identity",
            "mapping": {
                str(label): object_id
                for label, object_id in enumerate(layout.object_ids)
            },
        },
        "source_images": {
            "count": len(layout.object_ids) * len(layout.all_angles),
            "objects": len(layout.object_ids),
            "views_per_object": len(layout.all_angles),
            "shape": [layout.height, layout.width, len(CHANNELS)],
            "format": "binary P6 PPM with three channel maxima",
            "source_sample_byte_order": "big-endian for 16-bit samples",
            "source_manifest_available": source_manifest is not None,
        },
        "split": {
            "unit": "view angle within each physical object",
            "policy": "same cyclic contiguous azimuth block held out per object",
            "angle_step_degrees": layout.all_angles[1] - layout.all_angles[0],
            "query_start_angle_degrees": args.query_start_angle,
            "query_view_count_per_object": len(layout.query_angles),
            "query_angles_degrees_in_row_order": list(layout.query_angles),
            "training_pool_view_count_per_object": len(
                layout.pool_angles
            ),
            "training_pool_angles_degrees": list(layout.pool_angles),
            "leakage_control": (
                "adjacent held-out views are not randomly interleaved with "
                "training views"
            ),
        },
        "reference_vector_construction": {
            "metadata_file": "reference_vectors.csv",
            "method": "arithmetic_mean_centroid",
            "count_per_object": 1,
            "source": "normalized views assigned role=training_pool",
            "pool_matrix_materialized": args.write_training_pool,
        },
        "features": {
            "file": "features.csv",
            "count": layout.dimensions,
            "channels": list(CHANNELS),
        },
        "samples": {
            "file": "samples.csv",
            "count": len(layout.object_ids) * len(layout.all_angles),
            "counts_by_role": {
                "training_pool": layout.pool_count,
                "query": layout.query_count,
            },
        },
        "transformations": [
            "decode P6 samples using their declared per-channel maxima",
            "normalize each RGB channel independently to [0,1]",
            "flatten images in pixel-row, pixel-column, RGB-channel order",
            "hold out a cyclic contiguous azimuth block for queries",
            "construct one arithmetic-mean centroid per object only from "
            "non-query views",
        ],
        "numpy_version": np.__version__,
        "generated_files": generated_files,
    }


def install_staged_outputs(
    staging_dir: Path,
    output_dir: Path,
    force: bool,
) -> None:
    staged_names = {
        path.name for path in staging_dir.iterdir() if path.is_file()
    }
    if force:
        for stale_name in MANAGED_OUTPUT_NAMES - staged_names:
            stale_path = output_dir / stale_name
            if stale_path.is_file():
                stale_path.unlink()
    for name in sorted(staged_names):
        (staging_dir / name).replace(output_dir / name)


def main() -> int:
    args = parse_args()
    try:
        layout = build_layout(args)
        input_dir = args.input_dir.expanduser().resolve()
        output_dir = args.output_dir.expanduser().resolve()
        if not input_dir.is_dir():
            raise RuntimeError(f"input directory does not exist: {input_dir}")
        inventory = collect_image_paths(
            input_dir / "images",
            layout.object_ids,
            layout.all_angles,
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(
            name
            for name in MANAGED_OUTPUT_NAMES
            if (output_dir / name).exists()
        )
        if existing and not args.force:
            print(
                f"Refusing to overwrite existing files in {output_dir}: "
                f"{', '.join(existing)}\nRun again with --force to replace them.",
                file=sys.stderr,
            )
            return 1

        print(
            f"Transforming {len(inventory):,} images: "
            f"{layout.pool_count:,} training-pool views and "
            f"{layout.query_count:,} held-out queries with "
            f"{layout.dimensions:,} dimensions.",
            flush=True,
        )
        with tempfile.TemporaryDirectory(
            prefix=".coil100-transform-",
            dir=output_dir,
        ) as temporary_directory:
            staging_dir = Path(temporary_directory)
            write_matrices(
                inventory,
                layout,
                staging_dir,
                args.write_training_pool,
            )
            write_labels(
                staging_dir / "reference_labels.npy",
                layout.object_ids,
                1,
            )
            write_labels(
                staging_dir / "query_labels.npy",
                layout.object_ids,
                len(layout.query_angles),
            )
            if args.write_training_pool:
                write_labels(
                    staging_dir / "training_pool_labels.npy",
                    layout.object_ids,
                    len(layout.pool_angles),
                )
            write_samples(staging_dir / "samples.csv", layout)
            write_reference_vectors(staging_dir / "reference_vectors.csv", layout)
            write_features(staging_dir / "features.csv", layout)

            source_manifest = read_source_manifest(input_dir)
            if source_manifest is None:
                with (staging_dir / "source_manifest.json").open(
                    "w",
                    encoding="utf-8",
                ) as output:
                    json.dump({"available": False}, output, indent=2)
                    output.write("\n")
            else:
                shutil.copyfile(
                    input_dir / "manifest.json",
                    staging_dir / "source_manifest.json",
                )

            metadata = build_dataset_metadata(
                layout,
                args,
                staging_dir,
                source_manifest,
            )
            with (staging_dir / "dataset.json").open(
                "w",
                encoding="utf-8",
            ) as output:
                json.dump(metadata, output, indent=2, sort_keys=True)
                output.write("\n")
            install_staged_outputs(staging_dir, output_dir, args.force)

    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(
        f"Saved {len(layout.object_ids):,} class centroids and "
        f"{layout.query_count:,} queries with {layout.dimensions:,} "
        f"dimensions to {output_dir}.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
