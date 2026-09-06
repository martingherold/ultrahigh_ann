#!/usr/bin/env python3
"""Generate a deterministic nearest-neighbor quickstart dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError as error:
    print(
        "generate_synthetic_dataset.py requires NumPy.",
        file=sys.stderr,
    )
    raise SystemExit(2) from error


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "synthetic" / "quickstart_v1"
DEFAULT_INTERPOLATIONS = "0.30,0.45,0.49,0.497"
DIFFICULTY_NAMES = ("easy", "medium", "hard", "near_boundary")
HASH_BLOCK_SIZE = 1024 * 1024
MANAGED_NAMES = {
    "reference_vectors.npy",
    "reference_labels.npy",
    "queries.npy",
    "query_labels.npy",
    "queries.csv",
    "dataset.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate reference vectors and controlled-margin queries for a "
            "small, deterministic L2 benchmark."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"destination directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--reference-vectors",
        type=int,
        default=64,
        help="number of reference vectors (default: 64)",
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=512,
        help="number of held-out queries (default: 512)",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=16_384,
        help="ambient dimension (default: 16384)",
    )
    parser.add_argument(
        "--informative-dimensions",
        type=int,
        default=2048,
        help="coordinates with full reference-vector variation (default: 2048)",
    )
    parser.add_argument(
        "--background-scale",
        type=float,
        default=0.02,
        help="reference-vector scale outside informative coordinates (default: 0.02)",
    )
    parser.add_argument(
        "--query-noise",
        type=float,
        default=0.02,
        help="coordinate-scaled Gaussian query noise (default: 0.02)",
    )
    parser.add_argument(
        "--interpolations",
        default=DEFAULT_INTERPOLATIONS,
        help=(
            "comma-separated target-to-rival interpolation fractions; each "
            "must be in [0, 0.5) (default: 0.30,0.45,0.49,0.497)"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="dataset generation seed (default: 2026)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace files from an earlier synthetic generation",
    )
    return parser.parse_args()


def parse_interpolations(value: str) -> tuple[float, ...]:
    fields = [field.strip() for field in value.split(",")]
    if not fields or any(not field for field in fields):
        raise ValueError("--interpolations must be a nonempty comma-separated list")
    try:
        interpolations = tuple(float(field) for field in fields)
    except ValueError as error:
        raise ValueError("--interpolations contains a non-numeric value") from error
    if any(not math.isfinite(value) or value < 0.0 or value >= 0.5 for value in interpolations):
        raise ValueError("every interpolation must be finite and in [0, 0.5)")
    return interpolations


def validate_args(args: argparse.Namespace) -> tuple[float, ...]:
    if args.reference_vectors < 2:
        raise ValueError("--reference-vectors must be at least 2")
    if args.reference_vectors > np.iinfo(np.uint16).max + 1:
        raise ValueError("--reference-vectors exceeds the uint16 label capacity")
    if args.queries < 1:
        raise ValueError("--queries must be positive")
    if args.dimensions < 1:
        raise ValueError("--dimensions must be positive")
    if not 1 <= args.informative_dimensions <= args.dimensions:
        raise ValueError(
            "--informative-dimensions must be between 1 and --dimensions"
        )
    if not math.isfinite(args.background_scale) or args.background_scale < 0.0:
        raise ValueError("--background-scale must be finite and nonnegative")
    if not math.isfinite(args.query_noise) or args.query_noise < 0.0:
        raise ValueError("--query-noise must be finite and nonnegative")
    if args.seed < 0:
        raise ValueError("--seed must be nonnegative")
    return parse_interpolations(args.interpolations)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(HASH_BLOCK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def nearest_neighbors(
    queries: np.ndarray,
    reference_vectors: np.ndarray,
) -> np.ndarray:
    queries64 = np.asarray(queries, dtype=np.float64)
    reference_vectors64 = np.asarray(reference_vectors, dtype=np.float64)
    query_norms = np.einsum("ij,ij->i", queries64, queries64)
    representative_norms = np.einsum(
        "ij,ij->i",
        reference_vectors64,
        reference_vectors64,
    )
    distances = (
        query_norms[:, None]
        + representative_norms[None, :]
        - 2.0 * queries64 @ reference_vectors64.T
    )
    return np.argmin(distances, axis=1).astype(np.intp, copy=False)


def nearest_rivals(reference_vectors: np.ndarray) -> np.ndarray:
    reference_vectors64 = np.asarray(reference_vectors, dtype=np.float64)
    norms = np.einsum("ij,ij->i", reference_vectors64, reference_vectors64)
    distances = norms[:, None] + norms[None, :] - 2.0 * (
        reference_vectors64 @ reference_vectors64.T
    )
    np.fill_diagonal(distances, np.inf)
    return np.argmin(distances, axis=1).astype(np.intp, copy=False)


def generate_arrays(
    args: argparse.Namespace,
    interpolations: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]], np.ndarray]:
    random_engine = np.random.default_rng(args.seed)
    informative_indices = np.sort(
        random_engine.choice(
            args.dimensions,
            size=args.informative_dimensions,
            replace=False,
        )
    ).astype(np.intp, copy=False)
    scales = np.full(args.dimensions, args.background_scale, dtype=np.float32)
    scales[informative_indices] = np.float32(1.0)

    reference_vectors = random_engine.standard_normal(
        (args.reference_vectors, args.dimensions),
        dtype=np.float32,
    )
    reference_vectors *= scales[None, :]
    reference_vectors = np.asarray(reference_vectors, dtype="<f4", order="C")
    reference_labels = np.arange(args.reference_vectors, dtype="<u2")

    rivals_by_target = nearest_rivals(reference_vectors)
    targets = np.arange(args.queries, dtype=np.intp) % args.reference_vectors
    random_engine.shuffle(targets)
    difficulty_indices = np.arange(args.queries, dtype=np.intp) % len(interpolations)
    random_engine.shuffle(difficulty_indices)
    interpolation_values = np.asarray(interpolations, dtype=np.float32)[
        difficulty_indices
    ]
    rivals = rivals_by_target[targets]

    queries = (
        (np.float32(1.0) - interpolation_values[:, None])
        * reference_vectors[targets]
        + interpolation_values[:, None] * reference_vectors[rivals]
    )
    if args.query_noise != 0.0:
        noise = random_engine.standard_normal(
            (args.queries, args.dimensions),
            dtype=np.float32,
        )
        noise *= scales[None, :]
        queries += np.float32(args.query_noise) * noise
    queries = np.asarray(queries, dtype="<f4", order="C")

    repair_steps = np.zeros(args.queries, dtype=np.uint8)
    for _ in range(8):
        predictions = nearest_neighbors(queries, reference_vectors)
        incorrect = predictions != targets
        if not np.any(incorrect):
            break
        queries[incorrect] = np.float32(0.5) * (
            queries[incorrect] + reference_vectors[targets[incorrect]]
        )
        repair_steps[incorrect] += 1
    predictions = nearest_neighbors(queries, reference_vectors)
    if np.any(predictions != targets):
        raise RuntimeError("could not keep every synthetic query in its target cell")

    query_labels = np.asarray(targets, dtype="<u2")
    records: list[dict[str, object]] = []
    for row in range(args.queries):
        difficulty_index = int(difficulty_indices[row])
        difficulty = (
            DIFFICULTY_NAMES[difficulty_index]
            if len(interpolations) == len(DIFFICULTY_NAMES)
            else f"band_{difficulty_index}"
        )
        records.append(
            {
                "row_index": row,
                "target_label": int(targets[row]),
                "rival_label": int(rivals[row]),
                "difficulty": difficulty,
                "requested_interpolation": float(interpolation_values[row]),
                "effective_interpolation": float(
                    interpolation_values[row] / (2 ** int(repair_steps[row]))
                ),
                "repair_steps": int(repair_steps[row]),
            }
        )
    return (
        reference_vectors,
        reference_labels,
        queries,
        query_labels,
        records,
        informative_indices,
    )


def main() -> int:
    args = parse_args()
    try:
        interpolations = validate_args(args)
        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(
            name for name in MANAGED_NAMES if (output_dir / name).exists()
        )
        if existing and not args.force:
            print(
                f"Refusing to overwrite existing files in {output_dir}: "
                f"{', '.join(existing)}\nRun again with --force to replace them.",
                file=sys.stderr,
            )
            return 1

        (
            reference_vectors,
            reference_labels,
            queries,
            query_labels,
            query_records,
            informative_indices,
        ) = generate_arrays(args, interpolations)

        with tempfile.TemporaryDirectory(
            prefix=".synthetic-quickstart-",
            dir=output_dir,
        ) as temporary_directory:
            staging = Path(temporary_directory)
            np.save(
                staging / "reference_vectors.npy",
                reference_vectors,
                allow_pickle=False,
            )
            np.save(
                staging / "reference_labels.npy",
                reference_labels,
                allow_pickle=False,
            )
            np.save(staging / "queries.npy", queries, allow_pickle=False)
            np.save(
                staging / "query_labels.npy",
                query_labels,
                allow_pickle=False,
            )
            with (staging / "queries.csv").open(
                "w", encoding="utf-8", newline=""
            ) as output:
                writer = csv.DictWriter(
                    output,
                    fieldnames=tuple(query_records[0]),
                )
                writer.writeheader()
                writer.writerows(query_records)

            difficulty_counts: dict[str, int] = {}
            for record in query_records:
                difficulty = str(record["difficulty"])
                difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
            metadata = {
                "dataset_name": "Synthetic nearest-neighbor quickstart",
                "format_version": 2,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "purpose": (
                    "Illustrative smoke benchmark; not empirical evidence for "
                    "the theoretical or real-data claims."
                ),
                "distance": "l2",
                "seed": args.seed,
                "reference_vectors": {
                    "file": "reference_vectors.npy",
                    "labels_file": "reference_labels.npy",
                    "shape": list(reference_vectors.shape),
                    "dtype": "float32",
                    "construction": "coordinate-scaled independent Gaussian vectors",
                },
                "queries": {
                    "file": "queries.npy",
                    "labels_file": "query_labels.npy",
                    "metadata_file": "queries.csv",
                    "shape": list(queries.shape),
                    "dtype": "float32",
                    "construction": "target-to-nearest-rival interpolation plus scaled noise",
                    "difficulty_counts": difficulty_counts,
                    "exact_target_agreement": 1.0,
                },
                "model": {
                    "informative_dimension_count": args.informative_dimensions,
                    "informative_indices": informative_indices.tolist(),
                    "informative_scale": 1.0,
                    "background_scale": args.background_scale,
                    "query_noise": args.query_noise,
                    "interpolations": list(interpolations),
                },
                "numpy_version": np.__version__,
            }
            generated_files = {}
            for name in sorted(MANAGED_NAMES - {"dataset.json"}):
                generated_files[name] = {"sha256": sha256(staging / name)}
            metadata["generated_files"] = generated_files
            with (staging / "dataset.json").open(
                "w", encoding="utf-8"
            ) as output:
                json.dump(metadata, output, indent=2, sort_keys=True)
                output.write("\n")

            staged_names = {path.name for path in staging.iterdir() if path.is_file()}
            if args.force:
                for stale in MANAGED_NAMES - staged_names:
                    stale_path = output_dir / stale
                    if stale_path.is_file():
                        stale_path.unlink()
            for name in sorted(staged_names):
                (staging / name).replace(output_dir / name)

    except (OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    total_bytes = sum(
        (output_dir / name).stat().st_size
        for name in MANAGED_NAMES
        if (output_dir / name).is_file()
    )
    print(
        f"Saved {args.reference_vectors} reference vectors x {args.dimensions} "
        f"dimensions and {args.queries} queries to {output_dir} "
        f"({total_bytes / (1024 * 1024):.1f} MiB).",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
