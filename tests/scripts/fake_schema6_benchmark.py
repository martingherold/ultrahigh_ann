#!/usr/bin/env python3
"""Small schema-6 benchmark stand-in used by subprocess integration tests."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


setup_path = Path(sys.argv[sys.argv.index("--setup") + 1]).resolve()
directives: dict[str, str] = {}
configured_runs: list[tuple[str, str, dict[str, str]]] = []
for raw_line in setup_path.read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#") or line.startswith("ultrahigh_ann_"):
        continue
    fields = raw_line.split("\t")
    if fields[0] == "run":
        configured_runs.append(
            (
                fields[1],
                fields[2],
                dict(field.split("=", 1) for field in fields[3:]),
            )
        )
    else:
        directives[fields[0]] = fields[1]


def approximation(agreement: float) -> dict[str, object]:
    optimal = round(agreement * 10)
    non_optimal = 10 - optimal
    distribution = {
        "count": 10,
        "mean": 1.01,
        "median": 1.0,
        "percentile_95": 1.05,
        "percentile_99": 1.05,
        "maximum": 1.05,
    }
    return {
        "distance_optimal_count": optimal,
        "distance_optimal_rate": agreement,
        "non_optimal_count": non_optimal,
        "non_optimal_rate": 1.0 - agreement,
        "reference_improvement_count": 0,
        "distance_ratio": distribution,
        "zero_optimum_query_count": 0,
        "zero_optimum_non_optimal_count": 0,
        "non_optimal_ratio_count": non_optimal,
        "non_optimal_distance_ratio_mean": 1.05 if non_optimal else None,
        "non_optimal_relative_excess_mean": 0.05 if non_optimal else None,
        "approximation_guarantee_failures": [
            {
                "epsilon": epsilon,
                "violation_count": non_optimal,
                "violation_rate": 1.0 - agreement,
            }
            for epsilon in (0.001, 0.005, 0.01, 0.02, 0.05, 0.10)
        ],
        "returned_neighbor_rank": distribution,
        "by_multiplicative_margin": [],
    }


def result_run(
    name: str,
    index: str,
    parameters: dict[str, str],
) -> dict[str, object]:
    reference = name == directives["reference"]
    repetitions = int(parameters.get("repetitions", 0))
    seed = int(parameters.get("seed", 0))
    projection = int(parameters.get("projection_dimension", 0))
    if reference:
        query_us, accuracy, agreement, build_ms = 1000.0, 0.8, 1.0, 2.0
        index_bytes, unique, multiplicity = 10_000, 100, 100
    elif index == "hierarchical":
        query_us = 100.0 + 2.0 * projection
        accuracy = 0.85 + projection / 1000.0
        agreement = 0.90 + projection / 200.0
        build_ms = 3.0 + projection
        index_bytes, unique, multiplicity = (
            500 + 50 * projection,
            repetitions // 2,
            repetitions * 4,
        )
    else:
        query_us, accuracy, build_ms = 200.0, 0.8, 3.0
        agreement = 0.90 + repetitions / 10_000 + seed / 100_000
        index_bytes, unique, multiplicity = (
            1_000,
            repetitions // 2,
            repetitions * 4,
        )
    elapsed_ms = query_us / 100.0
    measurement = {
        "batch_size": 1,
        "workspace_payload_bytes": 8 + projection,
        "trial_ms": [elapsed_ms],
        "median_ms": elapsed_ms,
        "median_microseconds_per_query": query_us,
        "median_queries_per_second": 1_000_000.0 / query_us,
        "exact_neighbor_agreement_count": round(agreement * 10),
        "exact_neighbor_agreement": agreement,
        "correct": round(accuracy * 10),
        "accuracy": accuracy,
        "label_agreement_with_exact": agreement,
        "approximation": None if reference else approximation(agreement),
    }
    backend = parameters.get("backend", "cpu")
    return {
        "name": name,
        "index": index,
        "backend": backend,
        "strategy": parameters.get(
            "strategy", "gemm" if backend == "cuda" else "sequential"
        ),
        "reference": reference,
        "repetitions": repetitions,
        "seed": seed,
        "projection_dimension": projection,
        "warmups": int(parameters.get("warmups", directives.get("warmups", 0))),
        "trials": int(parameters.get("trials", directives.get("trials", 1))),
        "build_ms": build_ms,
        "space": {
            "index_payload_bytes": index_bytes,
            "query_workspace_payload_bytes": 0,
            "unique_coordinates": unique,
            "sampled_multiplicity": multiplicity,
        },
        "device": None,
        "measurements": [measurement],
    }


configured_output = Path(directives["json_output"])
output_path = (
    configured_output
    if configured_output.is_absolute()
    else (setup_path.parent / configured_output).resolve()
)
cuda_enabled = any(
    parameters.get("backend", "cpu") == "cuda"
    for _, _, parameters in configured_runs
)
reference_vectors_sha256 = "a" * 64
report = {
    "schema_version": 6,
    "generated_at_utc": "2026-01-01T00:00:00Z",
    "setup_file": str(setup_path),
    "provenance": {
        "source": {
            "project_version": "0.2.0",
            "git": {"available": False, "commit": None, "dirty": None},
        },
        "build": {
            "cmake_version": "3.29.3",
            "build_type": "Release",
            "cxx": {
                "standard": 20,
                "compiler_id": "fixture",
                "compiler_version": "1.0",
                "flags": "",
            },
            "cuda": {
                "enabled": cuda_enabled,
                "compiler_id": "NVIDIA" if cuda_enabled else None,
                "compiler_version": "13.3" if cuda_enabled else None,
                "architectures": "75" if cuda_enabled else None,
                "flags": "-O3" if cuda_enabled else None,
            },
        },
        "host": {
            "operating_system": {
                "name": "fixture OS",
                "version": "1",
                "architecture": "x86_64",
            },
            "processor_model": "fixture CPU",
            "physical_memory_bytes": 1_000_000,
        },
        "cuda_runtime": (
            {
                "device": 0,
                "device_name": "fixture GPU",
                "compute_capability": "7.5",
                "total_global_memory_bytes": 4_000_000_000,
                "compiled_runtime": {"encoded": 13030, "version": "13.3"},
                "runtime": {"encoded": 13030, "version": "13.3"},
                "driver": {"encoded": 13030, "version": "13.3"},
            }
            if cuda_enabled
            else None
        ),
        "invocation": {
            "arguments": ["--setup", str(setup_path)],
            "setup_sha256": hashlib.sha256(setup_path.read_bytes()).hexdigest(),
        },
    },
    "outputs": {
        "json": str(output_path),
        "csv": directives["csv_output"],
    },
    "dataset": {
        "directory": "dataset",
        "reference_vectors_file": "dataset/reference_vectors.npy",
        "reference_labels_file": "dataset/reference_labels.npy",
        "queries_file": "dataset/queries.npy",
        "query_labels_file": "dataset/query_labels.npy",
        "reference_vectors_sha256": reference_vectors_sha256,
        "reference_labels_sha256": "b" * 64,
        "queries_sha256": "c" * 64,
        "query_labels_sha256": "d" * 64,
        "reference_vector_count": 33,
        "query_count_available": 10,
        "query_count_run": 10,
        "dimension": 100,
        "labels_available": True,
        "hash_ms": 0.5,
        "load_ms": 1.0,
    },
    "settings": {
        "distance": directives.get("distance", "l2"),
        "reference_run": directives["reference"],
        "max_queries": int(directives.get("max_queries", 0)),
        "device": 0,
        "diagnostics": directives.get("diagnostics", "selected_distances"),
    },
    "sampling_probabilities": {
        "required": True,
        "policy": "sequential",
        "source_file": None,
        "source_sha256": None,
        "reference_vectors_sha256": reference_vectors_sha256,
        "build_ms": 12.5,
        "load_ms": 0.0,
        "coordinate_count": 100,
        "sampling_mass": 4.0,
        "gpu": None,
    },
    "diagnostic_execution": {
        "build_ms": 1.0,
        "evaluation_ms": 1.0,
        "payload_bytes": 800,
        "query_geometry": None,
    },
    "runs": [
        result_run(name, index, parameters)
        for name, index, parameters in configured_runs
    ],
}
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(report), encoding="utf-8")
