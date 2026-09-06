#!/usr/bin/env python3
"""Regression tests for the deliberately small public experiment catalog."""

from __future__ import annotations

import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = PROJECT_ROOT / "experiments"
EXPECTED_SETUPS = {
    "synthetic/synthetic_quickstart_l2.tsv",
    "tcga_pancancer/cpu/tcga_pancancer_r33_l2_cpu_sparse_sweep.tsv",
    "tcga_pancancer/cpu/tcga_pancancer_r33_l2_cpu_uniform_vs_flat.tsv",
    "tcga_pancancer/cuda/tcga_pancancer_r7028_l2_cuda_crossover.tsv",
    "tcga_pancancer/cuda/tcga_pancancer_r33_l1_cuda_exact_vs_flat.tsv",
    "tcga_pancancer/cuda/tcga_pancancer_r33_l2_cuda_numerical_validation.tsv",
}


def active_lines(path: Path) -> list[list[str]]:
    return [
        line.split("\t")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]


class ExperimentCatalogTest(unittest.TestCase):
    def test_contains_exactly_the_six_canonical_setups(self) -> None:
        observed = {
            path.relative_to(EXPERIMENTS).as_posix()
            for path in EXPERIMENTS.rglob("*.tsv")
        }
        self.assertEqual(observed, EXPECTED_SETUPS)
        for relative in observed:
            lines = active_lines(EXPERIMENTS / relative)
            self.assertEqual(lines[0], ["ultrahigh_ann_benchmark_setup_v2"])

    def test_l1_cuda_setup_uses_the_cpu_reference_and_direct_queries(self) -> None:
        setup = (
            EXPERIMENTS
            / "tcga_pancancer"
            / "cuda"
            / "tcga_pancancer_r33_l1_cuda_exact_vs_flat.tsv"
        )
        lines = active_lines(setup)
        self.assertIn(["distance", "l1"], lines)
        self.assertIn(["reference", "exact_cpu"], lines)
        runs = [fields for fields in lines if fields[0] == "run"]
        self.assertEqual(runs[0][1:3], ["exact_cpu", "exact"])
        self.assertEqual(runs[1][1:3], ["exact_cuda_direct", "exact"])
        cuda_runs = [fields for fields in runs if "backend=cuda" in fields]
        self.assertTrue(cuda_runs)
        self.assertTrue(all("strategy=direct" in fields for fields in cuda_runs))

    def test_l2_numerical_setup_isolates_direct_and_gemm_exact_scans(self) -> None:
        setup = (
            EXPERIMENTS
            / "tcga_pancancer"
            / "cuda"
            / "tcga_pancancer_r33_l2_cuda_numerical_validation.tsv"
        )
        lines = active_lines(setup)
        runs = [fields for fields in lines if fields[0] == "run"]
        self.assertEqual(
            [fields[1] for fields in runs],
            ["exact_cpu", "exact_cuda_direct", "exact_cuda_gemm"],
        )
        self.assertEqual([fields[2] for fields in runs], ["exact"] * 3)
        self.assertIn("strategy=direct", runs[1])
        self.assertIn("strategy=gemm", runs[2])

    def test_large_pool_cuda_setup_is_sparse_paired_and_complete(self) -> None:
        setup = (
            EXPERIMENTS
            / "tcga_pancancer"
            / "cuda"
            / "tcga_pancancer_r7028_l2_cuda_crossover.tsv"
        )
        lines = active_lines(setup)
        self.assertIn(["representatives_file", "representative_pool.npy"], lines)
        self.assertIn(["reference", "exact_cuda_direct"], lines)
        self.assertIn(["batch_sizes", "128"], lines)
        self.assertIn(["warmups", "1"], lines)
        self.assertIn(["trials", "3"], lines)
        runs = [fields for fields in lines if fields[0] == "run"]
        self.assertEqual(len(runs), 30)
        self.assertEqual(
            [fields[1] for fields in runs[:2]],
            ["exact_cuda_direct", "exact_cuda_gemm"],
        )
        for index in ("exact", "flat", "hierarchical"):
            strategies = {
                parameter.split("=", 1)[1]
                for fields in runs
                if fields[2] == index
                for parameter in fields[3:]
                if parameter.startswith("strategy=")
            }
            self.assertEqual(strategies, {"direct", "gemm"})

        self.assertEqual(
            {
                int(next(
                    value.split("=", 1)[1]
                    for value in fields
                    if value.startswith("repetitions=")
                ))
                for fields in runs
                if fields[2] == "flat"
            },
            {1, 2, 4, 8, 16, 32, 48, 64},
        )
        hierarchy_runs = [
            fields for fields in runs if fields[2] == "hierarchical"
        ]
        self.assertEqual(
            {
                (
                    int(next(
                        value.split("=", 1)[1]
                        for value in fields
                        if value.startswith("repetitions=")
                    )),
                    int(next(
                        value.split("=", 1)[1]
                        for value in fields
                        if value.startswith("projection_dimension=")
                    )),
                )
                for fields in hierarchy_runs
            },
            {
                (4, 32),
                (4, 128),
                (16, 128),
                (16, 512),
                (64, 512),
                (128, 1536),
            },
        )
        self.assertEqual(
            {
                int(next(
                    value.split("=", 1)[1]
                    for value in fields
                    if value.startswith("seed=")
                ))
                for fields in runs
                if fields[2] in {"flat", "hierarchical"}
            },
            {42},
        )
        paired_strategies: dict[tuple[str, str, str, str], set[str]] = {}
        for fields in runs:
            if fields[2] not in {"flat", "hierarchical"}:
                continue
            repetitions = next(
                value for value in fields if value.startswith("repetitions=")
            )
            seed = next(value for value in fields if value.startswith("seed="))
            projection = next(
                (
                    value
                    for value in fields
                    if value.startswith("projection_dimension=")
                ),
                "projection_dimension=0",
            )
            strategy = next(
                value.split("=", 1)[1]
                for value in fields
                if value.startswith("strategy=")
            )
            paired_strategies.setdefault(
                (fields[2], repetitions, projection, seed), set()
            ).add(strategy)
        self.assertTrue(paired_strategies)
        self.assertTrue(
            all(
                strategies == {"direct", "gemm"}
                for strategies in paired_strategies.values()
            )
        )


if __name__ == "__main__":
    unittest.main()
