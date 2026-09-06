#!/usr/bin/env python3
"""Regression tests for the TCGA L2 uniform baseline setup."""

from __future__ import annotations

import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SETUPS = (
    (
        PROJECT_ROOT
        / "experiments"
        / "tcga_pancancer"
        / "cpu"
        / "tcga_pancancer_r33_l2_cpu_uniform_vs_flat.tsv",
        "../../../data/processed/tcga_pancancer_v1",
        "../../../results/raw/tcga_pancancer_r33_l2_cpu_uniform_vs_flat.json",
        {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048},
    ),
)


class TcgaUniformVsFlatSetupTest(unittest.TestCase):
    def test_contains_matched_parameter_grid(self) -> None:
        seeds = {7, 20, 42, 1312, 2026}
        for setup, expected_dataset, expected_output, repetitions in SETUPS:
            with self.subTest(setup=setup.name):
                expected = {
                    (method, repetition, seed)
                    for method in ("flat", "uniform")
                    for repetition in repetitions
                    for seed in seeds
                }
                runs: set[tuple[str, int, int]] = set()
                directives: dict[str, str] = {}
                run_count = 0
                for line in setup.read_text(encoding="utf-8").splitlines():
                    if not line or line.startswith("#"):
                        continue
                    fields = line.split("\t")
                    if fields[0] in ("dataset", "json_output"):
                        directives[fields[0]] = fields[1]
                    if fields[0] != "run":
                        continue
                    if fields[2] == "exact":
                        continue
                    settings = dict(field.split("=", 1) for field in fields[3:])
                    runs.add(
                        (
                            fields[2],
                            int(settings["repetitions"]),
                            int(settings["seed"]),
                        )
                    )
                    run_count += 1

                self.assertEqual(directives["dataset"], expected_dataset)
                self.assertEqual(directives["json_output"], expected_output)
                self.assertEqual(run_count, len(expected))
                self.assertEqual(runs, expected)


if __name__ == "__main__":
    unittest.main()
