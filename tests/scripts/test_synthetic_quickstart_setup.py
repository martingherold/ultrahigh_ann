#!/usr/bin/env python3
"""Regression tests for the synthetic L2 quickstart setup."""

from __future__ import annotations

import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SETUP = PROJECT_ROOT / "experiments" / "synthetic" / "synthetic_quickstart_l2.tsv"


class SyntheticQuickstartSetupTest(unittest.TestCase):
    def test_contains_compact_comparison_grid(self) -> None:
        directives: dict[str, str] = {}
        runs: set[tuple[str, int, int | None, int]] = set()
        run_count = 0
        for line in SETUP.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if fields[0] in (
                "dataset",
                "json_output",
                "csv_output",
                "reference",
                "distance",
                "max_queries",
            ):
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
                    int(settings["projection_dimension"])
                    if "projection_dimension" in settings
                    else None,
                    int(settings["seed"]),
                )
            )
            run_count += 1

        self.assertEqual(directives["distance"], "l2")
        self.assertEqual(
            directives["dataset"],
            "../../data/synthetic/quickstart_v1",
        )
        self.assertEqual(directives["reference"], "exact")
        self.assertEqual(
            directives["json_output"],
            "../../results/raw/synthetic_quickstart_l2.json",
        )
        seeds = (7, 42, 2026)
        for repetition in (1, 2, 4, 8, 16, 32):
            for seed in seeds:
                self.assertIn(("flat", repetition, None, seed), runs)
                self.assertIn(("uniform", repetition, None, seed), runs)
        hierarchical_grid = {
            (4, 8),
            (4, 16),
            (8, 8),
            (8, 16),
            (8, 32),
            (16, 16),
            (16, 32),
        }
        self.assertEqual(
            {run for run in runs if run[0] == "hierarchical"},
            {
                ("hierarchical", repetition, projection, seed)
                for repetition, projection in hierarchical_grid
                for seed in seeds
            },
        )
        self.assertEqual(run_count, 57)
        self.assertEqual(len(runs), 57)


if __name__ == "__main__":
    unittest.main()
