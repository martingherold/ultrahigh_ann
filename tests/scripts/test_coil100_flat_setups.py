#!/usr/bin/env python3
"""Validate the committed COIL-100 flat sweep setups."""

from __future__ import annotations

import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Coil100FlatSetupsTest(unittest.TestCase):
    def test_versioned_setups_contain_the_full_flat_grids(self) -> None:
        expected = {
            (repetitions, seed)
            for repetitions in (32, 64, 128, 256, 512, 1024)
            for seed in (7, 20, 42, 1312, 2026)
        }
        for distance in ("l1", "l2"):
            setup = (
                PROJECT_ROOT
                / "experiments"
                / f"coil100_{distance}_flat_sweep.tsv"
            )
            directives = [
                line.split("\t")
                for line in setup.read_text(encoding="utf-8").splitlines()
                if line and not line.startswith("#")
            ]
            self.assertIn(["distance", distance], directives)
            self.assertIn(
                ["dataset", "../data/processed/coil100_v1"],
                directives,
            )
            runs = [fields for fields in directives if fields[0] == "run"]
            self.assertEqual(len(runs), 30)
            observed: set[tuple[int, int]] = set()
            for fields in runs:
                self.assertEqual(fields[2], "flat")
                settings = dict(field.split("=", 1) for field in fields[3:])
                observed.add(
                    (int(settings["repetitions"]), int(settings["seed"]))
                )
            self.assertEqual(observed, expected)


if __name__ == "__main__":
    unittest.main()

