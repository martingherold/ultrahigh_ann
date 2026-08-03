#!/usr/bin/env python3
"""Validate the committed COIL-100 hierarchical sweep setups."""

from __future__ import annotations

import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Coil100HierarchicalSetupsTest(unittest.TestCase):
    def test_versioned_setups_contain_the_full_matched_grids(self) -> None:
        for distance in ("l1", "l2"):
            setup = (
                PROJECT_ROOT
                / "experiments"
                / f"coil100_{distance}_hierarchical_sweep.tsv"
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
            flat = [fields for fields in runs if fields[2] == "flat"]
            hierarchical = [
                fields for fields in runs if fields[2] == "hierarchical"
            ]
            self.assertEqual(len(runs), 210)
            self.assertEqual(len(flat), 30)
            self.assertEqual(len(hierarchical), 180)
            observed_projections = {
                int(fields[-1].partition("=")[2]) for fields in hierarchical
            }
            self.assertEqual(observed_projections, {4, 8, 16, 32, 64, 96})


if __name__ == "__main__":
    unittest.main()

