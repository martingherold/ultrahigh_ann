#!/usr/bin/env python3
"""Tests for the synthetic quickstart runner."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "benchmark" / "run_synthetic_quickstart.py"


class RunSyntheticQuickstartTest(unittest.TestCase):
    def test_dry_run_shows_reproducible_pipeline(self) -> None:
        result = subprocess.run(
            (sys.executable, str(SCRIPT), "--dry-run", "--figures"),
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("generate_synthetic_dataset.py", result.stdout)
        self.assertIn("nearest_neighbor_benchmark", result.stdout)
        self.assertIn("synthetic_quickstart_l2.tsv", result.stdout)
        self.assertIn("create_benchmark_figures.py", result.stdout)


if __name__ == "__main__":
    unittest.main()
