# Scripts

Run these utilities from the repository root. They are grouped by the stage of
the reproducibility workflow:

- `data/` downloads, validates, transforms, and expands datasets.
- `benchmark/` executes benchmark setups and prints aggregate summaries.
- `reporting/` joins checkpoint reports and creates figures.

For the shortest end-to-end example, run:

```sh
python3 scripts/benchmark/run_synthetic_quickstart.py
```

Dataset-specific commands are documented in
[`data/README.md`](../data/README.md), and benchmark/reporting commands are documented in
[`experiments/README.md`](../experiments/README.md).
