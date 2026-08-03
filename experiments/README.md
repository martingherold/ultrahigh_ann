# Benchmark setup files

Versioned `.tsv` setup files let one C++ benchmark process evaluate many
approximate-index configurations against one exact baseline. The dataset is
loaded once, exact queries are executed once, and their predictions remain in
memory while every named approximate run is evaluated. The metric-specific,
data-dependent importance probabilities are also computed once and reused by
all repetitions, seeds, flat indexes, and hierarchical projection dimensions.
Per-run `build_ms` therefore measures only sampling and index construction;
the shared probability time and payload are recorded under
`shared_preprocessing.sampling_probabilities` in the JSON report. That object
also records the exact sampling mass `sampling_mass`, its ratio to `r`, and
the uniform coordinate probability that preserves the same mass.

## Version 1 format

Blank lines and lines whose first non-whitespace character is `#` are ignored.
The first remaining line must be:

```text
ultrahigh_ann_benchmark_setup_v1
```

Every subsequent line is tab-separated. Global directives are:

```text
dataset<TAB>PATH
output<TAB>PATH
distance<TAB>l1|l2
max_queries<TAB>N
```

`dataset` and `output` are required exactly once. Relative paths are resolved
against the directory containing the setup file. `max_queries` is optional and
defaults to zero, meaning all available queries. `distance` is optional and
defaults to `l1` for compatibility with existing version-one files. It selects
the exact, importance-sampled flat, uniform, and hierarchical metric
implementations together.

Approximate runs use:

```text
run<TAB>NAME<TAB>METHOD<TAB>KEY=VALUE<TAB>KEY=VALUE...
```

Names must be unique and may contain letters, digits, `.`, `_`, and `-`.
Supported method schemas are:

```text
run<TAB>NAME<TAB>flat<TAB>repetitions=N<TAB>seed=N
run<TAB>NAME<TAB>uniform<TAB>repetitions=N<TAB>seed=N
run<TAB>NAME<TAB>hierarchical<TAB>repetitions=N<TAB>seed=N<TAB>projection_dimension=N
```

`flat` uses the data-dependent importance probabilities. `uniform` replaces
them by the constant probability `S(C)/d`, so both methods have expected
sampled multiplicity `T*S(C)` at a matched `T`. Both scan weighted sampled
coordinates directly; only `hierarchical` applies Cauchy/JL projection.

Unknown methods and parameters are rejected rather than ignored. New index
types can extend the `METHOD` and method-specific key set without changing the
global format.

Run a setup with:

```sh
./build/nearest_representative_benchmark \
    --setup experiments/tcga_pancancer_l1_flat_sweep.tsv
./build/nearest_representative_benchmark \
    --setup experiments/tcga_pancancer_l2_flat_sweep.tsv
```

## Synthetic quickstart

`synthetic_quickstart_l2.tsv` is a compact, three-seed smoke benchmark over
generated data. It compares flat and mass-matched uniform sampling for
`T = 1, 2, 4, 8, 16, 32` and includes seven small hierarchical configurations
per seed.
Generate and run it with:

```sh
python3 scripts/run_synthetic_quickstart.py
```

Use `--figures` to create the Pareto plot after the run. This setup is intended
to make the software easy to inspect; it is not part of the publication-level
experimental evidence.

## Pan-Cancer datasets with several representatives per cancer

The base transform can retain the training-only representative pool. The
expansion script then clusters each cancer project independently and computes
each final centroid over all 60,660 genes:

```sh
python3 scripts/transform_tcga_pancancer.py --write-representative-pool --force
python3 scripts/expand_tcga_pancancer_representatives.py \
    --representatives-per-class 4
```

Queries are never used to select genes, form clusters, or compute centroids.
The generated `representative_labels.npy` maps all 132 representative rows
back to the 33 cancer labels. Reports consequently contain both
`agreement_with_exact` (same subcentroid) and `label_agreement_with_exact`
(same cancer); the latter is the appropriate fidelity measure for this setup.

The compact, three-seed matched L2 pilot can be run and summarized with:

```sh
python3 scripts/run_benchmark_sweep.py \
    --setup experiments/tcga_pancancer_k4_l2_hierarchical_pilot.tsv
```

It tests `T = 128, 256, 512` and `p = 16, 32, 64, 96, 128`, with a matched
flat run for each `(T, seed)` pair. Its raw report is
`results/raw/tcga_pancancer_k4_l2_hierarchical_pilot.json`.

The smaller `tcga_pancancer_k4_l2_high_p_followup.tsv` setup holds `T = 128`
fixed and tests `p = 160, 192, 256` over the same three seeds. It locates the
point where projection fidelity improves only after hierarchy approaches or
crosses below matched-flat query speed.

The corresponding `k10` setups use ten training-only subcentroids per cancer
and therefore `r = 330`. `tcga_pancancer_k10_l2_hierarchical_pilot.tsv`
repeats the k4 grid directly. The two follow-ups hold `T = 128` fixed and span
`p = 160, 192, 256, 320, 384, 448, 512, 640`, locating both the query-time
crossover and the remaining fidelity gap. Generate the dataset with:

```sh
python3 scripts/expand_tcga_pancancer_representatives.py \
    --representatives-per-class 10
```

Those 78 unique exploratory configurations are consolidated in
`tcga_pancancer_k10_l2_combined_sweep.tsv`. The consolidated sweep additionally
tests `T = 8, 16, 32` with `p = 16, 32, 64, 96, 128` over all three seeds to
cover the low-latency frontier. Flat-only controls at
`T = 1, 2, 3, 4, 5, 6, 7` test whether flat search also dominates the fastest
hierarchical points. The setup contains 153 runs in total. Run exact search and
probability construction once for the entire grid with:

```sh
python3 scripts/run_benchmark_sweep.py \
    --setup experiments/tcga_pancancer_k10_l2_combined_sweep.tsv
```

## Pan-Cancer L2 uniform-versus-importance baselines

The `tcga_pancancer_l2_uniform_vs_flat.tsv` and
`tcga_pancancer_k10_l2_uniform_vs_flat.tsv` setups compare mass-matched uniform
sampling with the direct importance-sampled flat index at `r = 33` and
`r = 330`, respectively. The `r = 33` experiment tests
`T = 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048`; the `r = 330`
experiment stops at `T = 512`. Both use seeds `7, 20, 42, 1312, 2026`, giving
60 and 50 matched pairs, respectively. Exact search and the L2 importance
probabilities used to determine `S(C)` are each computed once per dataset. Run
them with:

```sh
python3 scripts/run_benchmark_sweep.py \
    --setup experiments/tcga_pancancer_l2_uniform_vs_flat.tsv
python3 scripts/run_benchmark_sweep.py \
    --setup experiments/tcga_pancancer_k10_l2_uniform_vs_flat.tsv
```

The raw reports are written to `results/raw/tcga_pancancer_l2_uniform_vs_flat.json`
and `results/raw/tcga_pancancer_k10_l2_uniform_vs_flat.json`. Flat summaries
use the standard `_summary.json`, `_runs.csv`, and `_aggregates.csv` filenames;
uniform summaries use `_uniform_summary.json`, `_uniform_runs.csv`, and
`_uniform_aggregates.csv`.

For example, create the `r = 33` flat-versus-uniform Pareto plot with:

```sh
python3 scripts/create_benchmark_figures.py \
    --report results/raw/tcga_pancancer_l2_uniform_vs_flat.json \
    --figures pareto
```

For approximation or margin curves, use `--line-method flat-uniform` and
select a manageable subset with `--repetitions`.

## Pan-Cancer L2 hierarchical sweep

The matched hierarchy experiment is defined by
`tcga_pancancer_l2_hierarchical_sweep.tsv`. It evaluates the 33-representative
Pan-Cancer dataset using:

- flat repetitions
  `T = 1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128, 256, 512, 1024`;
- hierarchical repetitions `T = 8, 16, 32, 64, 128, 256, 512, 1024`;
- projection dimensions `p = 1, 2, 4, 8, 16, 32, 48, 64, 96, 128`;
- seeds `7, 20, 42, 1312, 2026`;
- one flat control for every `(T, seed)` pair.

This gives 400 hierarchical runs, 40 matched flat controls, and 35 additional
flat-only low-latency controls. Exact L2 search is still executed only once. At
`r = 33`, `p = 32` is close to the arithmetic
crossover with flat comparison. The higher dimensions deliberately extend past
that point to establish whether acceptable projection fidelity appears only
after hierarchy has lost its speed advantage.

Run the experiment and generate matched comparison CSV/JSON summaries with:

```sh
python3 scripts/run_benchmark_sweep.py \
    --setup experiments/tcga_pancancer_l2_hierarchical_sweep.tsv
```

If a report from the earlier `p <= 32` grid already exists, regenerate it with:

```sh
python3 scripts/run_benchmark_sweep.py \
    --setup experiments/tcga_pancancer_l2_hierarchical_sweep.tsv \
    --force
```

The raw batch report is written to
`results/raw/tcga_pancancer_l2_hierarchical_sweep.json`. The runner also writes
`_summary.json`, `_comparisons.csv`, and `_aggregates.csv` files beside it. It
exports every flat control, including controls without a hierarchical match, to
the additional `_flat_summary.json`, `_flat_runs.csv`, and
`_flat_aggregates.csv` files. A
hierarchical configuration is useful when `hierarchical_speedup_vs_flat` is
above one while `hierarchical_agreement_delta_vs_flat` remains acceptably close
to zero. Comparing only classification accuracy can be misleading because an
approximate method can occasionally disagree with exact search and still happen
to predict the ground-truth label.

To run the same setup directly without generating the secondary summaries:

```sh
./build/nearest_representative_benchmark \
    --setup experiments/tcga_pancancer_l2_hierarchical_sweep.tsv
```

## COIL-100 L1 and L2 hierarchical sweeps

COIL-100 has 100 representatives, so hierarchy has substantially more room to
reduce the final representative-comparison dimension. The L1 and L2 setup files
use the same matched design:

- repetitions `T = 32, 64, 128, 256, 512, 1024`;
- projection dimensions `p = 4, 8, 16, 32, 64, 96`;
- seeds `7, 20, 42, 1312, 2026`;
- one flat control for every `(T, seed)` pair.

Each metric therefore has 180 hierarchical runs and 30 flat controls. Its exact
search is computed once. Here `p = 96` acts as the near-full control for
`r = 100`; the smaller projections test whether hierarchy improves query time
without losing too much agreement with the matched flat index.

After downloading and transforming COIL-100, run both metrics with:

```sh
python3 scripts/run_benchmark_sweep.py \
    --setup experiments/coil100_l1_hierarchical_sweep.tsv
python3 scripts/run_benchmark_sweep.py \
    --setup experiments/coil100_l2_hierarchical_sweep.tsv
```

The raw reports are `coil100_l1_hierarchical_sweep.json` and
`coil100_l2_hierarchical_sweep.json` under `results/raw/`. Each receives the
same `_summary.json`, `_comparisons.csv`, `_aggregates.csv`, and flat-only
`_flat_summary.json`, `_flat_runs.csv`, and `_flat_aggregates.csv` companion
files as the Pan-Cancer hierarchy experiment.

## COIL-100 flat sweeps

Dedicated flat-only setups avoid constructing every hierarchical projection
when the goal is to establish the sampling curve. Both L1 and L2 use
`T = 32, 64, 128, 256, 512, 1024` with the same five seeds, giving 30 flat runs
per metric and one exact execution per metric.

Run both with:

```sh
python3 scripts/run_benchmark_sweep.py \
    --setup experiments/coil100_l1_flat_sweep.tsv
python3 scripts/run_benchmark_sweep.py \
    --setup experiments/coil100_l2_flat_sweep.tsv
```

The raw reports are `coil100_l1_flat_sweep.json` and
`coil100_l2_flat_sweep.json` under `results/raw/`. Each gets `_summary.json`,
`_runs.csv`, and `_aggregates.csv` companions.

Batch and single-run reports use JSON schema version 4. Batch reports contain
one top-level `exact` result and a `runs` array with each name, method,
settings, and result. In either mode, the JSON settings record the selected
distance metric. When the benchmark is launched from the repository root, as
in the commands in this repository, dataset and setup paths inside it are
recorded relative to that root so reports remain portable. Paths outside the
working directory remain absolute.

Schema version 4 builds one reusable exact query/representative distance table
after all timed queries. Its construction and evaluation costs are recorded
under `diagnostics` and are explicitly excluded from `query_total_ms`. The
shared `query_geometry` section reports zero-distance and tied optima together
with the finite multiplicative-margin distribution and the fixed margin
buckets `[1,1.01)`, `[1.01,1.05)`, `[1.05,1.10)`, `[1.10,1.25)`, and
`[1.25,infinity)`.

Every approximate result contains an `approximation` object with:

- tie-aware exact representative agreement and the non-optimal rate;
- mean, median, 95th percentile, 99th percentile, and maximum of
  `returned_distance / exact_distance`;
- the mean ratio and mean relative excess conditional on a non-optimal return;
- empirical violation rates for approximation factors `1.01`, `1.05`, `1.10`,
  and `1.20`;
- returned-representative rank statistics, using competition rank so all exact
distance ties have rank one;
- non-optimal rates and mean ratios within every multiplicative-margin bucket.

Ratio statistics exclude queries whose exact distance is zero, because their
ratio is undefined. Their total and non-optimal counts are reported separately,
and a zero-distance miss counts as a violation for every finite approximation
factor. `agreement_with_exact` is distance-based and accepts any optimal tie;
`agreement_with_exact_index_choice` preserves the stricter comparison with the
specific row selected by the exact index. The sweep runner exports scalar
approximation metrics to their per-run and aggregate summaries and copy the
shared query-geometry diagnostics into the summary JSON.

## Publication figures

Install the optional plotting dependency and create all three figures from a
schema-version 4 report with:

```sh
python3 -m pip install -r requirements-figures.txt
python3 scripts/create_benchmark_figures.py \
    --report results/raw/tcga_pancancer_l2_hierarchical_sweep.json
```

The command writes PDF and 300-DPI PNG versions under `results/plots/`:

- `_margin_error` shows non-optimal return rate by multiplicative-margin
  bucket, with the query distribution in the background;
- `_approximation_failures` shows empirical
  `Pr[returned_distance / exact_distance > 1 + epsilon]`;
- `_latency_fidelity_pareto` compares flat, hierarchical, and exact search and
  marks the observed query-time/agreement frontier.

Points and lines are means over runs with the same method parameters. The
shaded line-plot bands span the observed seed minimum and maximum. The default
line figures show the flat configurations, while the Pareto figure always
shows every configuration. To inspect selected hierarchy curves without
overloading the plots, filter both `T` and `p`, for example:

```sh
python3 scripts/create_benchmark_figures.py \
    --report results/raw/tcga_pancancer_k10_l2_combined_sweep.json \
    --line-method hierarchical \
    --repetitions 128 \
    --projection-dimensions 64 128 256 320 448 \
    --prefix tcga_pancancer_k10_l2_hierarchy
```

Use `--figures margin`, `--figures failure`, or `--figures pareto` to generate
individual figures. `--formats pdf png svg`, `--output-dir`, and `--dpi` control
the artifacts without modifying the source report.
