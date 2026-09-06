# Benchmark setup files

Version-two tab-separated setups drive the single
`nearest_representative_benchmark` executable. Every setup names its exact
reference explicitly and writes two outputs: a complete JSON report and a
trial-level CSV projection. Blank lines and comments beginning with `#` are
ignored.

The first non-comment line is:

```text
ultrahigh_ann_benchmark_setup_v2
```

Required global directives are:

```text
dataset<TAB>PATH
json_output<TAB>PATH
csv_output<TAB>PATH
reference<TAB>RUN_NAME
```

Optional globals select `distance` (`l1` or `l2`), `max_queries`, `warmups`,
`trials`, comma-separated `batch_sizes`, CUDA `device`, and `diagnostics`
(`none`, `selected_distances`, or `full_distance_table`). Dataset filenames
default to `representatives.npy`, `representative_labels.npy`, `queries.npy`,
and `query_labels.npy`; the corresponding `*_file` directives override them.
Relative paths are resolved from the setup file.

Approximate runs share one probability model. Select `probability_policy` as
`sequential`, `cpu_parallel`, `gpu_fp32`, `gpu_cublas_fp32`, or `load`.
The first three computation policies support both distances. For L1,
`gpu_fp32` directly reduces absolute differences. For L2, the cuBLAS policy
uses an FP32 Gram matrix for pairwise distances and the same CUDA
coordinate-maximum reduction as `gpu_fp32`; it requires an additional
`4 * representative_count^2` bytes of device memory. L1 rejects
`gpu_cublas_fp32` because L1 distance has no matrix-multiplication identity.
The `load` policy requires a `probabilities<TAB>PATH` directive.
Probability tools write UAP version-two files. In addition to the metric and
matrix shape, each file stores the SHA-256 of the exact representative-matrix
file used to construct it. Loading recomputes that digest and rejects stale
probabilities from a different same-shaped matrix. Older UAP versions are not
accepted.

Each run has this form:

```text
run<TAB>NAME<TAB>INDEX<TAB>KEY=VALUE...
```

`INDEX` is `exact`, `flat`, `uniform`, or `hierarchical`. Common parameters are
`backend=cpu|cuda`, `strategy=...`, `batch_sizes=...`, `warmups=N`, and
`trials=N`. Approximate runs require `repetitions=N` and `seed=N`; hierarchical
runs additionally require `projection_dimension=N`. CPU L2 supports
`sequential`, `parallel_queries`, `parallel_representatives`, and `automatic`.
CUDA L2 exact, flat, uniform, and hierarchical runs support `direct` and
`gemm`. CUDA L2 hierarchy gathers sampled coordinates on the host, transfers
the compact sampled matrix, and projects it on the device using a direct
binary32 reduction kernel or cuBLAS SGEMM. The projection stays on the device
for the corresponding direct or SGEMM scan; all stages are included in query
time. Transform construction remains a one-time host-side step and its values
are narrowed to binary32. Projected representatives are computed from that
narrowed transform with binary32 accumulation before upload.

CUDA L1 exact, flat, and uniform support `direct`; `gemm` is rejected because
the direct L1 distance has no equivalent matrix-multiplication identity.
Hierarchical CUDA L1 supports both strategies for its linear binary32 Cauchy
projection. Both then use a CUDA median-of-absolute-differences scan and
deterministic argmin. CPU L1 and CPU hierarchical runs require sequential
execution. CUDA index construction rejects Cauchy coefficients that overflow
during conversion or representative projections that overflow during binary32
accumulation.

For example:

```text
ultrahigh_ann_benchmark_setup_v2
dataset<TAB>../../data/processed/example
json_output<TAB>../../results/raw/example.json
csv_output<TAB>../../results/raw/example.csv
distance<TAB>l2
reference<TAB>exact_cpu
diagnostics<TAB>selected_distances
run<TAB>exact_cpu<TAB>exact<TAB>backend=cpu<TAB>strategy=sequential
run<TAB>flat_t64<TAB>flat<TAB>backend=cpu<TAB>strategy=automatic<TAB>repetitions=64<TAB>seed=42
```

Run or validate it with:

```sh
./build/nearest_representative_benchmark \
    --setup experiments/DATASET/SETUP.tsv
./build/nearest_representative_benchmark \
    --setup experiments/DATASET/SETUP.tsv --validate-only
```

Use `selected_distances` for large representative pools. It evaluates the
full-dimensional distance of only the exact and returned representatives.
`full_distance_table` additionally reports returned rank and nearest-neighbor
margin, but materializes every query/representative distance.

## Report contract

Every benchmark producer and consumer in this repository uses JSON report
schema version 5. The exact result is an ordinary entry in `runs`, identified by
`reference: true`; timing results live in each run's `measurements` array, one
entry per batch size. Preprocessing and diagnostic metadata live under
`sampling_probabilities` and `diagnostic_execution`. The report-level
`provenance` object records project version, source commit and dirty state,
compiler and build configuration, host and CUDA runtime metadata, normalized
invocation, and the setup digest. Dataset and probability sections contain
SHA-256 identities for every loaded artifact. Older report layouts are rejected
rather than adapted at read time.

Run and summarize a setup with:

```sh
python3 scripts/benchmark/run_benchmark_sweep.py \
    --setup experiments/DATASET/SETUP.tsv
```

Derived summary JSON uses the separate `summary_schema_version` field and
records `source_report_schema_version: 5`; it is not another benchmark-report
schema. `create_benchmark_figures.py` reads schema 5 directly. For reports with
more than one measurement batch size, select one explicitly with
`--batch-size N`.

Checkpointed runs can be combined into one schema-5 JSON/CSV pair before
plotting. The joiner verifies that provenance, dataset, settings,
probabilities, diagnostics, and exact results agree; it merges the exact timing
trials and rejects duplicate approximate run names. Its provenance identifies
and hashes every source report:

```sh
python3 scripts/reporting/join_benchmark_reports.py \
    --reports results/raw/sweep_000.json results/raw/sweep_001.json \
    --output results/raw/sweep_joined.json
```

The figure selectors are `margin` for non-optimal returns by nearest-neighbor
margin, `failure` for empirical approximation-violation curves, `pareto` for
runtime versus exact representative agreement, and `distance` for runtime
versus relative distance excess. The distance figure plots
`100 * (returned_distance / reference_distance - 1)` and accepts
`--distance-statistic mean|median|p95|p99|maximum`; `mean` is the default.
When a report contains multiple execution strategies, latency figures render
them as separate series rather than connecting direct and GEMM observations.
Filled, solid-line markers denote direct scans; hollow, dashed-line markers
denote GEMM scans. Marker shape and color continue to identify the index type.
When approximate CUDA runs use batching but a CPU reference has one fixed B1
measurement, that single reference measurement is retained as the baseline.
For a dense joined sweep, add `--compact`. It keeps every observation while
omitting per-point labels; hierarchical projection dimension is encoded by
color and repetitions by marker size.

For example, after running the r=7028 CUDA setup, create its mean-excess plot
at batch size 128 with:

```sh
python3 scripts/reporting/create_benchmark_figures.py \
    --report results/raw/tcga_pancancer_r7028_l2_cuda_crossover.json \
    --batch-size 128 \
    --figures distance pareto \
    --distance-statistic mean \
    --compact
```

## Included configurations

The checked-in catalog is intentionally limited to six experiments. It keeps
the two configurations underlying the existing public TCGA claims, adds three
focused CUDA experiments, and retains one deterministic smoke test:

1. `synthetic/synthetic_quickstart_l2.tsv` — deterministic smoke test.
2. `tcga_pancancer/cpu/tcga_pancancer_r33_l2_cpu_sparse_sweep.tsv` — headline trade-off.
3. `tcga_pancancer/cpu/tcga_pancancer_r33_l2_cpu_uniform_vs_flat.tsv` — matched baseline.
4. `tcga_pancancer/cuda/tcga_pancancer_r33_l1_cuda_exact_vs_flat.tsv` — L1 metric and CUDA coverage.
5. `tcga_pancancer/cuda/tcga_pancancer_r7028_l2_cuda_crossover.tsv` — large representative-pool CUDA crossover study.
6. `tcga_pancancer/cuda/tcga_pancancer_r33_l2_cuda_numerical_validation.tsv` — direct/GEMM arithmetic audit.

Superseded local grids may be retained beside them with a `.tsv.archive`
suffix. That suffix is Git-ignored, so archived exploratory grids cannot
accidentally expand the public experiment surface. Dataset preparation tools
for COIL-100, GSE2034, and the expanded representative sets remain under
`scripts/` even though their exploratory grids are no longer checked in.

Run any prepared setup directly:

```sh
./build/nearest_representative_benchmark \
    --setup experiments/DATASET/SETUP.tsv
```

The CPU experiment files use an explicit sequential exact reference. The CUDA
L1 and numerical-validation setups also use that CPU result as their reference,
making binary32 selection differences visible in exact-choice agreement and
distance-ratio fields.

The large-pool CUDA experiment
`tcga_pancancer/cuda/tcga_pancancer_r7028_l2_cuda_crossover.tsv`
uses the 7,028 training representatives and all 3,013 held-back queries. At a
fixed batch size of 128 it compares direct and unrefined-GEMM scans for exact,
flat, and hierarchical indexes. Flat runs use
`T={1,2,4,8,16,32,48,64}`. The hierarchy follows the sparse crossover
staircase `(T,p)={(4,32),(4,128),(16,128),(16,512),(64,512),(128,1536)}`.
The low end tests the observed direct-scan advantage at lower fidelity; the
high end tests where flat search takes over. Keeping the GEMM counterpart for
every point makes the strategy-dependent result visible instead of presenting
the direct result as universal. Approximate runs use the fixed seed `42`, one
warmup, and three timed trials. This makes the 30-run setup a deterministic,
reviewer-runnable demonstration rather than a sampling-variance study. The two
exact runs use one trial. Scalable `selected_distances` diagnostics avoid a
full distance table.

`exact_cuda_direct` is the reference for this large-pool artifact; it is an FP32
device reference, not a claim of equivalence to the CPU binary64 exact index.
Run it with:

```sh
./build-cuda/nearest_representative_benchmark \
    --setup experiments/tcga_pancancer/cuda/tcga_pancancer_r7028_l2_cuda_crossover.tsv
```

Its JSON report is the canonical record. Its CSV contains one row per timing
trial and repeats scalar quality, memory, backend, strategy, batch-size, and
device fields for convenient analysis. The approximation violation columns use
`epsilon = 0.001, 0.005, 0.01, 0.02, 0.05, 0.10`.
