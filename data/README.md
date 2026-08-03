# Data

This directory contains local inputs and generated datasets. Large data files
are intentionally excluded from Git.

```text
raw/          Original, unmodified dataset downloads
processed/    Normalized or converted datasets
splits/       Version-controlled train/test split definitions
synthetic/    Generated benchmark instances
```

Keep dataset download and preparation code in `scripts/`. Record the source
URL, download date, checksum, normalization, and generation parameters for
each experiment.

Small fixtures needed by automated tests should go in `tests/data/` and may be
committed when their licenses permit it.

## Synthetic quickstart

Generate the small, deterministic representative-query dataset used by the
recruiter-facing quickstart with:

```sh
python3 scripts/generate_synthetic_dataset.py
```

The default creates 64 representatives and 512 queries in 16,384 dimensions
under `data/synthetic/quickstart_v1/`. Representative variation is concentrated
in 2,048 informative coordinates with low-variation background coordinates.
Queries interpolate between a target and its nearest rival at four controlled
difficulty levels and are verified against exact L2 search. No training data is
created or required. The generated directory is excluded from Git.

## TCGA Kidney Cancers

Fetch the GDC-backed matrices published by UCSC Xena:

```sh
python3 scripts/download_tcga.py
```

UCI lists dataset 892 as an external dataset, so it cannot be imported with
the generated `fetch_ucirepo(id=892)` example on the UCI page. The downloader
instead retrieves the three matching GDC STAR-FPKM matrices for KICH, KIRC,
and KIRP directly from the GDC-backed Xena hub. Together they contain 1,024
samples and 60,660 genes.

The downloaded matrices store genes in rows and samples in columns, with
values represented as `log2(FPKM+1)`. The script also writes `targets.csv` and
a checksum/provenance `manifest.json` to `data/raw/tcga/`. The total download
is approximately 273 MB. Generated files are excluded from Git. Use `--force`
to replace a previous download or `--output-dir PATH` to select another
destination.

Transform the matrices into sample-major, row-contiguous `float32`
representatives and queries for the ANN implementation:

```sh
python3 scripts/transform_tcga.py
```

By default, the transform retains primary solid tumors (TCGA sample-type code
`01`) and writes to `data/processed/tcga_kidney_v1/`. Participants are assigned
to a representative-construction pool or a held-out query set, stratified by
cancer subtype. The transform produces one arithmetic-mean centroid per subtype
in `representatives.npy`, held-out vectors in `queries.npy`, their labels,
the full `representative_pool.npy` for constructing alternative representatives
such as medoids or subcentroids, sample and feature tables, and a JSON provenance
record. The source matrices are streamed, so their full contents are never held
in memory. Use `--help` for filtering, role fraction, input/output, seed, and
overwrite options.

After configuring a release build, load the generated representatives and
queries into the exact, flat, and hierarchical C++ indexes with:

```sh
./build/nearest_representative_benchmark \
    --distance l1 \
    --output results/raw/tcga_l1_benchmark.json
./build/nearest_representative_benchmark \
    --distance l2 \
    --output results/raw/tcga_l2_benchmark.json
```

The executable intentionally does not load `representative_pool.npy`; that
matrix is needed only when constructing alternative representatives. Benchmark
settings, timings, accuracy, correct counts, and agreement with exact search are
written to `results/raw/tcga_l1_benchmark.json` or
`results/raw/tcga_l2_benchmark.json`. Both reports also record
logical index and query-workspace bytes, unique coordinates acquired, their
fraction of the full dimension, and total sampled multiplicity.

## TCGA Pan-Cancer

Download matching GDC STAR-FPKM matrices for all 33 TCGA cancer projects from
the UCSC Xena GDC hub:

```sh
python3 scripts/download_tcga_pancancer.py
```

The current compressed download is approximately 2.8 GiB. The downloader
streams each file, retries transient failures, records HTTP metadata and SHA-256
checksums, and validates that every cohort contains the same 60,660 Ensembl
features in the same order. It writes the matrices, `targets.csv`, and
`manifest.json` to `data/raw/tcga_pancancer/` only after all downloads validate.
Use `--force` to replace a complete earlier download and `--retries` to change
the number of attempts per file.

The raw matrices include every sample type published for each project. Primary
disease filtering belongs in the transformation step and must be project-aware;
in particular, sample-type code `01` must not be assumed for every cancer.

Transform the matrices into 33 project centroids and participant-disjoint
held-out queries with:

```sh
python3 scripts/transform_tcga_pancancer.py
```

The default filter retains primary solid tumors (sample type `01`) for 32
projects and primary blood-derived cancer samples (sample type `03`) for LAML.
With the current download this retains 10,041 samples before the default 70/30
participant split. Override an individual project when a different experimental
cohort is intended; for example, include both primary and metastatic melanoma
with:

```sh
python3 scripts/transform_tcga_pancancer.py \
    --sample-types TCGA-SKCM=01,06
```

The transform streams all source matrices, verifies their feature alignment,
constructs each centroid only from its representative-pool participants, and
writes `representatives.npy`, `queries.npy`, `query_labels.npy`, sample and
feature tables, and complete JSON provenance to
`data/processed/tcga_pancancer_v1/`. The representative-construction samples
remain identified in `samples.csv` but are not duplicated into another large
matrix by default. Pass `--write-representative-pool` if alternative centroid,
medoid, or subcentroid construction requires `representative_pool.npy`.

To learn several representatives per cancer project, materialize that
training-only pool and cluster each project independently:

```sh
python3 scripts/transform_tcga_pancancer.py \
    --write-representative-pool --force
python3 scripts/expand_tcga_pancancer_representatives.py \
    --representatives-per-class 4
```

The default expansion selects 256 high-variance genes independently within
each training project for k-means assignments, then computes every subcentroid
over all features. It writes `tcga_pancancer_k4_v1/` with 132 representative
rows and `representative_labels.npy`, which maps those rows back to the 33
cancer labels. Held-out queries are hard-linked unchanged and never participate
in feature selection, clustering, or centroid construction.

Run the existing benchmark against the transformed data with:

```sh
./build/nearest_representative_benchmark \
    --distance l1 \
    --dataset data/processed/tcga_pancancer_v1 \
    --output results/raw/tcga_pancancer_l1_benchmark.json

./build/nearest_representative_benchmark \
    --distance l2 \
    --dataset data/processed/tcga_pancancer_v1 \
    --output results/raw/tcga_pancancer_l2_benchmark.json
```

For the reproducible multi-seed flat sweep, run the versioned batch setup. Exact
queries are computed once and shared by all 25 approximate configurations:

```sh
python3 scripts/run_benchmark_sweep.py \
    --setup experiments/tcga_pancancer_flat_sweep.tsv

python3 scripts/run_benchmark_sweep.py \
    --setup experiments/tcga_pancancer_l2_flat_sweep.tsv
```

To test whether the hierarchical L2 projection becomes useful with all 33
Pan-Cancer representatives, run the matched flat/hierarchical experiment. It
computes exact predictions once, compares ten projection dimensions over five
repetition counts and five seeds, and writes aggregate summaries:

```sh
python3 scripts/run_benchmark_sweep.py \
    --setup experiments/tcga_pancancer_l2_hierarchical_sweep.tsv
```

The expanded hierarchy grid now includes ten projection dimensions through
`p = 128`. Use `--force` to replace a report produced by the earlier grid.

## COIL-100

Download the processed COIL-100 archive from Columbia University and validate
its complete inventory with:

```sh
python3 scripts/download_coil100.py
```

The PPM tarball is about 250 MiB and contains 7,200 views: 100 physical objects at
72 azimuths separated by 5 degrees. The downloader streams and hashes the tarball,
safely extracts only after checking archive paths and size limits, validates
every `(object, angle)` pair and PPM header, and writes `manifest.json` to
`data/raw/coil100/`. The extracted image payload is about 676 MiB. Use `--force`
to replace a previously validated download.

The source files use a COIL-specific binary P6 header with one maximum for each
RGB channel. Their samples are 16-bit and big-endian, despite the common
description of COIL-100 as 8-bit imagery. Transform them into the benchmark's
row-major format with:

```sh
python3 scripts/transform_coil100.py
```

The default split holds out the contiguous azimuth block from 0 through 85
degrees: 18 query views per object and 1,800 queries in total. Each of the 100
representatives is the arithmetic-mean centroid of that object's other 54
views, so query pixels never contribute to its representative. Pixels are
normalized channel-wise to `[0,1]` and flattened in row, column, RGB order,
giving 49,152 dimensions. The transform writes `representatives.npy`,
`queries.npy`, `query_labels.npy`, sample/feature/representative tables, and a
checksum-rich `dataset.json` to `data/processed/coil100_v1/`.

Repeat the experiment with rotated held-out sectors to measure split
sensitivity, for example:

```sh
python3 scripts/transform_coil100.py \
    --query-start-angle 90 \
    --output-dir data/processed/coil100_start90_v1
```

Pass `--write-representative-pool` only when constructing medoids or multiple
subcentroids; the optional matrix is about 1.0 GiB. The standard output can be
loaded directly by the benchmark executable:

```sh
./build/nearest_representative_benchmark \
    --distance l1 \
    --dataset data/processed/coil100_v1 \
    --output results/raw/coil100_l1_benchmark.json

./build/nearest_representative_benchmark \
    --distance l2 \
    --dataset data/processed/coil100_v1 \
    --output results/raw/coil100_l2_benchmark.json
```

For matched multi-seed flat/hierarchical sweeps over both distances, with exact
search computed once per distance, run:

```sh
python3 scripts/run_benchmark_sweep.py \
    --setup experiments/coil100_l1_hierarchical_sweep.tsv
python3 scripts/run_benchmark_sweep.py \
    --setup experiments/coil100_l2_hierarchical_sweep.tsv
```

The full parameter grid and interpretation of the generated comparison
summaries are documented in `experiments/README.md`.

To run the substantially smaller dedicated flat-only sweeps instead:

```sh
python3 scripts/run_benchmark_sweep.py \
    --setup experiments/coil100_l1_flat_sweep.tsv
python3 scripts/run_benchmark_sweep.py \
    --setup experiments/coil100_l2_flat_sweep.tsv
```

Each setup executes 30 flat configurations and computes its exact baseline
once.
