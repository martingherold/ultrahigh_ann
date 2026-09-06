# Data

This directory contains local inputs and generated datasets. Large data files
are intentionally excluded from Git.

```text
raw/          Original, unmodified dataset downloads
processed/    Normalized or converted datasets
splits/       Guidance for reproducible train/test split definitions
synthetic/    Generated benchmark instances
```

Keep dataset download and preparation code in `scripts/data/`. Record the source
URL, download date, checksum, normalization, and generation parameters for
each experiment.

Small fixtures needed by automated tests should go in `tests/data/` and may be
committed when their licenses permit it.

## Synthetic quickstart

Generate the small, deterministic nearest-neighbor dataset used by the
recruiter-facing quickstart with:

```sh
python3 scripts/data/generate_synthetic_dataset.py
```

The default creates 64 reference vectors and 512 queries in 16,384 dimensions
under `data/synthetic/quickstart_v1/`. Variation among reference vectors is
concentrated in 2,048 informative coordinates with low-variation background coordinates.
Queries interpolate between a target and its nearest rival at four controlled
difficulty levels and are verified against exact L2 search. No training data is
created or required. The generated directory is excluded from Git. Use
`--reference-vectors`, `--queries`, and `--dimensions` to change its size.

Dataset generators write `reference_vectors.npy` for reference vectors and
`reference_labels.npy` for their class labels. Training samples and their labels
are stored in `training_pool.npy` and `training_pool_labels.npy` when the pool
is materialized. Centroid metadata goes in `reference_vectors.csv`, and
`samples.csv` identifies training samples with the role `training_pool`.

## TCGA Kidney Cancers

Fetch the GDC-backed matrices published by UCSC Xena:

```sh
python3 scripts/data/download_tcga_kidney.py
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
class centroids and queries for the ANN implementation:

```sh
python3 scripts/data/transform_tcga_kidney.py
```

By default, the transform retains primary solid tumors (TCGA sample-type code
`01`) and writes to `data/processed/tcga_kidney_v1/`. Participants are assigned
to a training pool or a held-out query set, stratified by
cancer subtype. The transform produces one arithmetic-mean centroid per subtype
in `reference_vectors.npy`, explicit centroid labels in
`reference_labels.npy`, held-out vectors in `queries.npy`, their labels,
the full `training_pool.npy` for constructing alternative reference vectors
such as medoids or subcentroids, sample and feature tables, and a JSON provenance
record. The source matrices are streamed, so their full contents are never held
in memory. Use `--training-fraction` to set the fraction of participants used
to construct centroids, and `--help` for filtering, input/output, seed, and
overwrite options.

The kidney-specific preparation workflow is retained as an additional dataset
example, but it is not one of the six checked-in benchmark configurations. The
canonical TCGA experiments use the Pan-Cancer workflow below. The benchmark
executable accepts version-two setup files rather than the former standalone
`--distance` and `--output` options.

## TCGA Pan-Cancer

Download matching GDC STAR-FPKM matrices for all 33 TCGA cancer projects from
the UCSC Xena GDC hub:

```sh
python3 scripts/data/download_tcga_pancancer.py
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
python3 scripts/data/transform_tcga_pancancer.py
```

The default filter retains primary solid tumors (sample type `01`) for 32
projects and primary blood-derived cancer samples (sample type `03`) for LAML.
With the current download this retains 10,041 samples before the default 70/30
participant split. Override an individual project when a different experimental
cohort is intended; for example, include both primary and metastatic melanoma
with:

```sh
python3 scripts/data/transform_tcga_pancancer.py \
    --sample-types TCGA-SKCM=01,06
```

The transform streams all source matrices, verifies their feature alignment,
constructs each centroid only from its training participants, and
writes `reference_vectors.npy`, `reference_labels.npy`, `queries.npy`,
`query_labels.npy`, sample and feature tables, and complete JSON provenance to
`data/processed/tcga_pancancer_v1/`. The training samples
remain identified in `samples.csv` but are not duplicated into another large
matrix by default. Pass `--write-training-pool` if alternative centroid,
medoid, or subcentroid construction requires `training_pool.npy`.

To learn several centroids per cancer project, materialize that
training-only pool and cluster each project independently:

```sh
python3 scripts/data/transform_tcga_pancancer.py \
    --write-training-pool --force
python3 scripts/data/expand_tcga_pancancer_centroids.py \
    --centroids-per-class 4
```

The default expansion selects 256 high-variance genes independently within
each training project for k-means assignments, then computes every subcentroid
over all features. It writes `tcga_pancancer_k4_v1/` with 132 centroid
rows and `reference_labels.npy`, which maps those rows back to the 33
cancer labels. Held-out queries are hard-linked unchanged and never participate
in feature selection, clustering, or centroid construction.

Run the two canonical CPU experiments against the transformed data with:

```sh
python3 scripts/benchmark/run_benchmark_sweep.py \
    --setup experiments/tcga_pancancer/cpu/tcga_pancancer_r33_l2_cpu_sparse_sweep.tsv

python3 scripts/benchmark/run_benchmark_sweep.py \
    --setup experiments/tcga_pancancer/cpu/tcga_pancancer_r33_l2_cpu_uniform_vs_flat.tsv
```

The CUDA L1 experiment uses the same matrices and keeps the CPU exact scan as
its numerical reference:

```sh
./build-cuda/nearest_neighbor_benchmark \
    --setup experiments/tcga_pancancer/cuda/tcga_pancancer_r33_l1_cuda_exact_vs_flat.tsv
```

## COIL-100

Download the processed COIL-100 archive from Columbia University and validate
its complete inventory with:

```sh
python3 scripts/data/download_coil100.py
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
python3 scripts/data/transform_coil100.py
```

The default split holds out the contiguous azimuth block from 0 through 85
degrees: 18 query views per object and 1,800 queries in total. Each of the 100
reference vectors is the arithmetic-mean centroid of that object's other 54
views, so query pixels never contribute to its centroid. Pixels are
normalized channel-wise to `[0,1]` and flattened in row, column, RGB order,
giving 49,152 dimensions. The transform writes `reference_vectors.npy`,
`reference_labels.npy`, `queries.npy`, `query_labels.npy`,
sample, feature, and centroid tables, and a
checksum-rich `dataset.json` to `data/processed/coil100_v1/`.

Repeat the experiment with rotated held-out sectors to measure split
sensitivity, for example:

```sh
python3 scripts/data/transform_coil100.py \
    --query-start-angle 90 \
    --output-dir data/processed/coil100_start90_v1
```

Pass `--write-training-pool` only when constructing medoids or multiple
subcentroids; the optional matrix is about 1.0 GiB. The former COIL-100 sweep
grids are intentionally not part of the six-setup public catalog. To extend the
evaluation locally, create a version-two setup pointing at the processed
directory using the contract in `experiments/README.md`.

## GSE2034 breast cancer relapse

Download the official NCBI GEO series matrix and patient-level clinical table:

```sh
python3 scripts/data/download_gse2034.py
```

The downloader retains all 286 samples and all 22,283 Affymetrix HG-U133A
(GPL96) probe sets. GEO stores the clinically relevant relapse endpoint in a
separate table rather than in the series-matrix sample characteristics. The
script discovers that table's current GEO blob URL, joins it by GSM accession,
and validates both files before installing them under `data/raw/gse2034/`.
It does not mistake the matrix's 10-positive `bone relapses` characteristic for
the overall distant-metastasis label.

The current patient table has 179 relapse-free and 107 distant-metastasis
cases. This differs by one case in each class from the older series summary's
180/106 counts. The patient-level table is used as the authoritative label
source and the discrepancy is recorded in `manifest.json`.

Create the benchmark matrices with:

```sh
python3 scripts/data/transform_gse2034.py
```

The default seed-42 stratified split places 200 patients in the training
pool and holds out 86 queries. The transform applies `log2(x+1)`, fits each
probe set's mean and population standard deviation using only the
training pool, and applies those statistics to both roles. It then forms
one training-only arithmetic-mean centroid for each outcome class. The output
under `data/processed/gse2034_v1/` contains two class centroids and 86 queries
in the original 22,283 dimensions, as well as the training pool, normalization
arrays, labels, sample/probe tables, checksums, and complete split provenance.

To replace each class centroid with ten training-only subcentroids while
preserving the original two-centroid dataset, run:

```sh
python3 scripts/data/expand_gse2034_centroids.py
```

This writes `data/processed/gse2034_k10_v1/` with 20 centroids: ten
labeled relapse-free and ten labeled distant-metastasis. Each class is
clustered independently with deterministic k-means++. The 256 highest-variance
probe sets within that class's training pool determine assignments, then
each final subcentroid is recomputed over all 22,283 probe sets. Queries are
hard-linked unchanged and never influence feature selection, clustering, or
centroid construction. Use `--centroids-per-class`,
`--clustering-features`, and `--seed` for alternative expansions.

GSE2034 preparation remains reproducible, but its exploratory sweep grids are
not part of the six-setup public catalog. Create a version-two setup pointing
at either processed directory when extending the biological validation.

Use `--training-fraction`, `--seed`, and `--output-dir` to create
additional independent splits. Do not use `--force` to rotate a split in place
when the old split's report needs to remain reproducible; write a new processed
directory and a matching setup instead.
