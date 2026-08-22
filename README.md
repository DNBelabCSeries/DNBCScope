# DNBCScope Python Tools

This `opensource` branch contains a deliberately small, source-only subset of
DNBCScope: the Python programs under `tools/` and this README. It does not
contain the desktop UI, the Rust/Tauri application shell, bundled Python
environments, private build configuration, model databases, or user data.

The tools are shared building blocks for local single-cell expression analysis,
metadata inspection, differential analysis, annotation, plotting, and data
conversion. They can be used independently where a command-line interface is
provided, or embedded by an application that speaks the worker protocols.

## Included programs

### Expression and project data

- `expression_source.py` — select and validate expression matrices from MTX
  directories or H5AD files.
- `native_csr.py` — read the disk-backed native CSR/CSC representation used by
  large projects.
- `convert_mtx.py` — convert a Matrix Market directory into project-oriented
  metadata and binary buffers.
- `convert_h5ad.py` — convert an H5AD file into MTX and TSV outputs.
- `export_h5ad.py` — export an expression source and supplied metadata to H5AD.
- `write_merged_h5ad.py` — write an H5AD container from already prepared CSR
  buffers.
- `merge_samples.py` — combine multiple sample sources into one AnnData object.

### Analysis and workers

- `run_analysis.py` — normalization, highly variable gene selection, PCA,
  neighbors, clustering, and UMAP/t-SNE-style projections.
- `run_differential.py` — pairwise and one-versus-rest differential expression,
  including pseudobulk-compatible result structures.
- `run_annotation.py` — marker-based and CellTypist-compatible annotation
  backends.
- `run_scrublet.py` — doublet scoring for H5AD or MTX expression sources.
- `scanpy_plot.py` — generate dot, violin, and matrix plot payloads.
- `scientific_worker.py` — resident JSON-lines worker for repeated analysis
  tasks without paying the scientific import cost for every request.

### Metadata, diagnostics, and runtime helpers

- `read_h5ad_meta.py` and `read_native_csr_meta.py` — inspect barcodes, genes,
  QC metrics, categories, and expression summaries.
- `compare_analysis_runs.py` — compare deterministic analysis metadata between
  two runs.
- `runtime_fingerprint.py` — record the scientific runtime and determinism
  contract used for cache validation.
- `process_metrics.py` — small cross-platform memory and process helpers.
- `download_celltypist_models.py` — download a selected CellTypist model with
  retries and checksum/size safeguards.

### Environment and packaging helpers

- `setup_env.py`, `bundle_python.py`, and `prune_python_env.py` — create,
  package, and reduce a scientific Python runtime for the desktop application.

These packaging helpers intentionally retain references to the larger
DNBCScope build configuration. That configuration is not part of this public
source subset, so use them only after supplying your own runtime configuration.

## Quick start

Python 3.12 is recommended. Create an isolated environment and install the
scientific packages required by the commands you plan to use:

```bash
python3.12 -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install numpy scipy pandas h5py anndata scanpy matplotlib scikit-learn
```

Optional workflows may additionally require packages such as `scrublet`,
`harmonypy`, `bbknn`, `python-igraph`, `leidenalg`, `celltypist`, or
`pydeseq2`. Install versions compatible with your Python and operating system;
this branch intentionally does not ship a platform-specific environment or a
dependency lock file.

Run commands from the repository root so sibling imports resolve correctly.
For example:

```bash
python tools/convert_h5ad.py input.h5ad converted/
python tools/read_h5ad_meta.py input.h5ad
python tools/run_scrublet.py input.h5ad
```

Most analysis commands also accept JSON input or emit JSON results. Their
`--help` output and module docstrings describe the exact contract. The
resident worker uses JSON lines on standard input/output; progress and
diagnostic messages are written to standard error so result streams remain
machine-readable.

## Data and model assets

This branch does not include expression data, annotation databases, downloaded
models, or the desktop application's native CSR writer. To use an annotation
workflow, provide the marker database/model assets separately and point the
tool to them using its documented command-line options or environment
variables. A native CSR manifest must be produced by a compatible writer
before `native_csr.py` can read it.

All input and output paths are explicit. The tools do not upload expression
data automatically. Network access is used only by commands that explicitly
download a model or a runtime archive.

## Development checks

The public subset has no bundled application test harness. A quick syntax
check is:

```bash
python -m compileall -q tools
```

For production use, validate numerical results, package versions, input
orientation, sparse-matrix dimensions, and generated files on a representative
dataset before integrating the tools into another application.

## Scope and license

This is a source snapshot, not a promise of compatibility with every release
of the full DNBCScope desktop application. The repository currently contains
no separate license file; reuse and redistribution rights must be confirmed
with the repository owner before incorporating this code into another product.
