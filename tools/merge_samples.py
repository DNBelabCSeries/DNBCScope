#!/usr/bin/env python3
"""Merge multiple single-cell samples into one h5ad file.

Usage: merge_samples.py '<json_input>'

JSON input format:
{
  "samples": [
    {"path": "/path/to/sample1", "name": "Sample1", "groups": ["control", "batch1"]},
    {"path": "/path/to/sample2.h5ad", "name": "Sample2", "groups": ["treated", "batch2"]}
  ],
  "group_columns": ["group", "group_batch"],
  "output_dir": "/path/to/output",
  "batch_correction": "harmony" | "combat" | "none"
}

The first group column is always written as `obs["group"]` for backwards
compatibility. Additional columns are written under their sanitized names
(as provided in `group_columns`).
"""

import json
import os
import sys
import time

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

try:
    from expression_source import normalize_windows_path, read_10x_mtx_flexible, select_count_matrix
except ImportError as error:
    raise RuntimeError(
        "DNBCScope could not load expression_source.py next to merge_samples.py"
    ) from error


def load_sample(path: str) -> ad.AnnData:
    """Load a single sample from MTX directory or h5ad file."""
    path = normalize_windows_path(path)
    if os.path.isfile(path) and path.lower().endswith(".h5ad"):
        # Do not materialize embeddings, layers, or large metadata columns
        # before minimize_for_merge selects the count matrix.  This matters
        # most on Windows, where the extra copies make the merge appear to
        # stall while the OS starts paging.
        return ad.read_h5ad(path, backed="r")
    elif os.path.isdir(path):
        h5ad_files = sorted(
            (name for name in os.listdir(path) if name.lower().endswith(".h5ad")),
            key=lambda name: (name.casefold(), name),
        )
        if h5ad_files:
            return ad.read_h5ad(os.path.join(path, h5ad_files[0]), backed="r")
        return read_10x_mtx_flexible(path)
    else:
        raise RuntimeError(f"Unsupported input: {path}")


def emit_progress(percent: int, message: str) -> None:
    """Emit merge progress on stderr so Rust can forward it to the UI."""
    print(
        f"DNBC_MERGE_PROGRESS|{max(0, min(100, int(percent)))}|{message}",
        file=sys.stderr,
        flush=True,
    )


def choose_count_matrix(adata: ad.AnnData):
    """Return the raw-count-like matrix and matching gene names."""
    matrix, names, _label, _kind = select_count_matrix(adata)
    return matrix, names


def as_merge_csr(matrix) -> sp.csr_matrix:
    """Materialize one merge matrix as CSR with compact float32 values."""
    if hasattr(matrix, "to_memory"):
        matrix = matrix.to_memory()
    if not sp.issparse(matrix):
        matrix = sp.csr_matrix(matrix)
    elif matrix.format != "csr":
        matrix = matrix.tocsr()
    if matrix.data.dtype != np.dtype("float32") and not np.issubdtype(
        matrix.data.dtype, np.complexfloating
    ):
        # Casting performs the unavoidable full value pass once.  The previous
        # finite/max checks scanned every non-zero value first and then cast it,
        # which made large per-sample reads noticeably slower.  Check the
        # converted buffer for overflow/NaN and retain the original dtype when
        # float32 cannot represent the values safely.
        with np.errstate(over="ignore", invalid="ignore"):
            narrowed = matrix.astype(np.float32, copy=False)
        if not narrowed.data.size or bool(np.isfinite(narrowed.data).all()):
            matrix = narrowed
    return matrix


def categorical_from_blocks(lengths: list[int], values: list[str]) -> pd.Categorical:
    """Build repeated sample metadata without allocating Python strings per cell."""
    total_rows = sum(lengths)
    categories: list[str] = []
    category_index: dict[str, int] = {}
    codes = np.empty(total_rows, dtype=np.int32)
    offset = 0
    for length, value in zip(lengths, values):
        normalized = str(value).strip() if value is not None else ""
        normalized = normalized or "unknown"
        code = category_index.get(normalized)
        if code is None:
            code = len(categories)
            categories.append(normalized)
            category_index[normalized] = code
        codes[offset : offset + length] = code
        offset += length
    return pd.Categorical.from_codes(codes, categories=categories)


def minimize_for_merge(
    adata: ad.AnnData,
    sample_name: str,
    group_values: list,
    group_columns: list,
    include_metadata: bool = True,
) -> ad.AnnData:
    """Strip heavy slots and keep only count matrix + minimal metadata.

    For a single-sample merge, writes compact categorical sample/group
    metadata. Multi-sample callers defer metadata construction until the final
    matrix exists so they do not retain two generations of per-cell codes.
    """
    matrix, gene_names = choose_count_matrix(adata)
    # AnnData exposes backed sparse h5ad matrices as a lightweight
    # ``CSRDataset`` proxy. It is not recognised by scipy.issparse(), and
    # passing that proxy directly to csr_matrix creates an object-dtype matrix
    # (then fails). Materialize this sample's selected count matrix exactly
    # once before the CSR conversion; merging necessarily needs the rows in
    # memory for the final sparse vstack anyway.
    matrix = as_merge_csr(matrix)

    obs = pd.DataFrame(index=adata.obs_names.copy())
    if include_metadata:
        row_count = int(adata.n_obs)
        obs["sample_id"] = categorical_from_blocks([row_count], [sample_name])
        obs["sample"] = obs["sample_id"]
        for col_index, col_name in enumerate(group_columns):
            value = group_values[col_index] if col_index < len(group_values) else ""
            value = str(value).strip() if value is not None else ""
            value = value or "unknown"
            obs[col_name] = categorical_from_blocks([row_count], [value])

    var = pd.DataFrame(index=gene_names.copy())
    minimal = ad.AnnData(X=matrix, obs=obs, var=var)
    minimal.var_names_make_unique()
    return minimal


def merge_minimal_adatas(
    adatas: list[ad.AnnData],
    sample_names: list[str],
    samples: list[dict],
    group_columns: list[str],
    progress_callback=None,
) -> ad.AnnData:
    """Concatenate already-minimized matrices without ``anndata.concat``.

    ``anndata.concat(join='inner')`` performs a general-purpose alignment and
    allocates intermediate AnnData objects. That is convenient but expensive
    for large sparse matrices, especially on Windows where the temporary
    copies can trigger paging. Here gene order is known and each sample is
    sliced once before a single sparse ``vstack``.
    """
    common_genes = adatas[0].var_names
    for adata in adatas[1:]:
        common_genes = common_genes.intersection(adata.var_names, sort=False)
    if len(common_genes) < 2:
        raise RuntimeError("Samples do not share enough common genes")

    matrices = []
    row_counts = [int(adata.n_obs) for adata in adatas]
    merged_barcodes = np.empty(sum(row_counts), dtype=object)
    row_offset = 0
    group_values_by_column: list[list[str]] = [
        [] for _column in group_columns
    ]
    for sample_index, (adata, sample_name) in enumerate(zip(adatas, sample_names)):
        # Most multi-sample uploads use the same reference and gene order for
        # every sample. Avoiding an otherwise unnecessary sparse column slice
        # saves a full matrix copy per sample. Only build an indexer when a
        # sample really needs reordering or intersection alignment.
        if adata.var_names.equals(common_genes):
            matrix = adata.X
        else:
            positions = adata.var_names.get_indexer(common_genes)
            if (positions < 0).any():
                raise RuntimeError(f"Failed to align genes for sample {sample_name}")
            matrix = adata.X[:, positions]
        matrix = as_merge_csr(matrix)
        matrices.append(matrix)

        rows = row_counts[sample_index]
        merged_barcodes[row_offset : row_offset + rows] = np.fromiter(
            (f"{barcode}-{sample_name}" for barcode in map(str, adata.obs_names)),
            dtype=object,
            count=rows,
        )
        row_offset += rows

        if progress_callback is not None:
            progress_callback(
                50 + round(22 * (sample_index + 1) / max(1, len(adatas))),
                f"Aligning sample {sample_index + 1}/{len(adatas)}",
            )

        if "groups" in samples[sample_index]:
            group_values = list(samples[sample_index]["groups"])
        else:
            group_values = [samples[sample_index].get("group", "")]
        while len(group_values) < len(group_columns):
            group_values.append("")
        for col_index, _column in enumerate(group_columns):
            value = group_values[col_index]
            value = str(value).strip() if value is not None else ""
            group_values_by_column[col_index].append(value or "unknown")

    if progress_callback is not None:
        progress_callback(76, "Stacking sparse expression matrices")
    obs = pd.DataFrame(index=pd.Index(merged_barcodes, dtype="object"))
    sample_category = categorical_from_blocks(row_counts, sample_names)
    obs["sample_id"] = sample_category
    obs["sample"] = sample_category
    for column, values in zip(group_columns, group_values_by_column):
        obs[column] = categorical_from_blocks(row_counts, values)
    merged = ad.AnnData(
        X=stack_csr_matrices(matrices),
        obs=obs,
        var=pd.DataFrame(index=common_genes.copy()),
    )
    merged.var_names_make_unique()
    return merged


def stack_csr_matrices(matrices: list[sp.csr_matrix]) -> sp.csr_matrix:
    """Stack CSR blocks without scipy's temporary COO/intermediate copies.

    ``scipy.sparse.vstack`` is convenient but can briefly retain the input
    blocks, a COO conversion, and the final CSR at the same time.  For a
    multi-sample million-cell merge that transient allocation is often what
    pushes Windows into paging.  All blocks have already been aligned and
    converted to CSR above, so their buffers can be copied directly into one
    pre-sized CSR allocation.
    """
    if not matrices:
        raise ValueError("Cannot stack an empty list of matrices")
    n_cols = matrices[0].shape[1]
    if any(matrix.shape[1] != n_cols for matrix in matrices):
        raise ValueError("CSR blocks must have the same number of columns")

    total_rows = sum(int(matrix.shape[0]) for matrix in matrices)
    total_nnz = sum(int(matrix.nnz) for matrix in matrices)
    data_dtype = np.result_type(*(matrix.data.dtype for matrix in matrices))
    index_dtype = np.result_type(*(matrix.indices.dtype for matrix in matrices))
    data = np.empty(total_nnz, dtype=data_dtype)
    indices = np.empty(total_nnz, dtype=index_dtype)
    indptr = np.empty(total_rows + 1, dtype=np.int64)
    indptr[0] = 0

    data_offset = 0
    row_offset = 0
    for matrix in matrices:
        rows = int(matrix.shape[0])
        nnz = int(matrix.nnz)
        data[data_offset : data_offset + nnz] = matrix.data
        indices[data_offset : data_offset + nnz] = matrix.indices
        row_counts = np.diff(matrix.indptr).astype(np.int64, copy=False)
        indptr[row_offset + 1 : row_offset + rows + 1] = np.cumsum(row_counts, dtype=np.int64) + data_offset
        data_offset += nnz
        row_offset += rows

    return sp.csr_matrix((data, indices, indptr), shape=(total_rows, n_cols))


def _load_json_argument(value):
    if value.startswith("@"):
        with open(value[1:], encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(value)


def main() -> None:
    if len(sys.argv) < 2:
        raise RuntimeError("Usage: merge_samples.py '<json_input>'")

    config = _load_json_argument(sys.argv[1])
    samples = config["samples"]
    output_dir = normalize_windows_path(config["output_dir"])
    os.makedirs(output_dir, exist_ok=True)

    if not samples:
        raise RuntimeError("No samples provided")
    names = [str(sample.get("name", "")).strip() for sample in samples]
    if any(not name for name in names):
        raise RuntimeError("Every sample must have a non-empty name")
    if len(set(names)) != len(names):
        raise RuntimeError("Sample names must be unique")

    # Group columns: first column is always "group" for backwards compat.
    # If the caller supplied column names, use them (coercing index 0).
    group_columns = config.get("group_columns") or ["group"]
    if not group_columns:
        group_columns = ["group"]
    if group_columns[0] != "group":
        group_columns = ["group"] + group_columns

    started = time.perf_counter()
    load_seconds = 0.0
    emit_progress(2, f"Preparing {len(samples)} sample datasets")
    adatas = []
    sample_names = []
    groups = []  # first-column values, for backwards-compatible output

    for sample_index, sample in enumerate(samples):
        path = sample["path"]
        name = sample["name"]
        # Accept both legacy "group" (string) and new "groups" (list).
        if "groups" in sample:
            group_values = list(sample["groups"])
        else:
            group_values = [sample.get("group", "")]
        # Pad to match group_columns length.
        while len(group_values) < len(group_columns):
            group_values.append("")

        sample_started = time.perf_counter()
        source = load_sample(path)
        try:
            adata = minimize_for_merge(
                source,
                name,
                group_values,
                group_columns,
                include_metadata=len(samples) == 1,
            )
        finally:
            if getattr(source, "isbacked", False):
                source.file.close()
        adatas.append(adata)
        sample_names.append(name)
        first_group = group_values[0] if group_values else ""
        first_group = str(first_group).strip() if first_group is not None else ""
        groups.append(first_group or "unknown")
        load_seconds += time.perf_counter() - sample_started
        emit_progress(
            8 + round(38 * (sample_index + 1) / max(1, len(samples))),
            f"Read sample {sample_index + 1}/{len(samples)} ({time.perf_counter() - sample_started:.1f}s)",
        )

    # Do not retain the last backed input object (and its file handles) while
    # the merged result is compressed.
    del source

    # Merge raw expression only. Normalization, dimensionality reduction, and
    # batch correction belong in run_analysis.py so they happen exactly once.
    merge_started = time.perf_counter()
    emit_progress(48, "Aligning genes across samples")
    if len(adatas) == 1:
        merged = adatas[0]
    else:
        merged = merge_minimal_adatas(
            adatas,
            sample_names,
            samples,
            group_columns,
            progress_callback=emit_progress,
        )
    merge_seconds = time.perf_counter() - merge_started
    emit_progress(80, f"Expression matrices combined ({merge_seconds:.1f}s)")
    # The sparse stack owns its own storage. Release per-sample matrices before
    # writing the merged h5ad so compression does not keep both generations in
    # RAM (a common source of paging on Windows).
    del adatas

    # Save as h5ad
    merged_path = os.path.join(output_dir, "merged.h5ad")
    # Keep the temporary filename short (important for Windows long paths) and
    # retain the .h5ad suffix so anndata always selects its H5AD writer.
    partial_path = os.path.join(output_dir, ".tmp.h5ad")
    batch_correction = config.get("batch_correction", "harmony")
    write_started = time.perf_counter()
    emit_progress(84, "Writing merged H5AD compatibility source")
    # Write beside the final file and commit with a rename. If the tracked
    # process is cancelled or the disk fills during compression, a partial
    # h5ad must never masquerade as a valid recent-project source on the next
    # launch.
    try:
        merged.write_h5ad(partial_path, compression="lzf")
        os.replace(partial_path, merged_path)
    finally:
        try:
            if os.path.exists(partial_path):
                os.remove(partial_path)
        except OSError:
            pass
    write_seconds = time.perf_counter() - write_started
    h5ad_bytes = os.path.getsize(merged_path)

    # Get all gene names
    gene_names = list(merged.var_names)

    emit_progress(
        100,
        f"Merge complete ({time.perf_counter() - started:.1f}s; "
        f"load {load_seconds:.1f}s; combine {merge_seconds:.1f}s; "
        f"write {write_seconds:.1f}s; {h5ad_bytes / (1024 ** 2):.1f} MiB)",
    )
    print(json.dumps({
        "merged_path": merged_path,
        "cell_count": int(merged.n_obs),
        "gene_names": gene_names,
        "sample_names": sample_names,
        "groups": groups,
        "group_columns": group_columns,
        "batch_correction": batch_correction,
        "performance": {
            "load_seconds": load_seconds,
            "merge_seconds": merge_seconds,
            "write_seconds": write_seconds,
            "total_seconds": time.perf_counter() - started,
            "h5ad_bytes": h5ad_bytes,
            "data_dtype": str(merged.X.dtype),
        },
    }))


if __name__ == "__main__":
    main()
