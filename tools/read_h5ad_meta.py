#!/usr/bin/env python3
"""
Read h5ad metadata (QC metrics, gene names, barcodes) without writing files to disk.
Outputs JSON to stdout for the Rust backend to parse.

Usage:
    python read_h5ad_meta.py <h5ad_path>                     # QC metadata
    python read_h5ad_meta.py <h5ad_path> --gene <gene_name>  # Gene expression vector
"""

import sys
import json
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)


def emit_progress(percent: int, message: str) -> None:
    """Emit QC metadata progress without polluting the JSON stdout payload."""
    print(
        f"DNBC_PROGRESS|qc|{max(0, min(100, int(percent)))}|{message}",
        file=sys.stderr,
        flush=True,
    )

try:
    from expression_source import (
        matrix_is_count_like,
        normalize_windows_path,
        select_expression_matrix,
    )
except ImportError as error:
    raise RuntimeError(
        "DNBCScope could not load expression_source.py next to read_h5ad_meta.py"
    ) from error

def parse_mito_prefixes(raw: str):
    """Parse comma-separated mito gene names or prefixes. Empty → ['MT'].
    Each token can be a prefix (MT, MT-) or a full gene name (MT-ND1);
    matching is case-insensitive via startswith."""
    prefixes = [p.strip() for p in str(raw).split(",") if p.strip()]
    return prefixes if prefixes else ["MT"]


def is_mito_gene(name: str, prefixes) -> bool:
    """Case-insensitive mitochondrial gene match.

    The common shorthand ``MT`` deliberately means ``MT-``, ``MT.``, or
    ``MT_``. Matching every ``MT*`` would incorrectly classify genes such as
    MTHFD1 and MTOR as mitochondrial. Other explicit prefixes and full gene
    names retain normal prefix semantics.
    """
    upper = name.upper()
    for prefix in prefixes:
        token = str(prefix).strip().upper()
        if token == "MT":
            if upper.startswith(("MT-", "MT.", "MT_")):
                return True
        elif upper.startswith(token):
            return True
    return False


def select_expression_source(adata):
    matrix, names, _is_counts, label, _kind = select_expression_matrix(
        adata, allow_negative=True
    )
    return matrix, [str(name) for name in names], label


def load_h5ad(h5ad_path: str, backed: bool = False):
    import anndata as ad
    import time

    # Rust may pass a `\\?\`-prefixed extended-length path for deep directories.
    # h5py/HDF5 rejects the prefix on ordinary paths (WinError 206); normalize it
    # exactly like the analysis/expression loaders so all h5ad reads agree.
    h5ad_path = normalize_windows_path(h5ad_path)
    start = time.time()
    adata = ad.read_h5ad(h5ad_path, backed="r") if backed else ad.read_h5ad(h5ad_path)
    elapsed = time.time() - start
    print(f"[PYTHON] H5AD file loaded in {elapsed:.3f}s", file=sys.stderr)

    selected_matrix, gene_names, source_label = select_expression_source(adata)
    print(f"[PYTHON] Expression source: {source_label}", file=sys.stderr)
    barcodes = [str(barcode) for barcode in adata.obs_names]

    expected_shape = (len(barcodes), len(gene_names))
    if selected_matrix.shape != expected_shape:
        raise ValueError(
            f"Unexpected selected expression matrix shape {selected_matrix.shape}; expected "
            f"{expected_shape} (cells x genes)"
        )

    return adata, gene_names, barcodes


def _stream_matrix_qc(matrix, n_cells, n_features, mito_mask, need_cell_metrics, progress_callback=None):
    """Scan a matrix in row chunks without materializing a backed h5ad.

    AnnData exposes backed sparse X as a CSRDataset. Slicing a bounded row
    range returns an ordinary scipy CSR matrix, so the same implementation
    works for backed dense, backed sparse, and in-memory matrices while keeping
    peak memory proportional to the chunk size.
    """
    import numpy as np
    import scipy.sparse as sp

    counts = np.zeros(n_cells, dtype=float) if need_cell_metrics else None
    genes = np.zeros(n_cells, dtype=float) if need_cell_metrics else None
    mito_counts = np.zeros(n_cells, dtype=float) if need_cell_metrics else None
    # Accumulate once while streaming the matrix so feature-search suggestions
    # can be ranked without loading one expression column per gene later.
    gene_expression_sums = np.zeros(n_features, dtype=np.float64)
    detected = np.zeros(n_features, dtype=bool)
    # Keep dense backed reads around a modest 16 MiB instead of allowing an
    # 8k-row chunk of a 30k-gene matrix to become a multi-gigabyte temporary.
    sparse_like = sp.issparse(matrix) or getattr(matrix, "format", None) in {"csr", "csc"}
    if sparse_like:
        chunk_size = 8192
    else:
        chunk_size = max(1, min(8192, (16 * 1024 * 1024) // max(1, n_features * 8)))

    for start in range(0, n_cells, chunk_size):
        stop = min(n_cells, start + chunk_size)
        chunk = matrix[start:stop]
        if hasattr(chunk, "to_memory"):
            chunk = chunk.to_memory()
        if not sp.issparse(chunk):
            chunk = np.asarray(chunk)
        elif chunk.format != "csr":
            chunk = chunk.tocsr()

        positive = chunk > 0
        if sp.issparse(positive):
            positive = positive.tocsr()
            detected[positive.indices] = True
            gene_expression_sums += np.asarray(chunk.sum(axis=0), dtype=np.float64).ravel()
            if need_cell_metrics:
                counts[start:stop] = np.asarray(chunk.sum(axis=1)).ravel()
                genes[start:stop] = np.asarray(positive.sum(axis=1)).ravel()
                if mito_mask.any():
                    mito_counts[start:stop] = np.asarray(
                        chunk[:, mito_mask].sum(axis=1)
                    ).ravel()

        else:
            positive = np.asarray(positive)
            detected |= np.any(positive, axis=0)
            gene_expression_sums += np.asarray(chunk.sum(axis=0), dtype=np.float64).ravel()
            if need_cell_metrics:
                counts[start:stop] = np.asarray(chunk.sum(axis=1)).ravel()
                genes[start:stop] = np.asarray(positive.sum(axis=1)).ravel()
                if mito_mask.any():
                    mito_counts[start:stop] = np.asarray(
                        chunk[:, mito_mask].sum(axis=1)
                    ).ravel()

        if progress_callback is not None:
            progress_callback(
                5 + round(90 * stop / max(1, n_cells)),
                f"Calculating QC metrics ({stop:,}/{n_cells:,} cells)",
            )

    return counts, genes, mito_counts, int(detected.sum()), gene_expression_sums


def _natural_category_key(value):
    """Numeric-aware sort key so cluster/sample labels order as 1, 2, 10."""
    import re

    return tuple(
        (0, int(part), "") if part.isdigit() else (1, 0, part)
        for part in re.split(r"(\d+)", value)
    )


def _is_numeric_category(value):
    import re

    return bool(re.match(r"^\s*[+-]?\d+(?:\.0+)?\s*$", value))


def category_info(obs, column):
    import numpy as np
    import pandas as pd

    if not column or column not in obs.columns:
        return None
    series = obs[column]
    from_categorical = isinstance(series.dtype, pd.CategoricalDtype)
    if from_categorical:
        # Categorical columns carry the category order chosen by whoever
        # produced the h5ad: DNBCScope exports keep custom-group order and
        # Scanpy natsorts cluster labels. Reuse that order so annotations,
        # samples and groups keep their original positions and colors.
        names = [str(category) for category in series.cat.categories]
        ids = np.asarray(series.cat.codes)
        missing = ids < 0
        if bool(missing.any()):
            # Pandas uses -1 for missing categorical values. Reusing an
            # existing Unassigned category is important: appending a second
            # label would make category labels and encoded ids disagree.
            if "Unassigned" in names:
                missing_id = names.index("Unassigned")
            else:
                names.append("Unassigned")
                missing_id = len(names) - 1
            ids = np.where(missing, missing_id, ids)
    else:
        # Plain string columns have no stored order; number categories by
        # first appearance and sort them naturally below.
        values = series.astype("string").fillna("Unassigned")
        ids, unique_names = values.factorize(sort=False)
        names = [str(name) for name in unique_names]
    if len(names) > 256:
        raise ValueError(
            f"Column {column!r} has {len(names)} categories; DNBCScope currently supports at most 256"
        )
    # Keep the stored category order for text labels (annotations, cell
    # types). Purely numeric labels (cluster ids) must follow numeric order
    # even when stored lexicographically ("0", "1", "10", "2"), and columns
    # without a stored order are sorted naturally for determinism.
    if from_categorical and not all(_is_numeric_category(name) for name in names):
        order = list(range(len(names)))
    else:
        order = sorted(range(len(names)), key=lambda i: _natural_category_key(names[i]))
    remap = np.empty(len(names), dtype=np.uint8)
    for new_index, old_index in enumerate(order):
        remap[old_index] = new_index
    counts = np.bincount(ids, minlength=len(names))
    return {
        "names": [names[old_index] for old_index in order],
        "ids": remap[np.asarray(ids)].astype(np.uint8).tolist(),
        "counts": counts[order].astype(int).tolist(),
    }


def humanize_obs_name(name):
    """Make an obs column readable without changing its source key.

    SHARED CONTRACT — keep in sync with `h5adColumnDisplayName` in
    `src/lib/h5ad-import.ts`. The two implementations must produce identical
    output for identical input; both are checked against the same fixture
    (`test/fixtures/h5ad-column-display.json`) by
    `test/python/test_read_h5ad_meta.py` and
    `test/unit/lib/h5ad-import.test.ts`. Edit both sides (and the fixture) when
    the rule changes.

    Transformation (applied in order):
      1. trim leading/trailing whitespace
      2. replace runs of "_" with a single space
      3. replace runs of "-" (with optional surrounding spaces) with " · "
      4. collapse internal whitespace to a single space, trim
      5. empty input -> "Unnamed column"
    """
    import re

    value = str(name).strip()
    value = re.sub(r"[_]+", " ", value)
    value = re.sub(r"\s*[-]+\s*", " · ", value)
    return re.sub(r"\s+", " ", value).strip() or "Unnamed column"


def embedding_xy(adata, key):
    import numpy as np

    if not key or key not in adata.obsm:
        return None
    values = np.asarray(adata.obsm[key])
    if values.ndim != 2 or values.shape[0] != adata.n_obs or values.shape[1] < 2:
        raise ValueError(f"Embedding {key!r} must have shape (cells, >=2)")
    xy = np.asarray(values[:, :2], dtype=np.float32)
    if not np.isfinite(xy).all():
        raise ValueError(f"Embedding {key!r} contains non-finite coordinates")
    return xy.reshape(-1).tolist()


def suggested_obs_role(name):
    import re

    normalized = re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")
    if normalized in {
        "cell_type",
        "celltype",
        "cell_types",
        "annotation",
        "predicted_labels",
        "cell_type_annotation",
    }:
        return "annotation"
    if any(token in normalized for token in ("sctype", "sc_type", "cell_annotation", "celltypist")):
        return "annotation"
    if normalized in {"leiden", "louvain", "seurat_clusters", "cluster", "clusters"}:
        return "clustering"
    if "cluster" in normalized and not normalized.endswith("score"):
        return "clustering"
    if normalized in {"sample", "sample_id", "orig_ident", "batch", "batch_id", "donor", "library", "replicate"}:
        return "sample"
    if normalized in {"group", "condition", "treatment", "disease", "status", "cell_source", "cohort"}:
        return "group"
    return None


def apply_group_column_aliases(adata, raw_group_columns):
    """Map selected source columns to canonical group/group_N columns."""
    columns = []
    for raw in raw_group_columns or []:
        name = str(raw).strip()
        if name and name in adata.obs.columns and name not in columns:
            columns.append(name)
    for index, raw in enumerate(columns):
        canonical = "group" if index == 0 else f"group_{index + 1}"
        if raw != canonical:
            adata.obs[canonical] = adata.obs[raw].copy()
    return columns


def apply_sample_column_alias(adata, raw_sample_column):
    """Map a configured source obs column to the canonical sample field."""
    sample_column = "" if raw_sample_column is None else str(raw_sample_column).strip()
    if not sample_column:
        return ""
    if sample_column not in adata.obs.columns:
        raise RuntimeError(f"Configured sample column is missing from h5ad obs: {sample_column}")
    if sample_column != "sample":
        adata.obs["sample"] = adata.obs[sample_column].copy()
    return sample_column


def inspect_h5ad(adata):
    import pandas as pd

    embeddings = []
    for key, value in adata.obsm.items():
        shape = getattr(value, "shape", ())
        if len(shape) == 2 and shape[0] == adata.n_obs and shape[1] >= 2:
            embeddings.append({"name": str(key), "dimensions": int(shape[1])})

    columns = []
    for name in adata.obs.columns:
        series = adata.obs[name]
        unique_count = int(series.nunique(dropna=False))
        categorical = (
            isinstance(series.dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(series.dtype)
            or pd.api.types.is_string_dtype(series.dtype)
            or pd.api.types.is_bool_dtype(series.dtype)
            or unique_count <= 256
        )
        columns.append(
            {
                "name": str(name),
                "kind": "categorical" if categorical else "numeric",
                "unique_count": unique_count,
                "suggested_role": suggested_obs_role(name),
            }
        )

    # Expression source is informational only (shown in the dialog). Do not
    # let a missing/invalid matrix block the inspection — fall back to a
    # default label so the "use existing analysis" button stays enabled.
    expression_source = "unknown"
    expression_is_counts = False
    try:
        matrix, _genes, source_label = select_expression_source(adata)
        expression_source = source_label
        expression_is_counts = bool(matrix_is_count_like(matrix))
    except Exception as e:
        print(f"[PYTHON] Expression source detection skipped: {e}", file=sys.stderr)

    return {
        "cell_count": int(adata.n_obs),
        "feature_count": int(adata.n_vars),
        "expression_source": expression_source,
        "expression_is_counts": expression_is_counts,
        "embeddings": embeddings,
        "obs_columns": columns,
    }


def get_qc_metadata(
    adata,
    gene_names,
    barcodes,
    import_config=None,
    mito_prefixes=None,
    progress_callback=None,
):
    import numpy as np
    import scipy.sparse

    import_config = import_config or {}
    sample_column = apply_sample_column_alias(adata, import_config.get("sample_column"))
    configured_groups = import_config.get("group_columns") or []
    # Keep the legacy `group` column usable when reopening older projects,
    # while selected H5AD dimensions take precedence for new imports.
    if configured_groups:
        apply_group_column_aliases(adata, configured_groups)

    if mito_prefixes is None:
        mito_prefixes = ["MT"]
    mat, selected_gene_names, source_label = select_expression_source(adata)
    gene_names = selected_gene_names
    n_cells = mat.shape[0]
    n_features = mat.shape[1]
    qc_available = True
    is_backed = bool(getattr(adata, "isbacked", False))
    mito_mask = np.array([is_mito_gene(name, mito_prefixes) for name in gene_names])
    streamed_mito_counts = None
    gene_expression_sums = None

    # Do not present sums of log/scaled expression as molecule counts. When raw
    # counts are unavailable, require standard precomputed QC columns.
    if matrix_is_count_like(mat):
        if is_backed:
            counts_per_cell = None
            genes_per_cell = None
        else:
            counts_per_cell = np.asarray(mat.sum(axis=1)).flatten()
            genes_per_cell = np.asarray((mat > 0).sum(axis=1)).flatten()
        precomputed_mito = None
    else:
        count_column = next((name for name in ("total_counts", "n_counts", "nCount_RNA") if name in adata.obs), None)
        gene_column = next((name for name in ("n_genes_by_counts", "n_genes", "nFeature_RNA") if name in adata.obs), None)
        mito_column = next((name for name in ("pct_counts_mt", "pct_mito", "percent.mt") if name in adata.obs), None)
        if not count_column or not gene_column or not mito_column:
            if not (import_config or {}).get("use_existing"):
                raise RuntimeError(
                    f"Cannot compute count-based QC from {source_label}. Provide raw counts or obs columns for total counts, detected genes, and mitochondrial percentage."
                )
            qc_available = False
            counts_per_cell = np.zeros(n_cells)
            genes_per_cell = np.zeros(n_cells)
            precomputed_mito = np.zeros(n_cells)
        else:
            counts_per_cell = adata.obs[count_column].to_numpy(dtype=float)
            genes_per_cell = adata.obs[gene_column].to_numpy(dtype=float)
            precomputed_mito = adata.obs[mito_column].to_numpy(dtype=float)

    # Backed sparse/dense datasets cannot use scipy's whole-matrix reductions
    # without first materializing X. Scan bounded row chunks instead. When
    # obs already contains QC columns (or QC is intentionally unavailable),
    # cell-level reductions can be skipped while feature totals are still
    # accumulated for the search dropdown.
    if is_backed:
        (
            streamed_counts,
            streamed_genes,
            streamed_mito_counts,
            detected_features,
            gene_expression_sums,
        ) = _stream_matrix_qc(
            mat,
            n_cells,
            n_features,
            mito_mask,
            need_cell_metrics=qc_available and precomputed_mito is None,
            progress_callback=progress_callback,
        )
        if qc_available and precomputed_mito is None:
            counts_per_cell = streamed_counts
            genes_per_cell = streamed_genes

    # Count detected features only when QC is available. A full dense scan can
    # be very expensive for a processed atlas, and the result is not used when
    # the imported project intentionally has no count-based QC.
    if not qc_available:
        detected_features = n_features
    elif is_backed:
        # The chunked branch above already computed the union of positive
        # feature columns.
        detected_features = int(detected_features)
    else:
        if gene_expression_sums is None:
            gene_expression_sums = np.asarray(mat.sum(axis=0), dtype=np.float64).ravel()
        if scipy.sparse.issparse(mat):
            # Count columns with at least one *positive* value, matching the
            # dense and streamed paths.  Looking only at sparse structure is
            # subtly wrong for explicit zero entries (and for negative values
            # in processed matrices): such columns are stored but not detected
            # genes.  ``getnnz`` on the boolean comparison avoids materializing
            # row/column coordinate arrays while preserving the semantics.
            detected_features = int(np.count_nonzero((mat > 0).getnnz(axis=0)))
        else:
            # For dense matrix
            detected_features = int(np.any(mat > 0, axis=0).sum())

    if not qc_available:
        pct_mito = np.zeros(n_cells)
    elif precomputed_mito is not None:
        pct_mito = precomputed_mito
    else:
        if streamed_mito_counts is not None:
            mito_counts = streamed_mito_counts
        elif mito_mask.any():
            mito_counts = np.asarray(mat[:, mito_mask].sum(axis=1)).flatten()
        else:
            mito_counts = np.zeros(n_cells)
        pct_mito = np.divide(
            mito_counts * 100.0,
            counts_per_cell,
            out=np.zeros(n_cells, dtype=float),
            where=counts_per_cell > 0,
        )
    if qc_available:
        counts_array = np.asarray(counts_per_cell, dtype=np.float64)
        genes_array = np.asarray(genes_per_cell, dtype=np.float64)
        mito_array = np.asarray(pct_mito, dtype=np.float64)
        if (
            counts_array.shape != (n_cells,)
            or genes_array.shape != (n_cells,)
            or mito_array.shape != (n_cells,)
            or not np.isfinite(counts_array).all()
            or not np.isfinite(genes_array).all()
            or not np.isfinite(mito_array).all()
            or (counts_array < 0).any()
            or (genes_array < 0).any()
            or (mito_array < 0).any()
            or (counts_array > np.finfo(np.float32).max).any()
            or (mito_array > np.finfo(np.float32).max).any()
            or not np.equal(genes_array, np.floor(genes_array)).all()
            or (genes_array > np.iinfo(np.uint32).max).any()
        ):
            raise RuntimeError("H5AD QC metadata contains invalid, non-finite, or negative values")
    if gene_expression_sums is not None:
        gene_expression_sums = np.asarray(gene_expression_sums, dtype=np.float64)
        if gene_expression_sums.shape != (n_features,) or not np.isfinite(gene_expression_sums).all():
            raise RuntimeError("H5AD gene expression totals are invalid")
    # Keep the h5ad IPC response columnar, just like native MTX metadata. A
    # million-cell list of three-key Python dictionaries is expensive both
    # to construct and to stringify; typed columns preserve the same values
    # while avoiding one object allocation per cell on both sides of IPC.
    qc_n_genes = genes_array.astype(np.uint32).tolist() if qc_available else None
    qc_n_counts = counts_array.astype(np.float32).tolist() if qc_available else None
    qc_pct_mito = np.round(mito_array, 4).astype(np.float32).tolist() if qc_available else None

    result = {
        "cell_count": n_cells,
        "feature_count": detected_features,
        "total_features": n_features,
        "gene_names": gene_names,
        "gene_expression_sums": gene_expression_sums.tolist() if gene_expression_sums is not None else None,
        "barcodes": barcodes,
        "qc_metrics": [],
        "qc_n_genes": qc_n_genes,
        "qc_n_counts": qc_n_counts,
        "qc_pct_mito": qc_pct_mito,
        "sample_info": category_info(adata.obs, "sample" if sample_column else "sample"),
        "group_info": category_info(adata.obs, "group"),
        "imported_analysis": None,
    }

    # Multi-group support: collect all group columns (group + group_*).
    primary_group = result["group_info"]
    group_info_list = []
    if primary_group:
        group_info_list.append({"column": "group", "name": humanize_obs_name(configured_groups[0]) if configured_groups else "group", "info": primary_group})
    for col in adata.obs.columns:
        if col == "group" or not col.startswith("group_"):
            continue
        info = category_info(adata.obs, col)
        if info:
            group_index = int(col.removeprefix("group_") or "0") - 1
            source_name = configured_groups[group_index] if 0 <= group_index < len(configured_groups) else col
            group_info_list.append({"column": col, "name": humanize_obs_name(source_name), "info": info})
    if group_info_list:
        result["group_info_list"] = group_info_list
    if import_config and import_config.get("use_existing"):
        umap_key = import_config.get("umap_key")
        cluster_column = import_config.get("cluster_column")
        positions_umap = embedding_xy(adata, umap_key)
        cluster = category_info(adata.obs, cluster_column)
        if positions_umap is None or cluster is None:
            raise ValueError("Existing-analysis import requires a valid embedding and clustering column")
        tsne_key = import_config.get("tsne_key")
        annotation_column = import_config.get("annotation_column")
        result["imported_analysis"] = {
            "positions_umap": positions_umap,
            "positions_tsne": embedding_xy(adata, tsne_key),
            "clustering_name": cluster_column,
            "cluster_info": cluster,
            "annotation_name": annotation_column or None,
            "annotation_info": category_info(adata.obs, annotation_column),
        }
    return result


def get_gene_expression(adata, gene_names, gene_name: str):
    import numpy as np
    import scipy.sparse
    import time

    start = time.time()

    # Find gene index (case-insensitive)
    gene_lower = gene_name.lower()
    target_idx = next(
        (i for i, name in enumerate(gene_names) if name.lower() == gene_lower),
        None,
    )

    if target_idx is None:
        print(f"Gene {gene_name} not found", file=sys.stderr)
        sys.exit(1)

    # Use the same matrix source as metadata/QC so gene names and values agree.
    mat, selected_gene_names, source_label = select_expression_source(adata)
    gene_names = selected_gene_names
    target_idx = next((i for i, name in enumerate(gene_names) if name.lower() == gene_lower), None)
    if target_idx is None:
        print(f"Gene {gene_name} not found", file=sys.stderr)
        sys.exit(1)
    if scipy.sparse.issparse(mat):
        # For sparse matrix: extract column directly using CSC format for efficient column access
        if not scipy.sparse.issparse(mat) or mat.format != 'csc':
            mat_csc = mat.tocsc()
        else:
            mat_csc = mat
        column = mat_csc[:, target_idx].toarray().flatten()
    else:
        column = np.asarray(mat[:, target_idx]).flatten()

    raw_vals = column.astype(np.float32)
    max_value = float(raw_vals.max()) if raw_vals.size else 0.0
    print(f"[PYTHON]   {source_label}: max={max_value:.6f}", file=sys.stderr)

    elapsed = time.time() - start
    print(
        f"[PYTHON] Gene {gene_name} expression loaded in {elapsed:.3f}s (max={max_value:.2f})",
        file=sys.stderr,
    )

    return raw_vals.tolist()


# Global cache for adata to avoid re-reading h5ad file
_adata_cache = {}
_gene_names_cache = {}
_barcodes_cache = {}


def get_cached_adata(h5ad_path: str, backed: bool = False):
    """Get adata from cache or load it."""
    cache_key = (h5ad_path, bool(backed))
    if cache_key not in _adata_cache:
        adata, gene_names, barcodes = load_h5ad(h5ad_path, backed=backed)
        _adata_cache[cache_key] = adata
        _gene_names_cache[cache_key] = gene_names
        _barcodes_cache[cache_key] = barcodes
    return _adata_cache[cache_key], _gene_names_cache[cache_key], _barcodes_cache[cache_key]


def main():
    if len(sys.argv) < 2:
        print("Usage: read_h5ad_meta.py <h5ad_path> [--gene <name>]", file=sys.stderr)
        sys.exit(1)

    # Normalize at the command boundary as well as inside load_h5ad(). The
    # inspect path reads h5ad directly in backed mode, so normalizing only in
    # load_h5ad() would still pass a Rust-generated ``\\?\`` path to h5py.
    h5ad_path = normalize_windows_path(sys.argv[1])
    if not os.path.isfile(h5ad_path):
        # If it's a directory, find the h5ad file
        # APFS and NTFS do not guarantee the same readdir order.  Selecting
        # the first h5ad must therefore be lexical, otherwise two platforms
        # can analyse different files from the same directory.
        for fname in sorted(os.listdir(h5ad_path), key=lambda name: (name.casefold(), name)):
            if fname.lower().endswith(".h5ad"):
                h5ad_path = os.path.join(h5ad_path, fname)
                break

    if "--inspect" in sys.argv:
        # Inspection only needs obs/obsm metadata, not the expression matrix.
        # Use backed mode to avoid loading X into memory (which can be GBs for
        # large datasets and cause OOM before inspection finishes).
        import anndata as ad

        try:
            inspect_adata = ad.read_h5ad(h5ad_path, backed="r")
        except Exception:
            inspect_adata, _genes, _barcodes = get_cached_adata(h5ad_path)
        print(json.dumps(inspect_h5ad(inspect_adata), allow_nan=False))
        if getattr(inspect_adata, "file", None) is not None:
            inspect_adata.file.close()
        return

    # QC only needs obs/var plus chunked matrix access. Keep X backed so a
    # merged million-cell h5ad is not loaded into memory a second time after
    # the merge writer has just finished. Gene-expression requests retain the
    # historical in-memory path because they extract a full column.
    gene_requested = "--gene" in sys.argv
    if not gene_requested:
        emit_progress(2, "Opening h5ad metadata")
    adata, gene_names, barcodes = get_cached_adata(h5ad_path, backed=not gene_requested)

    if gene_requested:
        gene_idx = sys.argv.index("--gene")
        if gene_idx + 1 >= len(sys.argv):
            print("--gene requires a gene name", file=sys.stderr)
            sys.exit(1)
        gene_name = sys.argv[gene_idx + 1]
        values = get_gene_expression(adata, gene_names, gene_name)
        print(json.dumps(values, allow_nan=False))
    else:
        import_config = None
        if "--import-config" in sys.argv:
            config_idx = sys.argv.index("--import-config")
            if config_idx + 1 >= len(sys.argv):
                print("--import-config requires JSON", file=sys.stderr)
                sys.exit(1)
            import_config = json.loads(sys.argv[config_idx + 1])
        mito_prefixes = ["MT"]
        if "--mito-prefix" in sys.argv:
            mp_idx = sys.argv.index("--mito-prefix")
            if mp_idx + 1 < len(sys.argv):
                mito_prefixes = parse_mito_prefixes(sys.argv[mp_idx + 1])
        result = get_qc_metadata(
            adata,
            gene_names,
            barcodes,
            import_config,
            mito_prefixes,
            progress_callback=emit_progress,
        )
        emit_progress(100, "QC metadata ready")
        print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
