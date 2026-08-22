"""Shared expression-matrix selection for every DNBCScope Python pipeline.

The module deliberately validates a layer named ``counts`` instead of trusting
its name. Real-world h5ad files sometimes store normalized values there; treating
those values as counts would normalize/log-transform them a second time.
"""

import os


def normalize_windows_path(path):
    """Drop an unnecessary ``\\\\?\\`` prefix before calling Python wheels.

    Rust uses the Windows extended-length form for genuinely long paths.  It
    is accepted by Win32, but older h5py/HDF5 and joblib builds reject the same
    prefix for ordinary paths and report the misleading WinError 206.  Keep
    the prefix when the logical path is still long; only shorten paths that
    fit the legacy boundary so packaged and system Python behave identically.
    """
    if os.name != "nt" or not isinstance(path, str) or not path.startswith("\\\\?\\"):
        return path
    if path.startswith("\\\\?\\UNC\\"):
        logical = "\\\\" + path[8:]
    else:
        logical = path[4:]
    # Rust applies the same boundary using `str::encode_utf16().count()`.
    # Python's `len()` counts Unicode code points, so it under-counts non-BMP
    # characters (for example emoji, which occupy two UTF-16 code units on
    # Windows).  Keep the extended prefix whenever the Windows length reaches
    # the same threshold; otherwise h5py can receive a path that still hits
    # WinError 206 after the prefix was stripped.
    windows_length = len(logical.encode("utf-16-le", errors="surrogatepass")) // 2
    return logical if windows_length < 240 else path


def _sample_values(matrix, max_values=100_000):
    import numpy as np
    import scipy.sparse as sp

    # Backed h5ad datasets should not be materialized just to classify the
    # matrix. Sample several bounded blocks across both dimensions.
    if not sp.issparse(matrix) and hasattr(matrix, "shape") and len(matrix.shape) == 2:
        n_rows, n_cols = int(matrix.shape[0]), int(matrix.shape[1])
        if n_rows * n_cols > max_values:
            block_rows = min(n_rows, 128)
            block_cols = min(n_cols, 128)
            row_starts = np.linspace(0, max(0, n_rows - block_rows), num=min(4, max(1, n_rows)), dtype=int)
            col_starts = np.linspace(0, max(0, n_cols - block_cols), num=min(4, max(1, n_cols)), dtype=int)
            samples = []
            for row_start in sorted(set(row_starts.tolist())):
                for col_start in sorted(set(col_starts.tolist())):
                    block = matrix[
                        row_start:row_start + block_rows,
                        col_start:col_start + block_cols,
                    ]
                    values = block.data if sp.issparse(block) else np.asarray(block).ravel()
                    if values.size:
                        samples.append(np.asarray(values).ravel())
            if not samples:
                return np.asarray([], dtype=np.float64)
            values = np.concatenate(samples)
            if values.size > max_values:
                values = values[:max_values]
            return np.asarray(values, dtype=np.float64)

    # anndata's backed sparse dataset is a lightweight proxy rather than a
    # scipy sparse matrix. Small proxies can be materialized safely; large
    # ones already took the bounded block-sampling path above.
    if hasattr(matrix, "to_memory"):
        matrix = matrix.to_memory()
    values = matrix.data if sp.issparse(matrix) else np.asarray(matrix).ravel()
    if values.size > max_values:
        # Evenly-spaced indices avoid the systematic blind spots caused by a
        # fixed stride when invalid values follow a periodic pattern.
        indices = np.linspace(0, values.size - 1, num=max_values, dtype=np.int64)
        values = values[indices]
    return np.asarray(values, dtype=np.float64)


def matrix_is_count_like(matrix, max_values=100_000):
    import numpy as np

    values = _sample_values(matrix, max_values)
    return bool(
        values.size == 0
        or (
            np.isfinite(values).all()
            and (values >= 0).all()
            and np.allclose(values, np.rint(values), atol=1e-6, rtol=0.0)
        )
    )


def matrix_is_nonnegative(matrix, max_values=100_000):
    import numpy as np

    values = _sample_values(matrix, max_values)
    return bool(values.size == 0 or (np.isfinite(values).all() and (values >= 0).all()))


def _expression_candidates(adata):
    raw = getattr(adata, "raw", None)
    candidates = []
    if "counts" in adata.layers:
        candidates.append((adata.layers["counts"], adata.var_names, "layers['counts']", "counts"))
    candidates.append((adata.X, adata.var_names, "X", "x"))
    if raw is not None:
        candidates.append((raw.X, raw.var_names, "raw.X", "raw"))
    return candidates


def select_count_matrix(adata):
    """Return a validated raw count matrix and its metadata."""
    for matrix, names, label, kind in _expression_candidates(adata):
        if matrix_is_count_like(matrix):
            return matrix, names, f"{label} (validated counts)", kind
    raise RuntimeError(
        "No raw non-negative integer count matrix found in layers['counts'], X, or raw.X."
    )


def select_expression_matrix(adata, allow_negative=False):
    """Return ``(matrix, var_names, is_counts, source_label, source_kind)``."""

    raw = getattr(adata, "raw", None)
    candidates = _expression_candidates(adata)

    try:
        matrix, names, label, kind = select_count_matrix(adata)
        return matrix, names, True, label, kind
    except RuntimeError:
        pass

    # Prefer raw processed expression, then a non-negative counts layer/X.
    processed_order = []
    if raw is not None:
        processed_order.append((raw.X, raw.var_names, "raw.X", "raw"))
    if "counts" in adata.layers:
        processed_order.append((adata.layers["counts"], adata.var_names, "layers['counts']", "counts"))
    processed_order.append((adata.X, adata.var_names, "X", "x"))
    for matrix, names, label, kind in processed_order:
        if matrix_is_nonnegative(matrix):
            return matrix, names, False, f"{label} (processed)", kind

    if allow_negative:
        return adata.X, adata.var_names, False, "X (scaled/centered)", "x"
    raise RuntimeError(
        "No usable expression matrix found. Expected raw counts or non-negative "
        "processed expression in layers['counts'], X, or raw.X."
    )


def make_expression_adata(adata, allow_negative=False, cell_indices=None):
    """Build a minimal in-memory AnnData for a downstream analysis.

    Backed inputs are sliced before materialization so differential expression
    and annotation do not load unrelated cells, layers, embeddings, or obs
    columns from a large h5ad.
    """
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    matrix, names, is_counts, label, _kind = select_expression_matrix(
        adata, allow_negative=allow_negative
    )
    if cell_indices is not None:
        raw_indices = np.asarray(cell_indices)
        if raw_indices.ndim != 1:
            raise RuntimeError("Cell indices must be one-dimensional")
        if raw_indices.size:
            if not np.issubdtype(raw_indices.dtype, np.integer):
                if (
                    not np.issubdtype(raw_indices.dtype, np.floating)
                    or not np.isfinite(raw_indices).all()
                    or not np.equal(raw_indices, np.floor(raw_indices)).all()
                ):
                    raise RuntimeError("Cell indices must be finite integers")
            if (raw_indices < 0).any():
                raise RuntimeError("Cell indices must be non-negative")
            if raw_indices.max() > np.iinfo(np.intp).max:
                raise RuntimeError("Cell indices exceed the platform index range")
        indices = raw_indices.astype(np.intp, copy=False)
        if indices.size and (indices.min() < 0 or indices.max() >= adata.n_obs):
            raise RuntimeError("Cell index is outside the expression matrix")

        # h5py-backed dense datasets require monotonically increasing fancy
        # indices. Read in source order and restore the caller's order after
        # only the selected rows have been materialized.
        order = np.argsort(indices, kind="stable")
        sorted_indices = indices[order]
        if sorted_indices.size > 1 and np.equal(sorted_indices[1:], sorted_indices[:-1]).any():
            raise RuntimeError("Cell indices contain duplicates")
        restore_order = np.argsort(order, kind="stable")
        matrix = matrix[sorted_indices]
        if hasattr(matrix, "to_memory"):
            matrix = matrix.to_memory()
        if not sp.issparse(matrix):
            matrix = np.asarray(matrix)
        matrix = matrix[restore_order]

    if hasattr(matrix, "to_memory"):
        matrix = matrix.to_memory()
    if sp.issparse(matrix) and matrix.format != "csr":
        matrix = matrix.tocsr()

    row_count = int(matrix.shape[0])
    result = ad.AnnData(
        X=matrix,
        obs=pd.DataFrame(index=pd.RangeIndex(row_count).astype(str)),
        var=pd.DataFrame(index=names.copy()),
    )
    return result, is_counts, label


def read_10x_mtx_flexible(mtx_dir):
    """Tolerant 10x MTX loader accepting any combination of:

    - features.tsv(.gz) (v3+) or genes.tsv(.gz) (legacy v2)
    - compressed or uncompressed independently for matrix / features / barcodes

    Replaces sc.read_10x_mtx, which only looks for features.tsv.gz (or
    features.tsv) based on a single `compressed` flag derived from
    matrix.mtx.gz existence — failing when matrix is gzipped but features.tsv
    is not, or when only legacy genes.tsv.gz is present.
    """
    import gzip
    import os
    import scipy.io as scio
    import pandas as pd

    mtx_dir = normalize_windows_path(mtx_dir)

    def _find(names, label):
        for name in names:
            p = os.path.join(mtx_dir, name)
            if os.path.exists(p):
                return p
        raise RuntimeError(f"No {label} found in {mtx_dir}: tried {', '.join(names)}")

    matrix_path = _find(["matrix.mtx.gz", "matrix.mtx"], "matrix file")
    features_path = _find(
        ["features.tsv.gz", "features.tsv", "genes.tsv.gz", "genes.tsv"],
        "features/genes file",
    )
    barcodes_path = _find(["barcodes.tsv.gz", "barcodes.tsv"], "barcodes file")

    # Read matrix (scipy mmread accepts gzip file objects)
    if matrix_path.endswith(".gz"):
        with gzip.open(matrix_path, "rb") as fh:
            M = scio.mmread(fh, spmatrix=True)
    else:
        M = scio.mmread(matrix_path, spmatrix=True)

    # Cell Ranger mtx is (n_genes, n_cells); transpose to (n_cells, n_genes)
    # MTX ingestion only needs the AnnData container. Importing Scanpy here
    # added its full cold-start cost to multi-sample merge and Scrublet input
    # preparation, even though no Scanpy operation was used yet.
    import anndata as ad

    adata = ad.AnnData(M.T.tocsr())

    # Read features: prefer column 1 (gene_symbols in v3+), fallback column 0
    # 10x metadata is UTF-8. Never inherit the Windows process code page
    # (often GBK), otherwise a non-ASCII sample/gene label fails before the
    # scientific pipeline even starts.
    feat_opener = gzip.open if features_path.endswith(".gz") else open
    with feat_opener(features_path, "rt", encoding="utf-8", errors="replace") as fh:
        # Gene symbols and barcodes are identifiers, not nullable numeric
        # data.  Pandas' default NA vocabulary turns a literal symbol such as
        # ``NA`` into a missing value and later ``astype(str)`` changes it to
        # ``nan``, making expression lookup silently miss that feature.
        feat_df = pd.read_csv(fh, sep="\t", header=None, keep_default_na=False)
    if feat_df.shape[1] >= 2:
        var_names = feat_df[1].astype(str).values
        adata.var["gene_ids"] = feat_df[0].astype(str).values
    else:
        var_names = feat_df[0].astype(str).values
    # Deduplicate gene names (append .1, .2, ...) to match scanpy behavior
    if len(set(var_names)) != len(var_names):
        seen = {}
        uniq = []
        for n in var_names:
            if n in seen:
                seen[n] += 1
                uniq.append(f"{n}.{seen[n]}")
            else:
                seen[n] = 0
                uniq.append(n)
        var_names = uniq
    adata.var_names = var_names

    # Read barcodes
    bc_opener = gzip.open if barcodes_path.endswith(".gz") else open
    with bc_opener(barcodes_path, "rt", encoding="utf-8", errors="replace") as fh:
        obs_names = pd.read_csv(fh, header=None, keep_default_na=False)[0].astype(str).values
    adata.obs_names = obs_names

    return adata
