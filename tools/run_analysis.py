#!/usr/bin/env python3
"""
DNBCScope scanpy analysis pipeline.
Normalisation → HVG → PCA → KNN → UMAP → Leiden resolution sweep.

Usage:
    python run_analysis.py <mtx_dir> <cell_indices_json> <output_dir> <config_json>
"""

import sys
import json
import importlib.util
import hashlib
import math
import os
import platform
import time
import warnings

# The Rust launcher supplies a bounded thread budget for the desktop.  Direct
# script invocations keep a conservative single-thread default, while packaged
# runs can use several cores without allowing BLAS/Numba to take over the
# machine.  These must be set before importing numpy/numba/scanpy (scanpy is
# imported lazily below) to take effect.
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

warnings.filterwarnings("ignore")

PIPELINE_CONTRACT = "scanpy-leiden-v2"

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

try:
    from expression_source import (
        normalize_windows_path,
        read_10x_mtx_flexible,
        select_expression_matrix,
    )
    from native_csr import load_native_csr
    from process_metrics import peak_rss_bytes
    from runtime_fingerprint import build_runtime_fingerprint
except ImportError as error:
    raise RuntimeError(
        "DNBCScope could not load expression_source.py next to the analysis script. "
        "Rebuild the Windows package so the shared expression helper is included."
    ) from error


def log(msg: str) -> None:
    print(msg, flush=True, file=sys.stderr)


def progress(step: str, percent: int, message: str) -> None:
    print(f"DNBC_PROGRESS|{step}|{percent}|{message}", flush=True, file=sys.stderr)


def elapsed(started: float) -> str:
    return f"{time.monotonic() - started:.1f}s"


def _config_int(config, key, default, minimum, maximum):
    """Read a finite integer config value with a bounded safety contract.

    The Rust IPC layer validates the same fields, but the script is also used
    directly by a few repair/debug workflows.  Do not let a NaN, fractional
    value, bool, or unbounded integer reach Scanpy where it could turn into a
    surprising allocation or a much slower algorithm.
    """
    raw = config.get(key, default)
    if isinstance(raw, bool):
        raise RuntimeError(f"{key} must be an integer")
    if isinstance(raw, int):
        value = raw
    elif isinstance(raw, float) and math.isfinite(raw) and raw.is_integer():
        value = int(raw)
    else:
        raise RuntimeError(f"{key} must be a finite integer")
    if not minimum <= value <= maximum:
        raise RuntimeError(f"{key} must be between {minimum} and {maximum}")
    return value


def _config_float(config, key, default, minimum, maximum):
    raw = config.get(key, default)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise RuntimeError(f"{key} must be a finite number")
    value = float(raw)
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise RuntimeError(f"{key} must be a finite value between {minimum} and {maximum}")
    return value


def _config_bool(config, key, default):
    raw = config.get(key, default)
    if not isinstance(raw, bool):
        raise RuntimeError(f"{key} must be true or false")
    return raw


def apply_sample_column_alias(adata, raw_sample_column):
    """Map a configured obs column onto the canonical ``sample`` field."""
    sample_column = "" if raw_sample_column is None else str(raw_sample_column).strip()
    if not sample_column:
        return ""
    if sample_column not in adata.obs.columns:
        raise RuntimeError(
            f"Configured sample column is missing from h5ad obs: {sample_column}"
        )
    if sample_column != "sample":
        adata.obs["sample"] = adata.obs[sample_column].copy()
    return sample_column


def apply_group_column_aliases(adata, raw_group_columns):
    """Map selected source obs columns to canonical group/group_N fields."""
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


def runtime_diagnostics() -> dict:
    """Record the scientific runtime used for an analysis cache.

    macOS and Windows legitimately use different wheels (and often different
    BLAS/LAPACK implementations).  Keeping this small fingerprint beside the
    result makes a cross-platform discrepancy diagnosable instead of guessing
    from the installer version alone.
    """
    fingerprint = build_runtime_fingerprint()
    packages = fingerprint["packages"]
    determinism = fingerprint["determinism"]
    return {
        "python": platform.python_version(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "packages": packages,
        "determinism": determinism,
        "compatibility_key": fingerprint["compatibility_key"],
        "peak_rss_bytes": peak_rss_bytes(),
    }


def _sha256_files(output_dir, filenames):
    digest = hashlib.sha256()
    for filename in sorted(filenames):
        path = os.path.join(output_dir, filename)
        digest.update(filename.encode("utf-8"))
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _summarize_f32_file(path, chunk_bytes=4 * 1024 * 1024):
    """Summarize a little-endian f32 file without materializing it.

    Projection files are normally only a few megabytes, but a multi-million
    cell run can make UMAP/t-SNE large enough that the old ``read()`` call
    briefly duplicated the entire result near the end of analysis.  Keep a
    small carry buffer so arbitrary chunk boundaries are handled exactly like
    the previous byte-oriented implementation (trailing incomplete bytes are
    ignored).
    """
    import numpy as np

    if chunk_bytes < 4:
        raise ValueError("chunk_bytes must be at least four bytes")
    chunk_bytes -= chunk_bytes % 4
    carry = b""
    value_count = 0
    finite = True
    minimum = None
    maximum = None
    with open(path, "rb") as handle:
        while True:
            raw = handle.read(chunk_bytes)
            if not raw:
                break
            payload = carry + raw
            usable = len(payload) - (len(payload) % 4)
            if usable:
                values = np.frombuffer(payload, dtype="<f4", count=usable // 4)
                value_count += int(values.size)
                finite = finite and bool(np.isfinite(values).all())
                if values.size:
                    chunk_min = float(values.min())
                    chunk_max = float(values.max())
                    minimum = chunk_min if minimum is None else min(minimum, chunk_min)
                    maximum = chunk_max if maximum is None else max(maximum, chunk_max)
            carry = payload[usable:]
    return {
        "values": value_count,
        "finite": finite,
        "min": round(minimum, 5) if minimum is not None else None,
        "max": round(maximum, 5) if maximum is not None else None,
    }


def result_parity_diagnostics(output_dir, clusterings, n_cells, has_tsne=False):
    """Return exact and semantic signatures for cross-platform comparisons.

    Exact projection bytes can legitimately differ by a few floating-point
    ulps between Accelerate/MKL.  Cluster size profiles and finite/range
    summaries are therefore included as the stable parity signal; the exact
    hash remains useful for detecting a completely different result.
    """
    cluster_files = [f"clusters_{key}.u8" for key in sorted(clusterings)]
    projection_files = ["positions_umap.f32"] + (["positions_tsne.f32"] if has_tsne else [])
    all_files = projection_files + cluster_files + ["active_cell_indices.u32"]
    projection_summary = {}
    for filename in projection_files:
        projection_summary[filename] = _summarize_f32_file(
            os.path.join(output_dir, filename)
        )
    return {
        "exactSha256": _sha256_files(output_dir, all_files),
        "clusterSha256": _sha256_files(output_dir, cluster_files),
        "semantic": {
            "cellCount": int(n_cells),
            "clusterSizeProfiles": {
                key: sorted(int(count) for count in value["cluster_counts"])
                for key, value in sorted(clusterings.items())
            },
            "projection": projection_summary,
        },
    }


def _validated_projection(values, name, n_cells, np):
    """Convert one embedding to the on-disk f32 contract after validating it."""
    try:
        projection = np.asarray(values, dtype="<f4")
    except (TypeError, ValueError, OverflowError) as error:
        raise RuntimeError(f"{name} embedding cannot be converted to float32: {error}") from error
    expected_shape = (int(n_cells), 2)
    if projection.shape != expected_shape:
        raise RuntimeError(
            f"{name} embedding has shape {projection.shape}; expected {expected_shape}"
        )
    # Check after conversion as well as before it: a very large finite f64
    # value can become +/-inf when materialized as the compact f32 cache.
    if not np.isfinite(projection).all():
        raise RuntimeError(f"{name} embedding contains non-finite float32 coordinates")
    return projection


def _load_json_argument(value):
    if value.startswith("@"):
        with open(value[1:], encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(value)


def _read_u32_indices(path, np):
    """Read the binary cell-index sidecar without silently truncating it.

    ``numpy.fromfile`` drops a trailing partial element. That would turn a
    corrupted sidecar into a different, valid-looking analysis selection.
    Rust writes fixed-width U32 records, so reject any malformed byte length
    before handing the indices to the scientific pipeline.
    """
    item_size = np.dtype("<u4").itemsize
    size = os.path.getsize(path)
    if size % item_size != 0:
        raise RuntimeError(
            f"Selected cell-index file has invalid byte length ({size}); expected 4-byte records"
        )
    return np.fromfile(path, dtype="<u4")


def choose_preview_cell_indices(adata, cell_indices, max_cells):
    """Deterministically choose a representative preview subset.

    Sampling is proportional within the sample column when available so a
    smaller sample is not accidentally lost from a multi-sample preview.
    Returned indices stay sorted to keep backed h5ad slicing efficient.
    """
    import numpy as np

    indices = np.asarray(cell_indices, dtype=np.uint32)
    max_cells = max(500, int(max_cells))
    if indices.size <= max_cells:
        return indices

    rng = np.random.default_rng(0)
    selected_positions = []
    if "sample" in adata.obs.columns:
        labels = adata.obs.iloc[indices.astype(np.intp)]["sample"].astype(str).to_numpy()
        _, group_ids = np.unique(labels, return_inverse=True)
        counts = np.bincount(group_ids)
        ideal = counts.astype(np.float64) * (max_cells / indices.size)
        quotas = np.floor(ideal).astype(np.int64)
        if max_cells >= counts.size:
            quotas = np.maximum(quotas, 1)
        quotas = np.minimum(quotas, counts)
        while quotas.sum() > max_cells:
            candidates = np.flatnonzero(quotas > 1)
            if candidates.size == 0:
                candidates = np.flatnonzero(quotas > 0)
            quotas[candidates[np.argmax(quotas[candidates] - ideal[candidates])]] -= 1
        order = np.argsort(-(ideal - quotas), kind="stable")
        remaining = max_cells - int(quotas.sum())
        for group_id in order:
            if remaining == 0:
                break
            if quotas[group_id] < counts[group_id]:
                quotas[group_id] += 1
                remaining -= 1
        for group_id, quota in enumerate(quotas):
            positions = np.flatnonzero(group_ids == group_id)
            selected_positions.append(rng.choice(positions, size=int(quota), replace=False))
        positions = np.concatenate(selected_positions)
    else:
        positions = rng.choice(indices.size, size=max_cells, replace=False)
    return np.sort(indices[np.asarray(positions, dtype=np.intp)])


def minimize_adata_for_analysis(adata, cell_indices=None):
    """Keep one expression matrix and required metadata to reduce memory.

    AnnData.raw commonly contains log-normalized values, despite its name. Prefer
    an explicit counts layer, then a count-like X/raw matrix. If no count matrix
    exists, retain X as already-processed expression and do not normalize it again.
    """
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    matrix, var_names, normalize_input, source_label, _source_kind = (
        select_expression_matrix(adata, allow_negative=True)
    )

    # Slice only the selected matrix and retained obs columns.  Calling
    # ``adata[cell_indices].copy()`` first duplicates every h5ad layer, raw
    # matrix and embedding, even though this pipeline never uses them.
    # Avoiding that broad copy is especially important when QC re-analysis
    # keeps a subset of a large, richly annotated h5ad file.
    if cell_indices is not None:
        cell_indices = np.asarray(cell_indices, dtype=np.intp)
        is_full_selection = (
            cell_indices.size == adata.n_obs
            and (
                cell_indices.size == 0
                or np.array_equal(
                    cell_indices,
                    np.arange(cell_indices.size, dtype=np.intp),
                )
            )
        )
        if is_full_selection:
            source_obs = adata.obs
        else:
            # Backed h5ad datasets require increasing row indices for fancy
            # indexing. Keep the caller's order in ``source_obs`` and restore
            # it after slicing so the source-index column remains aligned.
            order = np.argsort(cell_indices, kind="stable")
            sorted_indices = cell_indices[order]
            # Slice a backed matrix before converting it to CSR. Converting
            # first would materialize the complete h5ad matrix even when QC
            # retained only a subset of cells.
            matrix = matrix[sorted_indices]
            restore_order = np.argsort(order, kind="stable")
            matrix = matrix[restore_order]
            source_obs = adata.obs.iloc[cell_indices]
    else:
        source_obs = adata.obs

    # Backed sparse datasets expose ``to_memory``; dense h5py datasets are
    # converted through numpy. At this point only the selected rows remain.
    if not sp.issparse(matrix):
        if hasattr(matrix, "to_memory"):
            matrix = matrix.to_memory()
        else:
            matrix = np.asarray(matrix)
    if not sp.issparse(matrix):
        matrix = sp.csr_matrix(matrix)
    elif matrix.format != "csr":
        matrix = matrix.tocsr()

    keep_obs = {}
    for key in source_obs.columns:
        # Keep the sample column, the primary "group" column, every extra
        # multi-group dimension (group_*), and the source index bookmark.
        if key in ("sample", "group") or key.startswith("group_"):
            keep_obs[key] = source_obs[key].copy()
    if "_dnbc_source_index" in source_obs.columns:
        keep_obs["_dnbc_source_index"] = source_obs["_dnbc_source_index"].copy()
    obs = pd.DataFrame(keep_obs, index=source_obs.index.copy())
    var = pd.DataFrame(index=var_names)
    result = ad.AnnData(X=matrix, obs=obs, var=var)
    result.uns["_dnbc_normalize_input"] = normalize_input
    result.uns["_dnbc_expression_source"] = source_label
    return result


def main() -> None:
    if len(sys.argv) != 5:
        print(
            "Usage: run_analysis.py <mtx_dir> <cell_indices_json> <output_dir> <config_json>",
            file=sys.stderr,
        )
        sys.exit(1)

    mtx_dir = normalize_windows_path(sys.argv[1])
    cell_indices_path = sys.argv[2]
    output_dir = sys.argv[3]
    config = _load_json_argument(sys.argv[4])
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Imports (deferred so startup errors surface early)
    # ------------------------------------------------------------------ #
    progress("startup", 3, "Loading Scanpy and scientific libraries")
    import numpy as np
    import scanpy as sc

    sc.settings.verbosity = 0
    sc.settings.set_figure_params(dpi=80)
    hvg_flavor = str(config.get("hvgFlavor", "seurat"))
    if hvg_flavor not in {"seurat", "cell_ranger", "seurat_v3"}:
        raise RuntimeError(f"Unsupported HVG flavor: {hvg_flavor}")
    n_hvgs = max(200, _config_int(config, "nHvgs", 2000, 1, 200_000))
    requested_pcs = max(2, _config_int(config, "nPcs", 30, 1, 500))
    requested_neighbors = max(2, _config_int(config, "nNeighbors", 15, 2, 5_000))
    raw_batch_correction = config.get("batchCorrection", "none")
    batch_correction = "none" if raw_batch_correction is None else str(raw_batch_correction)
    if batch_correction not in {"none", "harmony", "combat", "bbknn"}:
        raise RuntimeError(f"Unsupported batch correction: {batch_correction}")
    min_genes = _config_int(config, "minGenes", 200, 0, 10_000_000)
    min_cells = _config_int(config, "minCells", 3, 0, 10_000_000)
    max_pct_mito = _config_float(config, "maxPctMito", 15, 0, 100)
    # A QC re-analysis passes the exact cells selected in the QC panel. In that
    # mode, applying the creation-time cell filters again would make the
    # analysis result disagree with the visible preview. Keep feature-level
    # filtering (`minCells`) below, but skip cell-level filters explicitly.
    apply_cell_qc_filters = _config_bool(config, "applyCellQcFilters", True)
    # 线粒体基因前缀：逗号分隔多前缀，大小写不敏感匹配。默认 "MT" 是
    # MT-/MT./MT_ 的简写，不匹配 MTHFD1、MTOR 这类非线粒体基因。
    mito_prefix_raw = str(config.get("mitoPrefix", "MT")).strip()
    mito_prefixes = [p.strip() for p in mito_prefix_raw.split(",") if p.strip()]
    if not mito_prefixes:
        mito_prefixes = ["MT"]
    has_batches = False
    if hvg_flavor == "seurat_v3" and importlib.util.find_spec("skmisc") is None:
        hvg_flavor = "seurat"
        progress("hvg", 5, "scikit-misc unavailable; using Seurat HVG flavor")

    # ------------------------------------------------------------------ #
    #  1. Load data (supports both MTX directories and h5ad files)
    # ------------------------------------------------------------------ #
    progress("load", 10, "Reading expression data")
    backed_input = False
    if os.path.isfile(mtx_dir) and mtx_dir.lower().endswith(".json"):
        adata = load_native_csr(mtx_dir, include_barcodes=False)
        log(f"  → Opened native CSR: {adata.n_obs} cells, {adata.n_vars} genes")
    elif os.path.isfile(mtx_dir) and mtx_dir.lower().endswith(".h5ad"):
        import anndata as ad
        adata = ad.read_h5ad(mtx_dir, backed="r")
        backed_input = True
        log(f"  → Loaded h5ad: {adata.n_obs} cells, {adata.n_vars} genes")
    elif os.path.isdir(mtx_dir):
        # Find .h5ad file in directory
        h5ad_file = None
        # Directory enumeration order differs between APFS and NTFS.  A
        # directory may contain more than one h5ad (for example an original
        # and a processed export); always choose the same lexical candidate
        # on every platform instead of whichever entry the OS returns first.
        for fname in sorted(os.listdir(mtx_dir), key=lambda name: (name.casefold(), name)):
            if fname.lower().endswith(".h5ad"):
                h5ad_file = os.path.join(mtx_dir, fname)
                break
        if h5ad_file:
            import anndata as ad
            adata = ad.read_h5ad(h5ad_file, backed="r")
            backed_input = True
            log(f"  → Loaded h5ad from directory: {adata.n_obs} cells, {adata.n_vars} genes")
        else:
            adata = read_10x_mtx_flexible(mtx_dir)
            log(f"  → Loaded MTX: {adata.n_obs} cells, {adata.n_vars} genes")
    else:
        raise RuntimeError(f"Input path is neither an h5ad file nor an MTX directory: {mtx_dir}")

    # The import UI may map any categorical h5ad obs column (for example
    # ``orig.ident`` or ``batch``) to the application's canonical sample
    # dimension. Alias it before preview sampling and minimisation so every
    # downstream step uses the same sample labels without rewriting the
    # source h5ad file.
    sample_column = apply_sample_column_alias(adata, config.get("sampleColumn"))
    selected_group_columns = apply_group_column_aliases(adata, config.get("groupColumns"))

    if cell_indices_path.lower().endswith(".u32"):
        cell_indices = _read_u32_indices(cell_indices_path, np)
    else:
        # Backwards-compatible path for standalone/debug invocations.
        with open(cell_indices_path, encoding="utf-8") as fh:
            cell_indices = np.asarray(json.load(fh))
        if cell_indices.ndim != 1:
            raise RuntimeError("Selected cell indices must be a one-dimensional array")
        if cell_indices.size:
            if not np.issubdtype(cell_indices.dtype, np.integer):
                if not np.issubdtype(cell_indices.dtype, np.floating) or not np.isfinite(cell_indices).all():
                    raise RuntimeError("Selected cell indices must be finite integers")
                if not np.equal(cell_indices, np.floor(cell_indices)).all():
                    raise RuntimeError("Selected cell indices must be integers")
            if (cell_indices < 0).any() or (cell_indices > np.iinfo(np.uint32).max).any():
                raise RuntimeError("Selected cell indices are outside the U32 range")
        cell_indices = cell_indices.astype(np.uint32, copy=False)

    n_total = adata.n_obs
    source_cell_indices = np.asarray(cell_indices, dtype=np.uint32)
    if source_cell_indices.ndim != 1:
        raise RuntimeError("Selected cell indices must be a one-dimensional array")
    if source_cell_indices.size and int(source_cell_indices.max()) >= n_total:
        raise RuntimeError(
            f"Selected cell index {int(source_cell_indices.max())} is outside the input ({n_total} cells)"
        )
    if np.unique(source_cell_indices).size != source_cell_indices.size:
        raise RuntimeError("Selected cell indices contain duplicates")
    analysis_mode = str(config.get("analysisMode", "full"))
    if analysis_mode not in {"full", "fast_preview"}:
        raise RuntimeError(f"Unsupported analysis mode: {analysis_mode}")
    preview_max_cells = _config_int(config, "previewMaxCells", 100_000, 500, 5_000_000)
    if analysis_mode == "fast_preview" and source_cell_indices.size > preview_max_cells:
        original_selected_count = int(source_cell_indices.size)
        source_cell_indices = choose_preview_cell_indices(
            adata, source_cell_indices, preview_max_cells
        )
        progress(
            "preview",
            14,
            f"Fast preview: sampling {source_cell_indices.size:,} of "
            f"{original_selected_count:,} selected cells",
        )
        log(
            "  → Fast preview uses a deterministic representative subset; "
            "run Full analysis for publication-grade coordinates"
        )
    source_adata = adata
    try:
        adata = minimize_adata_for_analysis(source_adata, source_cell_indices)
    finally:
        if backed_input:
            source_adata.file.close()
        del source_adata
    adata.obs["_dnbc_source_index"] = source_cell_indices
    normalize_input = bool(adata.uns.pop("_dnbc_normalize_input", True))
    expression_source = adata.uns.pop("_dnbc_expression_source", "X")
    log(f"  → Expression source: {expression_source}")
    progress("filter", 20, f"Filtering cells: {adata.n_obs} / {n_total} retained")

    if adata.n_obs < 3:
        raise RuntimeError(
            f"Too few cells after filtering ({adata.n_obs}). Need at least 3."
        )

    if batch_correction == "harmony" and "sample" in adata.obs.columns:
        try:
            import harmonypy  # noqa: F401
        except ImportError:
            batch_correction = "combat"
            progress("batch", 24, "Harmony unavailable; using ComBat batch correction")
    if "sample" in adata.obs.columns:
        has_batches = adata.obs["sample"].nunique() > 1
    if batch_correction != "none" and not has_batches:
        log("  → Batch correction skipped: fewer than two samples")
        batch_correction = "none"

    # ------------------------------------------------------------------ #
    #  1b. Gene/cell quality filtering
    # ------------------------------------------------------------------ #
    if apply_cell_qc_filters and min_genes > 0:
        sc.pp.filter_cells(adata, min_genes=min_genes)
        progress("filter", 22, f"Cell quality filtering retained {adata.n_obs} cells")

    if adata.n_obs < 3:
        raise RuntimeError(
            f"Too few cells after quality filtering ({adata.n_obs}). Need at least 3."
        )

    if min_cells > 0:
        before = adata.n_vars
        sc.pp.filter_genes(adata, min_cells=min_cells)
        log(f"  → filter_genes(min_cells={min_cells}): {before} → {adata.n_vars} genes")
    if adata.n_vars < 3:
        raise RuntimeError(
            f"Too few genes after filtering ({adata.n_vars}). Need at least 3 for PCA."
        )

    # Mitochondrial filtering — 使用用户配置的前缀，大小写不敏感匹配
    if apply_cell_qc_filters and max_pct_mito > 0:
        var_names_upper = adata.var_names.str.upper()
        mito_genes = False
        for prefix in mito_prefixes:
            normalized_prefix = prefix.upper()
            if normalized_prefix == "MT":
                matched = var_names_upper.str.startswith(("MT-", "MT.", "MT_"))
            else:
                matched = var_names_upper.str.startswith(normalized_prefix)
            mito_genes = mito_genes | matched
        if mito_genes.any():
            adata.var["mt"] = mito_genes
            sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
            before = adata.n_obs
            adata = adata[adata.obs["pct_counts_mt"] <= max_pct_mito].copy()
            log(f"  → mito filter (prefixes={mito_prefixes}, max {max_pct_mito}%): {before} → {adata.n_obs} cells")
        else:
            log(f"  → mito filter skipped: no genes matched prefixes {mito_prefixes}")

    if adata.n_obs < 3:
        raise RuntimeError(
            f"Too few cells after mitochondrial filtering ({adata.n_obs}). Need at least 3."
        )

    # ------------------------------------------------------------------ #
    #  2. Preprocessing
    # ------------------------------------------------------------------ #
    if hvg_flavor == "seurat_v3" and not normalize_input:
        hvg_flavor = "seurat"
        progress("hvg", 26, "Count matrix unavailable; using Seurat HVG on existing expression")

    if hvg_flavor == "seurat_v3":
        progress("hvg", 28, "Selecting highly variable genes from raw counts")
        started = time.monotonic()
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=min(n_hvgs, adata.n_vars),
            flavor=hvg_flavor,
            batch_key="sample" if has_batches else None,
        )
        log(f"  → HVG selection completed in {elapsed(started)}")

    if normalize_input:
        progress("normalize", 34, "Normalizing counts and applying log1p")
        started = time.monotonic()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        log(f"  → Normalization completed in {elapsed(started)}")
    else:
        progress("normalize", 34, "Using existing processed expression without re-normalizing")

    if hvg_flavor != "seurat_v3":
        progress("hvg", 38, "Selecting highly variable genes")
        started = time.monotonic()
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=min(n_hvgs, adata.n_vars),
            flavor=hvg_flavor,
            batch_key="sample" if has_batches else None,
        )
        log(f"  → HVG selection completed in {elapsed(started)}")

    if "highly_variable" in adata.var.columns:
        n_hvg = int(adata.var["highly_variable"].sum())
        log(f"  → {n_hvg} HVGs selected")
        if n_hvg < 3:
            raise RuntimeError(
                f"Too few highly variable genes selected ({n_hvg}). Need at least 3 for PCA."
            )
        adata_hvg = adata[:, adata.var["highly_variable"]].copy()
    else:
        adata_hvg = adata
    if adata_hvg is not adata:
        del adata

    if batch_correction == "combat" and "sample" in adata_hvg.obs.columns:
        progress("batch", 44, "Applying ComBat batch correction to highly variable genes")
        sc.pp.combat(adata_hvg, key="sample")

    # Centering a sparse cells × HVGs matrix makes it dense.  At 500k cells and
    # 2,000 HVGs that one intermediate alone is roughly 3.7 GiB (float32), and
    # PCA needs additional working memory.  Keep the established centered PCA
    # path for ordinary projects, but preserve sparsity above a conservative
    # memory budget.  Scaling without centering still standardizes features;
    # the matching non-centred randomized SVD is the resource-safe PCA variant
    # for very large single-cell matrices.
    dense_scale_bytes = adata_hvg.n_obs * adata_hvg.n_vars * 4
    use_sparse_pca = dense_scale_bytes > 512 * 1024 * 1024
    if use_sparse_pca:
        progress("scale", 48, "Scaling HVGs with memory-safe sparse PCA")
        log(
            "  → Using sparse PCA path; centered scale would require "
            f"at least {dense_scale_bytes / 1024**3:.1f} GiB"
        )
        started = time.monotonic()
        sc.pp.scale(adata_hvg, zero_center=False, max_value=10)
        log(f"  → Sparse scaling completed in {elapsed(started)}")
    else:
        progress("scale", 48, "Scaling highly variable genes")
        started = time.monotonic()
        sc.pp.scale(adata_hvg, max_value=10)
        log(f"  → Scaling completed in {elapsed(started)}")

    # ------------------------------------------------------------------ #
    #  3. PCA
    # ------------------------------------------------------------------ #
    n_cells = adata_hvg.n_obs
    n_genes_hvg = adata_hvg.n_vars
    n_comps = min(requested_pcs, n_cells - 1, n_genes_hvg - 1)
    n_comps = max(2, n_comps)

    progress("pca", 58, f"Computing PCA ({n_comps} components)")
    started = time.monotonic()
    sc.pp.pca(
        adata_hvg,
        n_comps=n_comps,
        svd_solver="randomized",
        zero_center=not use_sparse_pca,
        random_state=0,
    )
    log(f"  → PCA completed in {elapsed(started)}")

    if batch_correction == "harmony" and "sample" in adata_hvg.obs.columns:
        progress("batch", 64, "Applying Harmony batch correction")
        import harmonypy

        started = time.monotonic()
        pca = np.asarray(adata_hvg.obsm["X_pca"], dtype=np.float64)
        harmony_out = harmonypy.run_harmony(
            pca,
            adata_hvg.obs,
            "sample",
            verbose=False,
            random_state=0,
        )
        corrected = np.asarray(harmony_out.Z_corr)
        if corrected.shape == pca.shape:
            adata_hvg.obsm["X_pca"] = corrected
        elif corrected.T.shape == pca.shape:
            adata_hvg.obsm["X_pca"] = corrected.T
        else:
            raise RuntimeError(
                f"Harmony returned shape {corrected.shape}; expected {pca.shape}"
            )
        progress("batch", 66, f"Harmony completed in {elapsed(started)}")

    # ------------------------------------------------------------------ #
    #  4. Neighbour graph + UMAP
    # ------------------------------------------------------------------ #
    n_neighbors = min(requested_neighbors, n_cells - 1)
    neighbor_transformer = (
        "bbknn"
        if batch_correction == "bbknn" and "sample" in adata_hvg.obs.columns
        else ("pynndescent" if n_cells >= 10_000 else "exact")
    )

    if batch_correction == "bbknn" and "sample" in adata_hvg.obs.columns:
        # BBKNN modifies the neighbor graph directly (batch-balanced KNN)
        progress("batch", 68, "Applying BBKNN batch correction")
        started = time.monotonic()
        import bbknn
        bbknn.bbknn(
            adata_hvg,
            batch_key="sample",
            n_pcs=n_comps,
            n_neighbors=n_neighbors,
        )
        log(f"  → BBKNN completed in {elapsed(started)}")
    else:
        progress("neighbors", 68, f"Building nearest-neighbor graph (k={n_neighbors})")
        started = time.monotonic()
        sc.pp.neighbors(
            adata_hvg,
            n_pcs=n_comps,
            n_neighbors=n_neighbors,
            use_rep="X_pca",
            transformer="pynndescent" if neighbor_transformer == "pynndescent" else None,
            random_state=0,
        )
        log(f"  → Neighbor graph completed in {elapsed(started)}")

    umap_min_dist = _config_float(config, "umapMinDist", 0.3, 0, 10)
    umap_spread = _config_float(config, "umapSpread", 1.2, 0.01, 100)
    compute_tsne = _config_bool(config, "computeTsne", False)

    progress("umap", 74, "Computing UMAP embedding")
    started = time.monotonic()
    sc.tl.umap(adata_hvg, min_dist=umap_min_dist, spread=umap_spread, random_state=0)
    log(f"  → UMAP completed in {elapsed(started)}")

    if compute_tsne:
        progress("tsne", 76, "Computing t-SNE embedding")
        started = time.monotonic()
        # Use sklearn directly for better control over t-SNE parameters
        from sklearn.manifold import TSNE as SklearnTSNE
        n_cells = adata_hvg.n_obs
        perplexity = min(30.0, max(1.0, (n_cells - 1) / 3.0), n_cells - 1e-3)
        X_pca = adata_hvg.obsm["X_pca"][:, :n_comps]
        tsne = SklearnTSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate="auto",
            init="pca",
            max_iter=1000,
            random_state=0,
        )
        adata_hvg.obsm["X_tsne"] = tsne.fit_transform(X_pca)
        log(f"  → t-SNE completed in {elapsed(started)} (perplexity={perplexity:.1f})")

    # ------------------------------------------------------------------ #
    #  5. Clustering resolution sweep
    # ------------------------------------------------------------------ #
    resolutions = [round(i / 10, 1) for i in range(1, 21)]
    clusterings = {}
    started = time.monotonic()
    for index, resolution in enumerate(resolutions):
        percent = 84 + int(index * 12 / len(resolutions))
        key = f"{resolution:.1f}"
        progress("clustering", percent, f"Running Leiden clustering (resolution {key})")
        try:
            sc.tl.leiden(
                adata_hvg,
                flavor="igraph",
                n_iterations=2,
                resolution=resolution,
                key_added=f"leiden_{key}",
                random_state=0,
            )
            labels = adata_hvg.obs[f"leiden_{key}"].astype(str).values
        except Exception as error:
            # A one-cluster fallback makes a failed scientific step look like
            # a valid result and can silently change every downstream marker
            # and annotation. Fail the run with the backend error instead;
            # packaged environments are expected to contain igraph/leidenalg.
            raise RuntimeError(
                f"Leiden failed at resolution {key}; check the bundled "
                f"igraph/leidenalg runtime: {error}"
            ) from error

        unique_clusters, inverse, raw_counts = np.unique(
            labels, return_inverse=True, return_counts=True
        )
        if len(unique_clusters) > 255:
            raise RuntimeError(
                f"Resolution {key} produced {len(unique_clusters)} clusters; maximum is 255"
            )
        sorted_indices = np.argsort(-raw_counts, kind="stable")
        remap = np.empty(len(sorted_indices), dtype=np.uint8)
        remap[sorted_indices] = np.arange(len(sorted_indices), dtype=np.uint8)
        cluster_ids = remap[inverse]
        if cluster_ids.ndim != 1 or cluster_ids.size != adata_hvg.n_obs:
            raise RuntimeError(
                f"Resolution {key} produced {cluster_ids.size} cluster ids for "
                f"{adata_hvg.n_obs} cells"
            )
        sorted_counts_raw = raw_counts[sorted_indices]
        if cluster_ids.size and int(cluster_ids.max()) >= len(sorted_counts_raw):
            raise RuntimeError(f"Resolution {key} produced an out-of-range cluster id")
        if int(raw_counts.sum()) != int(adata_hvg.n_obs):
            raise RuntimeError(f"Resolution {key} cluster counts do not sum to the cell count")
        cluster_ids.tofile(os.path.join(output_dir, f"clusters_{key}.u8"))
        sorted_names = [f"Cluster {i + 1}" for i in range(len(unique_clusters))]
        sorted_counts = sorted_counts_raw.astype(int).tolist()
        clusterings[key] = {
            "cluster_names": sorted_names,
            "cluster_counts": sorted_counts,
        }
    log(f"  → Leiden resolution sweep completed in {elapsed(started)}")

    # ------------------------------------------------------------------ #
    #  6. Write outputs
    # ------------------------------------------------------------------ #
    progress("write", 96, "Writing analysis results")
    umap_f32 = _validated_projection(adata_hvg.obsm["X_umap"], "UMAP", n_cells, np)
    umap_f32.tofile(os.path.join(output_dir, "positions_umap.f32"))

    has_tsne = "X_tsne" in adata_hvg.obsm
    if has_tsne:
        tsne_f32 = _validated_projection(adata_hvg.obsm["X_tsne"], "t-SNE", n_cells, np)
        tsne_f32.tofile(os.path.join(output_dir, "positions_tsne.f32"))

    # Metadata
    active_cell_indices = adata_hvg.obs["_dnbc_source_index"].to_numpy(dtype="<u4")
    if active_cell_indices.ndim != 1 or active_cell_indices.size != n_cells:
        raise RuntimeError(
            "Analysis source-index output is not one-dimensional or does not match the cell count"
        )
    if np.unique(active_cell_indices).size != active_cell_indices.size:
        raise RuntimeError("Analysis source-index output contains duplicate cells")
    active_cell_indices.tofile(os.path.join(output_dir, "active_cell_indices.u32"))
    runtime = runtime_diagnostics()
    runtime["parity"] = result_parity_diagnostics(
        output_dir, clusterings, n_cells, has_tsne=has_tsne
    )
    meta = {
        "pipeline_contract": PIPELINE_CONTRACT,
        "n_cells": int(n_cells),
        "algorithm": "leiden",
        "clusterings": clusterings,
        "has_binary_active_cell_indices": True,
        "has_tsne": has_tsne,
        "params": {
            "analysisMode": analysis_mode,
            "previewMaxCells": preview_max_cells,
            "hvgFlavor": hvg_flavor,
            "nHvgs": n_hvgs,
            "nPcs": requested_pcs,
            "nNeighbors": requested_neighbors,
            "minGenes": min_genes,
            "minCells": min_cells,
            "maxPctMito": max_pct_mito,
            "applyCellQcFilters": apply_cell_qc_filters,
            "mitoPrefix": mito_prefix_raw,
            "umapMinDist": umap_min_dist,
            "umapSpread": umap_spread,
            "computeTsne": compute_tsne,
            "batchCorrection": batch_correction,
            "sampleColumn": sample_column or None,
            "neighborTransformer": neighbor_transformer,
        },
        "runtime": runtime,
    }

    # Extract sample and group info if available
    sample_info = _category_info(adata_hvg, "sample", np)
    if sample_info:
        meta["sample_info"] = sample_info

    group_info = _category_info(adata_hvg, "group", np)
    if group_info:
        meta["group_info"] = group_info

    # Multi-group support: collect all group columns (group + group_*).
    # The first entry is always the primary "group" column. Each entry
    # records its obs column name so the frontend can label dimensions.
    group_display_names = (
        config.get("groupDisplayNames") or config.get("group_display_names") or {}
    )
    group_info_list = []
    if group_info:
        group_info_list.append({
            "column": "group",
            "name": group_display_names.get(
                "group", selected_group_columns[0] if selected_group_columns else "group"
            ),
            "info": group_info,
        })
    for col in adata_hvg.obs.columns:
        if col == "group" or not col.startswith("group_"):
            continue
        info = _category_info(adata_hvg, col, np)
        if info:
            group_index = int(col.removeprefix("group_") or "0") - 1
            group_info_list.append({
                "column": col,
                "name": group_display_names.get(
                    col,
                    selected_group_columns[group_index]
                    if 0 <= group_index < len(selected_group_columns)
                    else col,
                ),
                "info": info,
            })
    if group_info_list:
        meta["group_info_list"] = group_info_list

    with open(os.path.join(output_dir, "analysis_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, allow_nan=False)

    # Write sample/group IDs as binary
    if sample_info:
        np.array(sample_info["ids"], dtype=np.uint8).tofile(
            os.path.join(output_dir, "sample_ids.u8")
        )
    if group_info:
        np.array(group_info["ids"], dtype=np.uint8).tofile(
            os.path.join(output_dir, "group_ids.u8")
        )
    # Write each extra group dimension's ids under <column>.u8 for cache reload.
    for entry in group_info_list[1:]:
        col = entry["column"]
        np.array(entry["info"]["ids"], dtype=np.uint8).tofile(
            os.path.join(output_dir, f"{col}_ids.u8")
        )

    # The desktop reads binary outputs and analysis_meta.json directly. Avoid
    # serializing a second million-element result to stdout only for Rust to
    # discard it.
    progress("complete", 100, f"Complete: {n_cells} cells, UMAP + {len(resolutions)} resolutions")


def _sort_key(s: str):
    """Sort cluster labels numerically when possible."""
    try:
        return (0, int(s))
    except ValueError:
        return (1, s)


def _category_info(adata, column: str, np):
    if column not in adata.obs.columns:
        return None
    names, ids, counts = np.unique(
        adata.obs[column].astype(str).to_numpy(),
        return_inverse=True,
        return_counts=True,
    )
    # Category ids are uint8 on the IPC boundary, so ids 0–255 support
    # exactly 256 categories. Keep this aligned with the h5ad import path.
    if len(names) > 256:
        raise RuntimeError(f"{column} contains more than 256 categories")
    return {
        "names": names.tolist(),
        "ids": ids.astype(np.uint8).tolist(),
        "counts": counts.astype(int).tolist(),
    }


if __name__ == "__main__":
    main()
