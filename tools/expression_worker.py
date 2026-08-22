#!/usr/bin/env python3
"""Persistent JSON-lines worker for fast gene-expression column reads.

Memory optimization:
- Uses anndata backed='r' mode to keep matrix on disk (not fully loaded into memory)
- Byte-bounded LRU cache for recently accessed gene columns (default 64 MiB)
- For sparse matrices, only accessed columns are loaded into memory
"""

import json
import os
import sys
from collections import OrderedDict

# Keep direct execution deterministic with the resident scientific worker.
# Isolated/embeddable Windows Python does not always expose the script
# directory through sys.path, even when expression_source.py is beside it.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

try:
    from expression_source import (
        normalize_windows_path,
        read_10x_mtx_flexible,
        select_expression_matrix,
    )
except ImportError as error:
    raise RuntimeError(
        "DNBCScope could not load expression_source.py next to expression_worker.py"
    ) from error


def _nonnegative_env_int(name, default):
    try:
        return max(0, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        # A malformed inherited environment should not make the expression
        # worker disappear before it can report a useful error to the UI.
        return int(default)


# Cache compact float32 arrays, never Python ``list[float]`` objects. A Python
# float list costs far more than 4 bytes/value and made the old 200-column cap
# unsafe for million-cell projects. Allow deployments to tune the budget.
MAX_CACHE_BYTES = _nonnegative_env_int("DNBC_EXPRESSION_CACHE_BYTES", 64 * 1024 * 1024)

# Tuneable limits keep interactive h5ad files responsive without making the
# resident worker eagerly consume memory for large projects.
SMALL_H5AD_MAX_BYTES = _nonnegative_env_int(
    "DNBC_EXPRESSION_SMALL_H5AD_BYTES", 384 * 1024 * 1024
)
SMALL_H5AD_MAX_ELEMENTS = _nonnegative_env_int(
    "DNBC_EXPRESSION_SMALL_H5AD_ELEMENTS", 160_000_000
)
# A CSC copy is excellent for repeated interactive column reads, but keeping
# it beside a large mmap-backed CSR can create a dangerous memory spike.  The
# default covers PBMC-sized and ordinary desktop datasets while leaving very
# large projects on the bounded CSR fallback. Set to 0 to disable it.
NATIVE_CSC_MAX_NNZ = _nonnegative_env_int("DNBC_EXPRESSION_NATIVE_CSC_MAX_NNZ", 8_000_000)


def select_adata_source(adata):
    _matrix, names, _is_counts, _label, kind = select_expression_matrix(
        adata, allow_negative=True
    )
    return kind, names


def resolve_h5ad(path: str) -> str:
    path = normalize_windows_path(path)
    if os.path.isfile(path) and path.lower().endswith(".h5ad"):
        return path
    # Keep source selection deterministic across APFS/NTFS directory order.
    for name in sorted(os.listdir(path), key=lambda value: (value.casefold(), value)):
        if name.lower().endswith(".h5ad"):
            return os.path.join(path, name)
    raise RuntimeError(f"No h5ad file found in {path}")


def load_source(path: str, data_format: str):
    """Load expression data with memory optimization.
    
    For h5ad files, uses backed='r' mode to keep matrix on disk.
    Returns the matrix (or AnnData for backed mode), gene names, backed flag,
    and the selected AnnData source (counts/x/raw).
    """
    import scipy.sparse

    path = normalize_windows_path(path)

    if data_format == "native-csr":
        from native_csr import load_native_csr

        matrix, gene_names, _barcodes, _manifest = load_native_csr(
            path, as_anndata=False, include_barcodes=False
        )
        return matrix, gene_names, False, "native-csr"

    if data_format == "h5ad":
        import anndata as ad

        h5ad_path = resolve_h5ad(path)
        try:
            file_size = os.path.getsize(h5ad_path)
        except OSError:
            file_size = SMALL_H5AD_MAX_BYTES + 1

        # Probe the shape and expression source in backed mode first. Reading
        # a compressed dense h5ad directly into memory and checking its element
        # count afterwards defeats the memory guard: the dangerous allocation
        # has already happened by then. The lightweight probe keeps large
        # files backed while still allowing ordinary interactive files to be
        # materialized once for fast gene switching.
        backed_adata = None
        try:
            backed_adata = ad.read_h5ad(h5ad_path, backed="r")
            source_kind, gene_names = select_adata_source(backed_adata)
            can_materialize = (
                file_size <= SMALL_H5AD_MAX_BYTES
                and int(backed_adata.n_obs) * int(backed_adata.n_vars) <= SMALL_H5AD_MAX_ELEMENTS
            )
            if not can_materialize:
                return backed_adata, gene_names, True, source_kind

            backed_adata.file.close()
            backed_adata = None
            adata = ad.read_h5ad(h5ad_path)
            source_kind, gene_names = select_adata_source(adata)
            if source_kind == "counts":
                matrix = adata.layers["counts"]
            elif source_kind == "raw":
                matrix = adata.raw.X
            else:
                matrix = adata.X
            if scipy.sparse.issparse(matrix) and matrix.format != "csc":
                matrix = matrix.tocsc()
            return matrix, gene_names, False, source_kind
        except Exception:
            if backed_adata is not None:
                try:
                    backed_adata.file.close()
                except Exception:
                    pass
            # Fallback to in-memory loading if backed mode fails. This is only
            # a compatibility path for unusual h5ad encodings; normal files
            # have already been classified safely by the backed probe.
            adata = ad.read_h5ad(h5ad_path)
            source_kind, gene_names = select_adata_source(adata)
            if source_kind == "counts":
                matrix = adata.layers["counts"]
            elif source_kind == "raw":
                matrix = adata.raw.X
            else:
                matrix = adata.X
            if scipy.sparse.issparse(matrix) and matrix.format != "csc":
                matrix = matrix.tocsc()
            return matrix, gene_names, False, source_kind

    if data_format == "mtx":
        # Use the same tolerant 10x reader as Scanpy, differential expression,
        # and annotation.  The old local reader ignored barcodes and matrix
        # dimensions and kept duplicate gene symbols under their first lookup
        # spelling, so a gene query could address a different column than the
        # analysis that produced the clusters.
        adata = read_10x_mtx_flexible(path)
        matrix, gene_names, _is_counts, _label, source_kind = select_expression_matrix(
            adata, allow_negative=True
        )
        if scipy.sparse.issparse(matrix) and matrix.format != "csc":
            matrix = matrix.tocsc()
        return matrix, [str(name) for name in gene_names], False, source_kind

    raise RuntimeError(f"Unsupported expression source format: {data_format}")


def _maybe_build_native_csc(source, csc_ref):
    """Build one bounded native-CSR column view, or keep the CSR fallback."""
    if csc_ref[0] is False:
        return None
    if csc_ref[0] is not None or NATIVE_CSC_MAX_NNZ <= 0:
        return csc_ref[0]
    if getattr(source, "nnz", 0) > NATIVE_CSC_MAX_NNZ:
        # Remember the negative decision so every gene request does not repeat
        # the size check or accidentally attempt a later conversion.
        csc_ref[0] = False
        return None
    try:
        import scipy.sparse

        # copy=False still permits a format conversion, but avoids an
        # unnecessary second copy when a future native source is already CSC.
        csc_ref[0] = source if scipy.sparse.isspmatrix_csc(source) else source.tocsc(copy=False)
    except Exception:
        # A read-only mmap, an unusual SciPy build, or an allocation failure
        # must not make expression lookup unavailable. Keep the CSR path.
        csc_ref[0] = False
    return csc_ref[0] if csc_ref[0] is not False else None


def expression_values(
    source,
    gene_lookup,
    gene: str,
    column_cache: OrderedDict,
    cache_bytes: list[int],
    backed: bool,
    source_kind: str,
    native_csc_ref=None,
):
    """Get expression values for a gene.
    
    Uses LRU cache to avoid repeated disk reads in backed mode.
    """
    import numpy as np
    import scipy.sparse

    gene_lower = gene.lower()
    
    # Check cache first
    if gene_lower in column_cache:
        # Move to end (most recently used)
        column_cache.move_to_end(gene_lower)
        return column_cache[gene_lower]

    index = gene_lookup.get(gene_lower)
    if index is None:
        raise RuntimeError(f"Gene {gene} not found")

    if backed:
        # Backed mode: source is AnnData, read column from disk
        try:
            if source_kind == "counts":
                col = source.layers["counts"][:, index]
            elif source_kind == "raw":
                col = source.raw.X[:, index]
            else:
                col = source.X[:, index]
            
            if scipy.sparse.issparse(col):
                values = col.toarray().ravel()
            else:
                values = np.asarray(col).ravel()
        except Exception as e:
            raise RuntimeError(f"Failed to read gene {gene} from backed AnnData: {e}")
    elif source_kind == "native-csr":
        # Prefer the lazily-created bounded CSC view for small/medium native
        # projects.  For large matrices `_maybe_build_native_csc` returns None,
        # and SciPy's C-level CSR accessor remains the safe memory-bounded
        # fallback. In both cases only the requested dense vector enters the
        # LRU cache.
        csc_ref = native_csc_ref if native_csc_ref is not None else [None]
        column_source = _maybe_build_native_csc(source, csc_ref)
        column = (column_source[:, index] if column_source is not None else source[:, index])
        values = (
            column.toarray().ravel()
            if scipy.sparse.issparse(column)
            else np.asarray(column).ravel()
        )
    else:
        # In-memory mode: source is matrix
        if scipy.sparse.issparse(source):
            values = source[:, index].toarray().ravel()
        else:
            values = np.asarray(source[:, index]).ravel()

    result = np.asarray(values, dtype=np.float32).ravel()
    if not np.isfinite(result).all():
        raise RuntimeError(f"Expression values for {gene} contain non-finite values")
    
    # Add to cache with LRU eviction
    result_bytes = result.nbytes
    # A single over-budget column is still returned, but is not retained.
    if result_bytes <= MAX_CACHE_BYTES:
        while column_cache and cache_bytes[0] + result_bytes > MAX_CACHE_BYTES:
            _, evicted = column_cache.popitem(last=False)
            cache_bytes[0] -= evicted.nbytes
        column_cache[gene_lower] = result
        cache_bytes[0] += result_bytes
    
    return result


def emit(payload):
    sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def main():
    if len(sys.argv) != 3:
        raise RuntimeError("Usage: expression_worker.py <source_path> <h5ad|mtx|native-csr>")

    source_path, data_format = sys.argv[1], sys.argv[2]
    source, gene_names, backed, source_kind = load_source(source_path, data_format)
    # Large native sources may already contain a mmap-backed persistent CSC
    # sidecar built during import. Ordinary sources retain the bounded lazy
    # in-memory conversion below. Keep the public load_source contract stable.
    persistent_csc = None
    if source_kind == "native-csr":
        from native_csr import load_native_csc

        persistent_csc = load_native_csc(source_path)
    native_csc_ref = [persistent_csc]
    
    # Build gene lookup (case-insensitive)
    gene_lookup = {}
    for index, name in enumerate(gene_names):
        gene_lookup.setdefault(str(name).lower(), index)
    
    # LRU cache for column data
    column_cache = OrderedDict()
    cache_bytes = [0]
    
    # Get cell count
    if backed:
        cell_count = source.n_obs
        gene_count = len(gene_names)
    else:
        cell_count = source.shape[0]
        gene_count = source.shape[1]
    
    emit({"ok": True, "cells": cell_count, "genes": gene_count, "backed": backed, "source": source_kind})

    for line in sys.stdin:
        try:
            request = json.loads(line)
            command = request.get("command")
            if command == "close":
                emit({"ok": True})
                break
            if command == "clear_cache":
                column_cache.clear()
                cache_bytes[0] = 0
                emit({"ok": True, "cleared": True})
                continue
            if command != "gene":
                raise RuntimeError(f"Unsupported command: {command}")
            values = expression_values(
                source,
                gene_lookup,
                str(request["gene"]),
                column_cache,
                cache_bytes,
                backed,
                source_kind,
                native_csc_ref,
            )
            # Control messages remain JSON-lines, while expression values use a
            # compact length-prefixed f32 payload. This avoids creating a huge
            # Python ``list[float]`` and JSON string for every million-cell
            # request. Explicit little-endian keeps the Rust decoder portable.
            payload = values.astype("<f4", copy=False).tobytes(order="C")
            emit({"ok": True, "binary_bytes": len(payload)})
            sys.stdout.buffer.write(payload)
            sys.stdout.buffer.flush()
        except Exception as error:
            emit({"ok": False, "error": str(error)})


if __name__ == "__main__":
    main()
