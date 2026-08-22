"""Disk-backed CSR source shared by DNBCScope's Python pipelines.

Large all-MTX multi-sample imports persist these buffers directly and defer
H5AD serialization until the user explicitly exports an H5AD file.
"""

from __future__ import annotations

import json
import os

from expression_source import normalize_windows_path


NATIVE_CSR_FORMAT = "dnbcscope-native-csr-v2"
NATIVE_CSR_FORMATS = {"dnbcscope-native-csr-v1", NATIVE_CSR_FORMAT}


def is_native_csr_manifest(path: str) -> bool:
    path = normalize_windows_path(path)
    return os.path.isfile(path) and path.lower().endswith(".json")


def _resolved(manifest_path: str, value: object) -> str:
    path = normalize_windows_path(str(value))
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(manifest_path), path)
    return os.path.normpath(path)


def _open_array(path: str, dtype: str):
    """Open a fixed-width sidecar, including a valid zero-length buffer."""
    import numpy as np

    byte_size = os.path.getsize(path)
    item_size = np.dtype(dtype).itemsize
    if byte_size % item_size:
        raise RuntimeError(f"Native CSR sidecar has an unaligned byte length: {path}")
    if byte_size == 0:
        # np.memmap cannot represent an empty file, but an all-zero matrix is
        # a valid CSR with empty indices/data and a non-empty indptr buffer.
        return np.empty(0, dtype=dtype)
    return np.memmap(path, dtype=dtype, mode="r")


def _categorical_from_ranges(cell_count, samples, values):
    import numpy as np
    import pandas as pd

    categories = []
    category_index = {}
    codes = np.empty(cell_count, dtype=np.int32)
    cursor = 0
    for sample, value in zip(samples, values):
        start, end = int(sample["start"]), int(sample["end"])
        if start != cursor or end < start or end > cell_count:
            raise RuntimeError("Native CSR sample ranges are not contiguous")
        normalized = str(value) if value else "unknown"
        code = category_index.get(normalized)
        if code is None:
            code = len(categories)
            if code >= 256:
                raise RuntimeError("Native CSR metadata exceeds 256 categories")
            categories.append(normalized)
            category_index[normalized] = code
        codes[start:end] = code
        cursor = end
    if cursor != cell_count:
        raise RuntimeError("Native CSR sample ranges do not cover every cell")
    return pd.Categorical.from_codes(codes, categories=categories)


def load_native_csr(path: str, *, as_anndata: bool = True, include_barcodes: bool = True):
    """Open a validated native CSR manifest without copying its value buffers."""
    import numpy as np
    import scipy.sparse as sp

    manifest_path = normalize_windows_path(path)
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise RuntimeError("Native CSR manifest must be a JSON object")
    if manifest.get("format") not in (None, *NATIVE_CSR_FORMATS):
        raise RuntimeError(f"Unsupported native CSR format: {manifest.get('format')}")

    try:
        raw_cell_count = manifest["cell_count"]
        if isinstance(raw_cell_count, bool):
            raise ValueError
        cell_count = int(raw_cell_count)
        if cell_count != raw_cell_count:
            raise ValueError
        raw_gene_names = manifest["gene_names"]
        if not isinstance(raw_gene_names, list):
            raise ValueError
        gene_names = [str(value) for value in raw_gene_names]
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise RuntimeError("Native CSR manifest has invalid dimensions") from error
    # Keep malformed/untrusted manifests from requesting unbounded allocations
    # before the Rust IPC boundary has a chance to reject them.
    if cell_count <= 0 or cell_count > 10_000_000 or not gene_names or len(gene_names) > 10_000_000:
        raise RuntimeError("Native CSR manifest has invalid dimensions")
    # Older native-source manifests omitted ``data_dtype`` and used the
    # explicit ``*.data.f64`` filename. Keep those projects readable while
    # merged manifests continue to declare float32 explicitly.
    dtype_name = str(
        manifest.get(
            "data_dtype",
            "float64" if str(manifest.get("data_path", "")).lower().endswith(".f64") else "float32",
        )
    )
    dtype = {"float32": "<f4", "float64": "<f8"}.get(dtype_name)
    if dtype is None:
        raise RuntimeError(f"Unsupported native CSR data dtype: {dtype_name}")

    indptr_dtype = {"int32": "<i4", "int64": "<i8"}.get(
        str(manifest.get("indptr_dtype", "int64"))
    )
    indices_dtype = {"int32": "<i4", "uint32": "<u4"}.get(
        str(manifest.get("indices_dtype", "uint32"))
    )
    if indptr_dtype is None or indices_dtype is None:
        raise RuntimeError("Native CSR manifest has unsupported index dtypes")
    indptr = _open_array(_resolved(manifest_path, manifest["indptr_path"]), indptr_dtype)
    indices = _open_array(_resolved(manifest_path, manifest["indices_path"]), indices_dtype)
    data = _open_array(_resolved(manifest_path, manifest["data_path"]), dtype)
    if len(indptr) != cell_count + 1 or len(indices) != len(data):
        raise RuntimeError("Native CSR buffers have inconsistent lengths")
    # Even a v2 manifest marked as validated must have a monotone row pointer:
    # SciPy trusts the CSR structure and malformed offsets can otherwise make a
    # later row slice read arbitrary bytes or crash the worker. This is an
    # O(cell_count) check and does not scan the much larger nnz buffers on the
    # normal trusted path.
    if (
        int(indptr[0]) != 0
        or int(indptr[-1]) != len(data)
        or np.any(indptr < 0)
        or np.any(indptr[1:] < indptr[:-1])
    ):
        raise RuntimeError("Native CSR indptr does not match the data length")
    trusted_buffers = (
        manifest.get("format") == NATIVE_CSR_FORMAT
        and manifest.get("buffers_validated") is True
    )
    if not trusted_buffers and len(data) and (
        int(indices.min()) < 0 or int(indices.max()) >= len(gene_names)
    ):
        raise RuntimeError("Native CSR contains an out-of-range gene index")
    matrix = sp.csr_matrix(
        (data, indices, indptr), shape=(cell_count, len(gene_names)), copy=False
    )

    barcode_path = _resolved(manifest_path, manifest["barcodes_path"])
    barcodes = None
    if include_barcodes:
        with open(barcode_path, encoding="utf-8") as handle:
            barcodes = [line.rstrip("\r\n") for line in handle]
        if len(barcodes) != cell_count:
            raise RuntimeError("Native CSR barcode count does not match cell_count")
    if not as_anndata:
        return matrix, gene_names, barcodes, manifest

    import anndata as ad
    import pandas as pd

    # Analysis workers do not need barcode strings. A RangeIndex keeps their
    # AnnData lightweight while metadata/export callers can opt into the
    # barcode sidecar with ``include_barcodes=True``.
    if barcodes is None:
        obs = pd.DataFrame(index=pd.RangeIndex(cell_count))
    else:
        obs = pd.DataFrame(index=pd.Index(barcodes, dtype="object"))
    samples = list(manifest.get("samples") or [])
    if samples:
        sample_values = [str(sample.get("name") or "unknown") for sample in samples]
        sample_category = _categorical_from_ranges(cell_count, samples, sample_values)
        obs["sample_id"] = sample_category
        obs["sample"] = sample_category
        columns = [str(value) for value in manifest.get("group_columns") or ["group"]]
        for column_index, column in enumerate(columns):
            values = []
            for sample in samples:
                groups = list(sample.get("groups") or [])
                values.append(groups[column_index] if column_index < len(groups) else "unknown")
            obs[column] = _categorical_from_ranges(cell_count, samples, values)

    return ad.AnnData(
        X=matrix,
        obs=obs,
        var=pd.DataFrame(index=pd.Index(gene_names, dtype="object")),
    )


def load_native_csc(path: str):
    """Open the optional persistent gene-column index without copying it."""
    import numpy as np
    import scipy.sparse as sp

    manifest_path = normalize_windows_path(path)
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise RuntimeError("Native CSR manifest must be a JSON object")
    sidecar_keys = ("csc_indptr_path", "csc_indices_path", "csc_data_path")
    present = [isinstance(manifest.get(key), str) and bool(manifest.get(key)) for key in sidecar_keys]
    if not any(present):
        return None
    if not all(present):
        raise RuntimeError("Native CSC manifest is incomplete")

    cell_count = int(manifest["cell_count"])
    gene_count = len(manifest["gene_names"])
    indptr_dtype = {"int32": "<i4", "int64": "<i8"}.get(
        str(manifest.get("csc_indptr_dtype", "int64"))
    )
    indices_dtype = {"int32": "<i4", "uint32": "<u4"}.get(
        str(manifest.get("indices_dtype", "int32"))
    )
    data_dtype = {"float32": "<f4", "float64": "<f8"}.get(
        str(manifest.get("data_dtype", "float32"))
    )
    if indptr_dtype is None or indices_dtype is None or data_dtype is None:
        raise RuntimeError("Native CSC manifest has unsupported dtypes")
    indptr = _open_array(_resolved(manifest_path, manifest["csc_indptr_path"]), indptr_dtype)
    indices = _open_array(_resolved(manifest_path, manifest["csc_indices_path"]), indices_dtype)
    data = _open_array(_resolved(manifest_path, manifest["csc_data_path"]), data_dtype)
    if len(indptr) != gene_count + 1 or len(indices) != len(data):
        raise RuntimeError("Native CSC buffers have inconsistent lengths")
    if (
        int(indptr[0]) != 0
        or int(indptr[-1]) != len(data)
        or np.any(indptr < 0)
        or np.any(indptr[1:] < indptr[:-1])
    ):
        raise RuntimeError("Native CSC indptr does not match the data length")
    trusted_buffers = (
        manifest.get("format") == NATIVE_CSR_FORMAT
        and manifest.get("buffers_validated") is True
    )
    if not trusted_buffers and len(indices) and (
        int(indices.min()) < 0 or int(indices.max()) >= cell_count
    ):
        raise RuntimeError("Native CSC contains an out-of-range cell index")
    return sp.csc_matrix(
        (data, indices, indptr), shape=(cell_count, gene_count), copy=False
    )
