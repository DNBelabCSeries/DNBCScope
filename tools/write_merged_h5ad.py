#!/usr/bin/env python3
"""Serialize Rust-prepared CSR buffers as a merged AnnData/H5AD source.

The expensive Matrix Market parsing, gene intersection and sparse alignment
already happened in Rust. This script deliberately remains a thin AnnData
container writer so DNBCScope keeps upstream H5AD compatibility without
moving large text matrices through Python.
"""

from __future__ import annotations

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
    from expression_source import normalize_windows_path
except ImportError as error:
    raise RuntimeError(
        "DNBCScope could not load expression_source.py next to write_merged_h5ad.py"
    ) from error


def emit_progress(percent: int, message: str) -> None:
    print(
        f"DNBC_MERGE_PROGRESS|{max(0, min(100, int(percent)))}|{message}",
        file=sys.stderr,
        flush=True,
    )


def categorical_from_ranges(cell_count: int, samples: list[dict], values: list[str]):
    categories: list[str] = []
    category_index: dict[str, int] = {}
    codes = np.empty(cell_count, dtype=np.int32)
    for sample, value in zip(samples, values):
        normalized = value or "unknown"
        code = category_index.get(normalized)
        if code is None:
            code = len(categories)
            categories.append(normalized)
            category_index[normalized] = code
        codes[int(sample["start"]) : int(sample["end"])] = code
    return pd.Categorical.from_codes(codes, categories=categories)


def main() -> None:
    if len(sys.argv) != 2:
        raise RuntimeError("Usage: write_merged_h5ad.py <native_merge_manifest.json>")
    manifest_path = normalize_windows_path(sys.argv[1])
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)

    started = time.perf_counter()
    cell_count = int(manifest["cell_count"])
    gene_names = [str(value) for value in manifest["gene_names"]]
    samples = list(manifest["samples"])
    if cell_count <= 0 or len(gene_names) < 2 or not samples:
        raise RuntimeError("Native merge manifest has invalid dimensions")

    open_started = time.perf_counter()
    emit_progress(76, "Opening native sparse merge buffers")
    indptr_dtype = {"int32": "<i4", "int64": "<i8"}.get(
        str(manifest.get("indptr_dtype", "int64"))
    )
    indices_dtype = {"int32": "<i4", "uint32": "<u4"}.get(
        str(manifest.get("indices_dtype", "uint32"))
    )
    if indptr_dtype is None or indices_dtype is None:
        raise RuntimeError("Native merge manifest has unsupported index dtypes")
    indptr = np.memmap(
        normalize_windows_path(manifest["indptr_path"]), dtype=indptr_dtype, mode="r"
    )
    indices = np.memmap(
        normalize_windows_path(manifest["indices_path"]), dtype=indices_dtype, mode="r"
    )
    data_dtypes = {"float32": "<f4", "float64": "<f8"}
    data_dtype_name = str(
        manifest.get(
            "data_dtype",
            "float64" if str(manifest.get("data_path", "")).lower().endswith(".f64") else "float32",
        )
    )
    data_dtype = data_dtypes.get(data_dtype_name)
    if data_dtype is None:
        raise RuntimeError(
            f"Native merge manifest has unsupported data dtype: {data_dtype_name}"
        )
    data = np.memmap(
        normalize_windows_path(manifest["data_path"]), dtype=data_dtype, mode="r"
    )
    if len(indptr) != cell_count + 1 or len(indices) != len(data):
        raise RuntimeError("Native merge CSR buffers have inconsistent lengths")
    if int(indptr[-1]) != len(data):
        raise RuntimeError("Native merge CSR indptr does not match the data length")
    matrix = sp.csr_matrix(
        (data, indices, indptr), shape=(cell_count, len(gene_names)), copy=False
    )
    open_seconds = time.perf_counter() - open_started

    metadata_started = time.perf_counter()
    with open(
        normalize_windows_path(manifest["barcodes_path"]), encoding="utf-8"
    ) as handle:
        barcodes = [line.rstrip("\r\n") for line in handle]
    if len(barcodes) != cell_count:
        raise RuntimeError(
            f"Native merge has {len(barcodes)} barcodes for {cell_count} cells"
        )

    obs = pd.DataFrame(index=pd.Index(barcodes, dtype="object"))
    del barcodes
    sample_names = [str(sample["name"]) for sample in samples]
    sample_category = categorical_from_ranges(
        cell_count, samples, sample_names
    )
    obs["sample_id"] = sample_category
    obs["sample"] = sample_category
    group_columns = [str(value) for value in manifest.get("group_columns") or ["group"]]
    for column_index, column in enumerate(group_columns):
        group_values = []
        for sample in samples:
            values = list(sample.get("groups") or [])
            group_values.append(
                str(values[column_index])
                if column_index < len(values) and values[column_index]
                else "unknown"
            )
        obs[column] = categorical_from_ranges(cell_count, samples, group_values)

    merged = ad.AnnData(
        X=matrix,
        obs=obs,
        var=pd.DataFrame(index=pd.Index(gene_names, dtype="object")),
    )
    metadata_seconds = time.perf_counter() - metadata_started
    output_path = normalize_windows_path(manifest["output_path"])
    partial_path = normalize_windows_path(manifest["partial_path"])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    write_started = time.perf_counter()
    emit_progress(84, "Writing merged H5AD compatibility source")
    try:
        merged.write_h5ad(partial_path, compression="lzf")
        os.replace(partial_path, output_path)
    finally:
        try:
            if os.path.exists(partial_path):
                os.remove(partial_path)
        except OSError:
            pass
    write_seconds = time.perf_counter() - write_started
    h5ad_bytes = os.path.getsize(output_path)
    prepare_seconds = float(manifest.get("prepare_seconds", 0.0))
    csr_bytes = int(manifest.get("csr_bytes", 0))

    groups = []
    for sample in samples:
        values = list(sample.get("groups") or [])
        groups.append(str(values[0]) if values and values[0] else "unknown")
    writer_seconds = time.perf_counter() - started
    emit_progress(
        100,
        f"Merge complete ({prepare_seconds + writer_seconds:.1f}s; "
        f"prepare {prepare_seconds:.1f}s; metadata {metadata_seconds:.1f}s; "
        f"write {write_seconds:.1f}s; {h5ad_bytes / (1024 ** 2):.1f} MiB)",
    )
    print(
        json.dumps(
            {
                "merged_path": output_path,
                "cell_count": cell_count,
                "gene_names": gene_names,
                "sample_names": sample_names,
                "groups": groups,
                "group_columns": group_columns,
                "batch_correction": str(manifest.get("batch_correction", "none")),
                "data_path": "rust-csr",
                "performance": {
                    "prepare_seconds": prepare_seconds,
                    "open_seconds": open_seconds,
                    "metadata_seconds": metadata_seconds,
                    "write_seconds": write_seconds,
                    "writer_seconds": writer_seconds,
                    "total_seconds": prepare_seconds + writer_seconds,
                    "csr_bytes": csr_bytes,
                    "h5ad_bytes": h5ad_bytes,
                    "data_dtype": str(matrix.dtype),
                },
            }
        )
    )


if __name__ == "__main__":
    main()
