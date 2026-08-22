#!/usr/bin/env python3
"""Calculate project metadata directly from a persistent native CSR source."""

from __future__ import annotations

import json
import os
import struct
import sys

import numpy as np

from native_csr import load_native_csr


def emit_progress(percent: int, message: str) -> None:
    print(f"DNBC_MERGE_PROGRESS|{percent}|{message}", file=sys.stderr, flush=True)


def parse_mito_prefixes(raw: str):
    values = [value.strip().upper() for value in str(raw).split(",") if value.strip()]
    return values or ["MT"]


def is_mito_gene(name: str, prefixes) -> bool:
    upper = name.upper()
    for prefix in prefixes:
        if prefix == "MT":
            if upper.startswith(("MT-", "MT.", "MT_")):
                return True
        elif upper.startswith(prefix):
            return True
    return False


def category_from_ranges(cell_count, samples, values, *, include_ids=True):
    names = []
    lookup = {}
    ids = np.empty(cell_count, dtype=np.uint8)
    counts = []
    cursor = 0
    for sample, raw_value in zip(samples, values):
        start, end = int(sample["start"]), int(sample["end"])
        if start != cursor or end < start or end > cell_count:
            raise RuntimeError("Native CSR sample ranges are invalid")
        value = str(raw_value) if raw_value else "unknown"
        code = lookup.get(value)
        if code is None:
            code = len(names)
            if code >= 256:
                raise RuntimeError("Project metadata exceeds 256 categories")
            lookup[value] = code
            names.append(value)
            counts.append(0)
        ids[start:end] = code
        counts[code] += end - start
        cursor = end
    if cursor != cell_count:
        raise RuntimeError("Native CSR sample ranges do not cover every cell")
    return {"names": names, "ids": ids.tolist() if include_ids else ids, "counts": counts}


def _resolved_manifest_path(manifest_path: str, value: str) -> str:
    path = str(value)
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(os.path.abspath(manifest_path)), path)
    return os.path.abspath(path)


def write_qc_sidecar(path: str, n_genes, counts, pct_mito) -> None:
    """Persist compact QC records instead of serializing one dict per cell."""
    dtype = np.dtype(
        [
            ("cell_index", "<u8"),
            ("n_genes", "<u4"),
            ("n_counts", "<f8"),
            ("pct_mito", "<f8"),
        ]
    )
    records = np.empty(len(n_genes), dtype=dtype)
    records["cell_index"] = np.arange(len(n_genes), dtype="<u8")
    records["n_genes"] = np.asarray(n_genes, dtype="<u4")
    records["n_counts"] = np.asarray(counts, dtype="<f8")
    records["pct_mito"] = np.asarray(pct_mito, dtype="<f8")
    temporary_path = f"{path}.tmp"
    with open(temporary_path, "wb") as handle:
        handle.write(struct.pack("<4sIQ", b"DNMQ", 1, len(records)))
        records.tofile(handle)
    os.replace(temporary_path, path)


def write_category_sidecar(path: str, ids) -> None:
    """Persist one compact u8 category vector for native metadata."""
    values = np.asarray(ids, dtype=np.uint8)
    temporary_path = f"{path}.tmp"
    with open(temporary_path, "wb") as handle:
        values.tofile(handle)
    os.replace(temporary_path, path)


def compact_category_info(info, native_dir: str, filename: str):
    write_category_sidecar(os.path.join(native_dir, filename), info["ids"])
    return {
        "names": info["names"],
        "counts": info["counts"],
        "ids_path": filename,
    }


def main() -> None:
    if len(sys.argv) < 2:
        raise RuntimeError("Usage: read_native_csr_meta.py <manifest> [mito-prefix]")
    emit_progress(74, "Opening persistent merged matrix")
    sidecar_mode = bool(os.environ.get("DNBC_RESULT_PATH", "").strip())
    matrix, gene_names, _barcodes, manifest = load_native_csr(
        sys.argv[1], as_anndata=False, include_barcodes=not sidecar_mode
    )
    prefixes = parse_mito_prefixes(sys.argv[2] if len(sys.argv) > 2 else "MT")
    cell_count = matrix.shape[0]
    indptr, indices = matrix.indptr, matrix.indices

    emit_progress(78, "Calculating merged QC metrics")
    counts = np.asarray(matrix.sum(axis=1), dtype=np.float64).ravel()
    gene_expression_sums = np.asarray(matrix.sum(axis=0), dtype=np.float64).ravel()
    n_genes = np.diff(indptr).astype(np.uint32, copy=False)
    mito_mask = np.array([is_mito_gene(name, prefixes) for name in gene_names])
    if mito_mask.any():
        mito_counts = np.asarray(matrix[:, mito_mask].sum(axis=1), dtype=np.float64).ravel()
    else:
        mito_counts = np.zeros(cell_count, dtype=np.float64)
    pct_mito = np.divide(
        mito_counts * 100.0,
        counts,
        out=np.zeros(cell_count, dtype=np.float64),
        where=counts > 0,
    )
    if (
        not np.isfinite(counts).all()
        or not np.isfinite(gene_expression_sums).all()
        or not np.isfinite(pct_mito).all()
        or (counts < 0).any()
        or (gene_expression_sums < 0).any()
        or (pct_mito < 0).any()
    ):
        raise RuntimeError("Native CSR QC contains non-finite or negative values")
    detected = np.zeros(len(gene_names), dtype=bool)
    detected[indices] = True

    samples = list(manifest.get("samples") or [])
    sample_info = None
    group_info = None
    group_info_list = None
    if samples:
        sample_info = category_from_ranges(
            cell_count,
            samples,
            [sample.get("name") for sample in samples],
            include_ids=not sidecar_mode,
        )
        group_info_list = []
        columns = [str(value) for value in manifest.get("group_columns") or ["group"]]
        for column_index, column in enumerate(columns):
            values = []
            for sample in samples:
                groups = list(sample.get("groups") or [])
                values.append(groups[column_index] if column_index < len(groups) else "unknown")
            info = category_from_ranges(
                cell_count, samples, values, include_ids=not sidecar_mode
            )
            group_info_list.append({"column": column, "name": column, "info": info})
        group_info = group_info_list[0]["info"] if group_info_list else None

    emit_progress(82, "Merged source ready; H5AD write deferred until export")
    manifest_path = os.path.abspath(sys.argv[1])
    native_dir = os.path.dirname(manifest_path)
    # Keep descriptor references relative to the native directory.  The
    # merged project is deliberately portable and may be moved after import;
    # absolute paths here would make reopening its metadata fail on another
    # machine or user account.
    barcodes_path = os.path.relpath(
        _resolved_manifest_path(manifest_path, manifest["barcodes_path"]), native_dir
    )
    qc_metrics_path = "qc_metrics.bin"
    metadata_path = os.path.join(native_dir, "project_metadata.json")
    if sidecar_mode:
        write_qc_sidecar(os.path.join(native_dir, qc_metrics_path), n_genes, counts, pct_mito)
        metadata_sample_info = sample_info
        metadata_group_info = group_info
        metadata_group_info_list = group_info_list
        if sample_info is not None:
            metadata_sample_info = compact_category_info(sample_info, native_dir, "sample_ids.u8")
        if group_info_list is not None:
            metadata_group_info_list = []
            for index, dimension in enumerate(group_info_list):
                metadata_group_info_list.append(
                    {
                        "column": dimension["column"],
                        "name": dimension["name"],
                        "info": compact_category_info(
                            dimension["info"], native_dir, f"group_{index}_ids.u8"
                        ),
                    }
                )
            metadata_group_info = (
                metadata_group_info_list[0]["info"] if metadata_group_info_list else None
            )
        elif group_info is not None:
            metadata_group_info = compact_category_info(group_info, native_dir, "group_ids.u8")
        metadata_descriptor = {
            "format": "dnbcscope-native-project-metadata-v1",
            "cell_count": cell_count,
            "feature_count": int(detected.sum()),
            "total_features": len(gene_names),
            "gene_names": gene_names,
            "gene_expression_sums": gene_expression_sums.tolist(),
            "barcodes_path": barcodes_path,
            "qc_metrics_path": qc_metrics_path,
            "sample_info": metadata_sample_info,
            "group_info": metadata_group_info,
            "group_info_list": metadata_group_info_list,
        }
        metadata_temporary_path = f"{metadata_path}.tmp"
        with open(metadata_temporary_path, "w", encoding="utf-8") as handle:
            json.dump(
                metadata_descriptor,
                handle,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
        os.replace(metadata_temporary_path, metadata_path)
    sample_names = [str(sample.get("name") or "unknown") for sample in samples]
    groups = []
    for sample in samples:
        values = list(sample.get("groups") or [])
        groups.append(str(values[0]) if values and values[0] else "unknown")
    emit_progress(100, "Merge complete; H5AD write deferred until export")
    result = {
        "merged_path": manifest_path,
        "cell_count": cell_count,
        "gene_names": gene_names,
        "sample_names": sample_names,
        "groups": groups,
        "batch_correction": str(manifest.get("batch_correction", "none")),
        "source_format": "native-csr",
    }
    if sidecar_mode:
        result["native_metadata_path"] = metadata_path
    # The Rust launcher normally captures child stdout. A million-cell project
    # payload can be tens or hundreds of megabytes, so persist the structured
    # result to a sidecar and return only its path when requested. Direct
    # script callers retain the historical stdout JSON contract.
    result_path = os.environ.get("DNBC_RESULT_PATH", "").strip()
    if result_path:
        result_path = os.path.abspath(result_path)
        temporary_path = f"{result_path}.tmp"
        with open(temporary_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        os.replace(temporary_path, result_path)
        print(json.dumps({"result_path": result_path}, separators=(",", ":")))
    else:
        # Preserve the standalone script contract for callers that do not use
        # the desktop sidecar channel. This compatibility path is intentionally
        # opt-in because it recreates the large per-cell JSON payload.
        qc_metrics = [
            {
                "cell_index": index,
                "n_genes": int(n_genes[index]),
                "n_counts": float(counts[index]),
                "pct_mito": round(float(pct_mito[index]), 4),
            }
            for index in range(cell_count)
        ]
        result["project_payload"] = {
            "cell_count": cell_count,
            "feature_count": int(detected.sum()),
            "total_features": len(gene_names),
            "gene_names": gene_names,
            "gene_expression_sums": gene_expression_sums.tolist(),
            "barcodes": _barcodes,
            "qc_metrics": qc_metrics,
            "sample_info": sample_info,
            "group_info": group_info,
            "group_info_list": group_info_list,
            "imported_analysis": None,
        }
        print(json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
