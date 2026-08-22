#!/usr/bin/env python3
"""Run Scrublet on an h5ad file or a Matrix Market directory and emit JSON.

Usage: run_scrublet.py <path> [--per-sample <sample_column>]
"""

import gc
import json
import os
import re
import struct
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

try:
    from expression_source import normalize_windows_path, select_count_matrix, read_10x_mtx_flexible
except ImportError as error:
    raise RuntimeError(
        "DNBCScope could not load expression_source.py next to run_scrublet.py"
    ) from error

def as_scrublet_csr(matrix):
    """Return a CSR count matrix, narrowing exactly representable counts to f32.

    Scanpy's Scrublet pipeline creates several normalized/simulated copies of
    ``X``.  Matrix Market readers commonly produce f64 even though UMI counts
    are small integers, doubling the memory traffic through every copy.  All
    integer values up to 2**24 are represented exactly by float32; unusual
    matrices outside that safe range deliberately remain at their input dtype.
    """
    import numpy as np
    import scipy.sparse as sp

    if not sp.issparse(matrix):
        matrix = sp.csr_matrix(matrix)
    elif matrix.format != "csr":
        matrix = matrix.tocsr()

    data = matrix.data
    if data.size and (
        not np.isfinite(data).all()
        or (data < 0).any()
        or not np.allclose(data, np.rint(data), atol=1e-6, rtol=0.0)
    ):
        raise RuntimeError("Scrublet input contains non-finite, negative, or non-integer counts")
    if data.dtype != np.float32:
        safe_to_narrow = data.size == 0
        if data.size:
            safe_to_narrow = bool(
                np.isfinite(data).all()
                and data.min() >= 0
                and data.max() <= 2**24
            )
        if safe_to_narrow:
            matrix = matrix.astype(np.float32, copy=False)
    return matrix


def select_count_adata(adata, sample_column=None):
    """Return a minimal AnnData backed by raw counts, as required by Scrublet."""
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    try:
        matrix, names, _label, _kind = select_count_matrix(adata)
    except RuntimeError as error:
        raise RuntimeError(f"Scrublet requires raw counts. {error}") from error
    if not sp.issparse(matrix):
        if hasattr(matrix, "to_memory"):
            matrix = matrix.to_memory()
        else:
            matrix = np.asarray(matrix)
    matrix = as_scrublet_csr(matrix)
    # Scrublet only needs the sample label for per-sample mode. Copying every
    # metadata column from a large h5ad needlessly materializes tens or
    # hundreds of megabytes before the actual calculation starts.
    if sample_column and sample_column in adata.obs.columns:
        # Pandas' nullable values (especially NaN) do not compare equal to
        # themselves, which used to create an empty worker for a missing
        # sample. Normalize labels once so every cell belongs to exactly one
        # deterministic group in per-sample mode.
        labels = adata.obs[sample_column].astype("string").fillna("unknown").to_numpy(dtype=str)
        obs = pd.DataFrame({sample_column: labels}, index=adata.obs_names.copy())
    else:
        obs = pd.DataFrame(index=adata.obs_names.copy())
    return ad.AnnData(X=matrix, obs=obs, var=pd.DataFrame(index=names.copy()))


def run_scrublet_single(adata, expected_doublet_rate=None):
    """Run Scrublet on a single sample or combined data."""
    import numpy as np
    import scanpy as sc

    # Auto-calculate expected doublet rate from cell count if not provided
    n_cells = adata.n_obs
    if n_cells < 3:
        raise RuntimeError(f"Scrublet requires at least 3 cells; received {n_cells}")
    if expected_doublet_rate is None:
        expected_doublet_rate = min(0.008 * (n_cells / 1000), 0.08)
        expected_doublet_rate = max(expected_doublet_rate, 0.004)  # floor at 0.4%

    # Adaptive retry for PCA component constraints
    SCANPY_RECOVERABLE_ERRORS = (ValueError, np.linalg.LinAlgError)
    n_comps = None
    scrublet_ok = False
    last_error = None

    for attempt in range(3):
        try:
            kwargs = {}
            if n_comps is not None:
                kwargs["n_prin_comps"] = int(n_comps)
            sc.pp.scrublet(
                adata,
                expected_doublet_rate=expected_doublet_rate,
                threshold=None,
                copy=False,
                verbose=False,
                random_state=42,
                **kwargs,
            )
            scrublet_ok = True
            break
        except SCANPY_RECOVERABLE_ERRORS as e:
            last_error = e
            msg = str(e)
            match = re.search(r"min\(n_samples, n_features\)=(\d+)", msg)
            if match:
                n_comps = int(match.group(1)) - 1
            elif n_comps is None:
                n_comps = min(30, max(2, min(adata.n_obs, adata.n_vars) - 1))
            else:
                n_comps = max(2, n_comps // 2)
            if n_comps < 2:
                break

    if not scrublet_ok:
        raise RuntimeError(f"Scrublet failed after 3 attempts: {last_error or 'unknown numerical error'}")
    else:
        scores = adata.obs["doublet_score"].fillna(0.0).to_numpy()
        predicted = adata.obs["predicted_doublet"].fillna(False).to_numpy()

        # Derive the threshold from the boundary between predicted/non-predicted
        pos = scores[predicted]
        neg = scores[~predicted]
        if len(pos) > 0 and len(neg) > 0:
            used_threshold = float((pos.min() + neg.max()) / 2)
        else:
            used_threshold = float(np.median(scores) + 3 * np.median(np.abs(scores - np.median(scores))))
            predicted = scores > used_threshold

    n_predicted = int(predicted.sum())

    return scores, predicted, used_threshold, expected_doublet_rate, n_predicted


def available_memory_bytes():
    """Best-effort cross-platform available-memory query without psutil."""
    try:
        if os.name == "nt":
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong),
                    ("memory_load", ctypes.c_ulong),
                    ("total_phys", ctypes.c_ulonglong),
                    ("avail_phys", ctypes.c_ulonglong),
                    ("total_page_file", ctypes.c_ulonglong),
                    ("avail_page_file", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("avail_virtual", ctypes.c_ulonglong),
                    ("avail_extended_virtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.avail_phys)
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages * page_size)
    except Exception:
        return 0


def sparse_storage_bytes(matrix):
    import scipy.sparse as sp

    if not sp.issparse(matrix):
        return int(getattr(matrix, "nbytes", 0))
    return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)


def per_sample_worker_count(adata, sample_sizes):
    """Use at most two workers only when CPU and available RAM make it safe."""
    try:
        requested = max(1, min(2, int(os.environ.get("DNBC_SCRUBLET_MAX_WORKERS", "2"))))
    except (TypeError, ValueError):
        # A malformed inherited environment must fail safe, never abort QC.
        requested = 1
    if requested < 2 or len(sample_sizes) < 2 or (os.cpu_count() or 1) < 4:
        return 1
    available = available_memory_bytes()
    if available <= 0:
        return 1
    largest_two_fraction = sum(sorted(sample_sizes, reverse=True)[:2]) / max(1, adata.n_obs)
    # Scrublet creates simulated doublets plus normalized/PCA intermediates.
    # Eight times the two input slices is deliberately conservative.
    estimated_extra = int(sparse_storage_bytes(adata.X) * largest_two_fraction * 8)
    reserve = 2 * 1024**3
    return 2 if available > estimated_extra + reserve else 1


def emit_result(scores, predicted, threshold, expected_doublet_rate, n_doublets, per_sample=None):
    """Write compact Scrublet results for the desktop IPC boundary.

    The former JSON payload emitted one object per cell.  At one million cells
    that made Python serialize, Rust deserialize, and the WebView parse tens of
    megabytes after the scientific computation was already complete.  Keep the
    small summary as JSON, then append raw little-endian f32 scores and u8
    predictions.  Rust validates the frame without materializing those values.

    Frame: ``DBSL`` | u32 cell count | u32 JSON-byte count | JSON | f32[n] | u8[n]
    """
    import numpy as np

    scores = np.asarray(scores, dtype="<f4")
    predicted = np.asarray(predicted, dtype=np.uint8)
    if scores.ndim != 1 or predicted.ndim != 1 or scores.size != predicted.size:
        raise RuntimeError("Scrublet returned malformed per-cell results")
    if (
        not np.isfinite(scores).all()
        or not np.isfinite(float(threshold))
        or not np.isfinite(float(expected_doublet_rate))
        or not 0.0 <= float(expected_doublet_rate) <= 1.0
        or np.any(predicted > 1)
    ):
        raise RuntimeError("Scrublet returned non-finite or out-of-range results")
    metadata = {
        "threshold": round(float(threshold), 4),
        "expected_doublet_rate": round(float(expected_doublet_rate), 4),
        "n_doublets": int(n_doublets),
    }
    if per_sample is not None:
        metadata["per_sample"] = per_sample
    metadata_bytes = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    # Keep f32 scores naturally aligned after the variable-length JSON header.
    # JSON permits trailing whitespace, so this remains valid metadata.
    metadata_bytes += b" " * (-len(metadata_bytes) % 4)
    if len(metadata_bytes) > 16 * 1024 * 1024:
        raise RuntimeError("Scrublet summary metadata is unexpectedly large")

    output = sys.stdout.buffer
    output.write(struct.pack("<4sII", b"DBSL", int(scores.size), len(metadata_bytes)))
    output.write(metadata_bytes)
    output.write(scores.tobytes(order="C"))
    output.write(predicted.tobytes(order="C"))
    output.flush()


def main() -> None:
    if len(sys.argv) < 2:
        raise RuntimeError("Usage: run_scrublet.py <h5ad-or-mtx-path> [--per-sample <sample_column>]")

    import numpy as np
    import scanpy as sc

    path = normalize_windows_path(sys.argv[1])
    per_sample = False
    sample_column = "sample"

    # Parse arguments
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--per-sample":
            per_sample = True
            if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--"):
                sample_column = sys.argv[i + 1]
                i += 1
        i += 1

    backed_input = False
    if os.path.isfile(path) and path.lower().endswith(".json"):
        from native_csr import load_native_csr
        adata = load_native_csr(path, include_barcodes=False)
    elif os.path.isfile(path) and path.lower().endswith(".h5ad"):
        adata = sc.read_h5ad(path, backed="r")
        backed_input = True
    elif os.path.isdir(path):
        h5ad_files = sorted(
            (name for name in os.listdir(path) if name.lower().endswith(".h5ad")),
            key=lambda name: (name.casefold(), name),
        )
        if h5ad_files:
            adata = sc.read_h5ad(normalize_windows_path(os.path.join(path, h5ad_files[0])), backed="r")
            backed_input = True
        else:
            adata = read_10x_mtx_flexible(path)
    else:
        raise RuntimeError(f"Unsupported Scrublet input: {path}")

    source_adata = adata
    try:
        adata = select_count_adata(source_adata, sample_column if per_sample else None)
    finally:
        if backed_input:
            source_adata.file.close()
        del source_adata
    n_cells = adata.n_obs

    # Run Scrublet per sample or combined
    if per_sample and sample_column not in adata.obs.columns:
        raise RuntimeError(
            f"Per-sample Scrublet requires the sample column {sample_column!r}; it was not found in the input."
        )
    if per_sample:
        # Per-sample mode: run Scrublet on each sample separately
        scores = np.zeros(n_cells, dtype=np.float32)
        predicted = np.zeros(n_cells, dtype=bool)
        sample_results = []
        total_doublets = 0

        labels = adata.obs[sample_column].to_numpy(dtype=str)
        sample_names = list(dict.fromkeys(labels.tolist()))
        sample_entries = [(sample_name, np.flatnonzero(labels == sample_name)) for sample_name in sample_names]
        undersized = [(name, int(len(indices))) for name, indices in sample_entries if len(indices) < 3]
        if undersized:
            details = ", ".join(f"{name!r} ({count})" for name, count in undersized[:8])
            suffix = "…" if len(undersized) > 8 else ""
            raise RuntimeError(
                f"Per-sample Scrublet requires at least 3 cells per sample; undersized samples: {details}{suffix}"
            )
        worker_count = per_sample_worker_count(
            adata, [len(indices) for _name, indices in sample_entries]
        )

        def process_sample(entry):
            sample_name, sample_indices = entry
            sample_adata = adata[sample_indices].copy()
            result = run_scrublet_single(sample_adata)
            return sample_name, sample_indices, result

        if worker_count == 2:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="scrublet") as executor:
                processed_samples = list(executor.map(process_sample, sample_entries))
        else:
            processed_samples = [process_sample(entry) for entry in sample_entries]

        for sample_name, sample_indices, sample_result in processed_samples:
            sample_scores, sample_predicted, sample_threshold, sample_rate, sample_n_doublets = sample_result
            # Map back to original indices without a per-cell Python loop.
            scores[sample_indices] = sample_scores
            predicted[sample_indices] = sample_predicted

            total_doublets += sample_n_doublets
            sample_results.append({
                "sample_name": str(sample_name),
                "n_cells": int(len(sample_indices)),
                "n_doublets": sample_n_doublets,
                "threshold": round(sample_threshold, 4),
                "expected_doublet_rate": round(sample_rate, 4),
            })

        # Use overall threshold and rate for compatibility
        sample_sizes = np.asarray([r["n_cells"] for r in sample_results], dtype=float)
        used_threshold = np.average([r["threshold"] for r in sample_results], weights=sample_sizes)
        expected_doublet_rate = np.average([r["expected_doublet_rate"] for r in sample_results], weights=sample_sizes)
        n_predicted = total_doublets

        gc.collect()

        emit_result(
            scores,
            predicted,
            used_threshold,
            expected_doublet_rate,
            n_predicted,
            per_sample=sample_results,
        )
    else:
        # Combined mode: run Scrublet on all cells together
        scores, predicted, used_threshold, expected_doublet_rate, n_predicted = run_scrublet_single(adata)

        gc.collect()

        emit_result(scores, predicted, used_threshold, expected_doublet_rate, n_predicted)


if __name__ == "__main__":
    main()
