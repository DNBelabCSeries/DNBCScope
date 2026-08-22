"""Stdlib-only scientific runtime compatibility fingerprint.

The desktop uses this before reading an analysis cache.  It intentionally
records package versions and the numerical determinism contract, not the
current RSS or a full OS build string, so a cache is invalidated only when the
runtime that can change results has changed.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import sys


PACKAGE_NAMES = (
    "numpy",
    "scipy",
    "scanpy",
    "anndata",
    "numba",
    "pynndescent",
    "umap-learn",
    "leidenalg",
    "igraph",
    "harmonypy",
    "pydeseq2",
    "formulaic",
    "formulaic-contrasts",
)


def _positive_env_int(*names: str, default: int = 1) -> int:
    for name in names:
        try:
            value = int(os.environ.get(name, ""))
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return default


def build_runtime_fingerprint() -> dict:
    packages = {}
    for name in PACKAGE_NAMES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "missing"
    analysis_threads = _positive_env_int(
        "DNBC_ANALYSIS_THREADS", "NUMBA_NUM_THREADS", "OMP_NUM_THREADS"
    )
    numba_threads = _positive_env_int("NUMBA_NUM_THREADS")
    determinism = {
        "seed": 0,
        "analysis_threads": analysis_threads,
        "numba_threads": numba_threads,
        "numba_threading_layer": os.environ.get("NUMBA_THREADING_LAYER", "workqueue"),
        "mode": "bounded_parallel" if analysis_threads > 1 else "single_thread",
        "parallel_reductions": analysis_threads > 1,
        "leiden_flavor": "igraph",
        "leiden_iterations": 2,
        "neighbor_random_state": 0,
        "umap_random_state": 0,
    }
    contract = {
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "packages": packages,
        "determinism": determinism,
    }
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"compatibility_key": hashlib.sha256(encoded).hexdigest(), **contract}


if __name__ == "__main__":
    print(json.dumps(build_runtime_fingerprint(), separators=(",", ":")))
