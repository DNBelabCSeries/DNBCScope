#!/usr/bin/env python3
"""Resident Scanpy worker for differential-expression and annotation tasks."""

import contextlib
import gc
import io
import json
import os
import queue
import sys
import threading
import traceback

# The Windows embeddable Python distribution can run in isolated mode (via a
# restrictive ``python._pth`` file). In that mode the directory containing
# this entry script is not guaranteed to be on ``sys.path``. The resident
# worker is materialised together with run_annotation.py,
# run_differential.py, and expression_source.py, so make that sibling-module
# contract explicit before importing any of them.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# Eager imports pay the scientific runtime cost during idle prewarming.
import anndata  # noqa: F401
import numpy  # noqa: F401
import pandas  # noqa: F401
import scanpy  # noqa: F401
import scipy  # noqa: F401

import run_annotation
import run_differential
from process_metrics import memory_snapshot


try:
    IDLE_SECONDS = max(30, int(os.environ.get("DNBC_SCIENTIFIC_WORKER_IDLE_SECONDS", "300")))
except (TypeError, ValueError):
    IDLE_SECONDS = 300
try:
    MAX_TASKS = max(1, int(os.environ.get("DNBC_SCIENTIFIC_WORKER_MAX_TASKS", "20")))
except (TypeError, ValueError):
    MAX_TASKS = 20
try:
    MAX_RSS_BYTES = max(0, int(os.environ.get("DNBC_SCIENTIFIC_WORKER_MAX_RSS_BYTES", "0")))
except (TypeError, ValueError):
    MAX_RSS_BYTES = 0


def input_reader(messages):
    while True:
        line = sys.stdin.readline()
        messages.put(line)
        if not line:
            return


def execute(message):
    module_name = str(message.get("module", ""))
    module = {
        "differential": run_differential,
        "annotation": run_annotation,
    }.get(module_name)
    if module is None:
        raise RuntimeError(f"Unsupported scientific worker module: {module_name}")
    args = [str(value) for value in message.get("args", [])]
    result_path = str(message.get("result_path", ""))
    env_updates = {str(key): str(value) for key, value in message.get("env", {}).items()}
    if result_path:
        env_updates["DNBC_RESULT_PATH"] = result_path
    previous_env = {key: os.environ.get(key) for key in env_updates}
    previous_argv = sys.argv
    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        os.environ.update(env_updates)
        sys.argv = [getattr(module, "__file__", module_name), *args]
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            module.main()
        output = stdout.getvalue()
        if output and result_path and not os.path.exists(result_path):
            # Legacy entrypoints may print their JSON payload instead of
            # writing DNBC_RESULT_PATH.  Only accept stdout as a fallback when
            # it is itself strict JSON.  PyDESeq2's ``summary()`` also prints
            # a human-readable table; that diagnostic must never overwrite
            # the JSON file emitted by run_differential.py.
            def reject_non_finite_stdout(value):
                raise ValueError(f"non-finite JSON constant {value}")

            try:
                json.loads(output, parse_constant=reject_non_finite_stdout)
            except (ValueError, json.JSONDecodeError):
                output = ""
            else:
                with open(result_path, "w", encoding="utf-8") as handle:
                    handle.write(output)
        if result_path:
            # A successful module call must leave a complete JSON document or
            # a DNAR compact annotation frame. Checking the envelope here
            # turns an empty/truncated result into an actionable worker error
            # instead of letting Rust report an opaque parse failure.
            try:
                with open(result_path, "rb") as handle:
                    raw_payload = handle.read()
                if not raw_payload:
                    raise RuntimeError("Scientific task produced an empty result file")
                if raw_payload.startswith(b"DNAR"):
                    if len(raw_payload) < 16:
                        raise RuntimeError("Scientific task produced a truncated DNAR frame")
                    version = int.from_bytes(raw_payload[4:8], "little")
                    metadata_length = int.from_bytes(raw_payload[8:12], "little")
                    payload_length = int.from_bytes(raw_payload[12:16], "little")
                    expected_length = 16 + metadata_length + payload_length
                    if version != 1 or metadata_length <= 0 or expected_length != len(raw_payload):
                        raise RuntimeError("Scientific task produced an invalid DNAR frame")
                    json.loads(raw_payload[16 : 16 + metadata_length].decode("utf-8"))
                else:
                    payload = raw_payload.decode("utf-8")
                    if not payload.strip():
                        raise RuntimeError("Scientific task produced an empty result file")

                    def reject_non_finite(value):
                        raise ValueError(f"non-finite JSON constant {value}")

                    json.loads(payload, parse_constant=reject_non_finite)
            except FileNotFoundError as error:
                raise RuntimeError(
                    "Scientific task completed without producing its result file"
                ) from error
            except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
                raise RuntimeError(f"Scientific task produced invalid result output: {error}") from error
        return {"ok": True, "diagnostics": stderr.getvalue()[-16000:]}
    finally:
        sys.argv = previous_argv
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        gc.collect()


def is_native_failure(error: BaseException, traceback_text: str) -> bool:
    """Identify failures that can leave an extension/runtime contaminated.

    Ordinary data/parameter errors keep the resident worker alive.  Native
    extension failures are different: even when Python catches the wrapper
    exception, BLAS/HDF5/NumPy state may no longer be trustworthy, so the host
    should recycle the process before accepting another task.
    """

    if isinstance(error, (MemoryError, SystemError)):
        return True
    haystack = f"{error}\n{traceback_text}".lower()
    return any(
        marker in haystack
        for marker in (
            "segmentation fault",
            "access violation",
            "illegal instruction",
            "bus error",
            "stack overflow",
            "dll load failed",
            "numpy.core._multiarray_umath",
            "hdf5 library",
            "bad allocation",
        )
    )


def main():
    messages = queue.Queue()
    task_count = 0
    threading.Thread(target=input_reader, args=(messages,), daemon=True).start()
    print(json.dumps({"ok": True, "status": "ready"}), flush=True)
    while True:
        try:
            line = messages.get(timeout=IDLE_SECONDS)
        except queue.Empty:
            # Give the host a distinguishable lifecycle marker before the
            # process exits. Rust may not observe the exit until the next task,
            # but it can then tell an intentional idle shutdown from a crash.
            print(json.dumps({"ok": True, "status": "idle"}), flush=True)
            return
        if not line:
            return
        try:
            message = json.loads(line)
            if message.get("command") == "close":
                return
            response = execute(message)
            task_count += 1
        except BaseException as error:
            traceback_text = traceback.format_exc()[-24000:]
            response = {
                "ok": False,
                "error": str(error),
                "traceback": traceback_text,
                "recycle": is_native_failure(error, traceback_text),
            }
            if response["recycle"]:
                response["recycle_reason"] = "native_error"
            task_count += 1
        response["task_count"] = task_count
        response["memory"] = memory_snapshot()
        observed_rss = response["memory"].get("rss_bytes") or response["memory"].get("peak_rss_bytes")
        rss_over_budget = bool(MAX_RSS_BYTES and observed_rss and observed_rss >= MAX_RSS_BYTES)
        native_recycle = bool(response.get("recycle", False))
        # Recycle after a bounded number of requests even when the worker is
        # otherwise healthy.  Scanpy/anndata can retain allocator arenas and
        # native thread-pool state after a large differential/annotation job;
        # recycling bounds long-session RSS without adding startup cost to
        # every request.  The response is flushed before the idle marker so
        # Rust can consume the result and restart lazily on the next request.
        response["recycle"] = native_recycle or task_count >= MAX_TASKS or rss_over_budget
        if rss_over_budget and not native_recycle:
            response["recycle_reason"] = "rss_budget"
        print(json.dumps(response, separators=(",", ":")), flush=True)
        if response["recycle"]:
            print(json.dumps({"ok": True, "status": "idle"}), flush=True)
            return


if __name__ == "__main__":
    main()
