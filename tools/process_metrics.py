"""Small, dependency-free process memory helpers.

The desktop runtime already has NumPy/Scanpy available, but keeping this
module stdlib-only makes it safe to import before scientific libraries and
from the opt-in benchmark harness.  Values are bytes and ``None`` means the
platform does not expose a usable counter.
"""

from __future__ import annotations

import os
import re
import sys


def _windows_memory(peak: bool) -> int | None:
    if os.name != "nt":
        return None
    try:
        import ctypes

        class _ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        process = ctypes.windll.kernel32.GetCurrentProcess()
        if not ctypes.windll.psapi.GetProcessMemoryInfo(
            process, ctypes.byref(counters), counters.cb
        ):
            return None
        value = counters.PeakWorkingSetSize if peak else counters.WorkingSetSize
        return int(value)
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def _linux_rss(peak: bool) -> int | None:
    if sys.platform.startswith("linux"):
        filename = "/proc/self/status" if peak else "/proc/self/statm"
        try:
            if peak:
                with open(filename, encoding="ascii") as handle:
                    text = handle.read()
                match = re.search(r"^VmHWM:\s+(\d+)\s+kB$", text, re.MULTILINE)
                return int(match.group(1)) * 1024 if match else None
            with open(filename, encoding="ascii") as handle:
                fields = handle.read().split()
            return int(fields[1]) * os.sysconf("SC_PAGE_SIZE") if len(fields) > 1 else None
        except (OSError, ValueError, IndexError):
            return None
    return None


def _resource_rss(peak: bool) -> int | None:
    # resource exposes max RSS on Unix. There is no portable current-RSS
    # value, so only use it for peak measurements.
    if not peak:
        return None
    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux reports KiB, macOS reports bytes.
        return value * 1024 if sys.platform.startswith("linux") else value
    except (ImportError, OSError, ValueError):
        return None


def rss_bytes() -> int | None:
    """Return current resident set size where the OS exposes it."""

    return _windows_memory(False) or _linux_rss(False)


def peak_rss_bytes() -> int | None:
    """Return the process peak resident set size since it started."""

    return _windows_memory(True) or _linux_rss(True) or _resource_rss(True)


def memory_snapshot() -> dict[str, int | None]:
    """Return stable JSON-friendly current/peak RSS measurements."""

    return {"rss_bytes": rss_bytes(), "peak_rss_bytes": peak_rss_bytes()}
