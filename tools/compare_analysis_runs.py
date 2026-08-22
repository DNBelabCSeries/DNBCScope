#!/usr/bin/env python3
"""Compare two Scanpy analysis metadata files for cross-platform drift.

Usage: python compare_analysis_runs.py mac/analysis_meta.json win/analysis_meta.json
The command exits 0 when the deterministic contract and semantic cluster
profiles agree, and exits 1 with a JSON report when they do not.
"""

import json
import sys
from pathlib import Path


def _load(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def compare_metadata(left, right):
    left_runtime = left.get("runtime", {})
    right_runtime = right.get("runtime", {})
    left_parity = left_runtime.get("parity", {})
    right_parity = right_runtime.get("parity", {})
    left_semantic = left_parity.get("semantic", {})
    right_semantic = right_parity.get("semantic", {})
    checks = {
        "cellCount": left_semantic.get("cellCount") == right_semantic.get("cellCount"),
        "clusterSizeProfiles": left_semantic.get("clusterSizeProfiles")
        == right_semantic.get("clusterSizeProfiles"),
        "clusterLabels": left_parity.get("clusterSha256") == right_parity.get("clusterSha256"),
        "projectionShapeAndFinite": {
            key: value == right_semantic.get("projection", {}).get(key)
            for key, value in left_semantic.get("projection", {}).items()
        },
        "determinismContract": left_runtime.get("determinism") == right_runtime.get("determinism"),
    }
    checks["projectionShapeAndFinite"] = all(checks["projectionShapeAndFinite"].values())
    return {
        "semanticMatch": all(checks.values()),
        "exactMatch": left_parity.get("exactSha256") == right_parity.get("exactSha256"),
        "checks": checks,
        "runtimes": {"left": left_runtime, "right": right_runtime},
    }


def main():
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    report = compare_metadata(_load(sys.argv[1]), _load(sys.argv[2]))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["semanticMatch"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
