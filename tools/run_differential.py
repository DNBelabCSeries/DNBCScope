#!/usr/bin/env python3
"""Calculate pairwise differential expression or one-vs-rest markers."""

import json
import math
import numbers
import os
import sys

# See scientific_worker.py: the bundled Windows interpreter may run with an
# isolated ``python._pth`` and omit the directory containing this script.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

try:
    from expression_source import make_expression_adata, normalize_windows_path, read_10x_mtx_flexible
except ImportError as error:
    raise RuntimeError(
        "DNBCScope could not load expression_source.py next to run_differential.py"
    ) from error


# The heat-map toolbar exposes 2/5/10 markers per group.  Persist enough
# complete statistics for the largest choice instead of truncating every
# analysis to a global 60-row panel (which is too small for projects with
# many clusters).
HEATMAP_MARKERS_PER_GROUP = 10


def emit_json_result(value):
    """Write directly to the resident worker result file when available."""
    # Rust deserializes the result with serde_json, which deliberately rejects
    # Python's non-standard ``NaN``/``Infinity`` tokens.  Scanpy can produce
    # non-finite statistics for zero-variance groups, so fail at the Python
    # boundary instead of writing a payload that looks like JSON but cannot be
    # consumed by the desktop client.
    encoded = json.dumps(value, separators=(",", ":"), allow_nan=False)
    result_path = os.environ.get("DNBC_RESULT_PATH")
    if result_path:
        # Atomic replacement prevents Rust from ever observing a truncated
        # result if the worker is interrupted while writing a large marker
        # table (especially common on Windows when an antivirus scanner opens
        # the file at the same time).
        temporary_path = f"{result_path}.tmp.{os.getpid()}"
        try:
            with open(temporary_path, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, result_path)
        finally:
            try:
                if os.path.exists(temporary_path):
                    os.unlink(temporary_path)
            except OSError:
                pass
    else:
        print(encoded)


def select_expression_adata(adata, cell_indices=None):
    selected, is_counts, _source_label = make_expression_adata(
        adata, cell_indices=cell_indices
    )
    return selected, is_counts


def load_pydeseq2():
    """Load the supported PyDESeq2 API with an actionable stale-runtime error.

    PyDESeq2 0.4.x stores one-dimensional masks and size factors in AnnData's
    ``varm``/``obsm`` containers.  That representation is not compatible with
    the newer AnnData shape contract used by the bundled runtime.  Versions
    0.5.x moved those fields to ``var``/``obs`` and also introduced the
    formulaic design API, which is the contract DNBCScope now packages.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        from packaging.version import Version

        pydeseq2_version = Version(version("pydeseq2"))
        minimum_version = Version("0.5.4")
    except PackageNotFoundError as error:
        raise RuntimeError(
            "Multi-sample DE requires PyDESeq2 >=0.5.4 in the DNBCScope analysis "
            "environment. Rebuild the bundled Python environment from "
            "requirements/analysis-lock.txt."
        ) from error
    except Exception as error:
        raise RuntimeError(
            "DNBCScope could not determine the installed PyDESeq2 version. "
            "Rebuild the bundled Python environment."
        ) from error

    if pydeseq2_version < minimum_version:
        raise RuntimeError(
            f"PyDESeq2 {pydeseq2_version} is incompatible with the bundled "
            "AnnData runtime. Multi-sample DE requires PyDESeq2 >=0.5.4; "
            "rebuild the DNBCScope analysis environment."
        )

    try:
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.ds import DeseqStats
    except ImportError as error:
        raise RuntimeError(
            "PyDESeq2 is installed but its formulaic dependencies are missing. "
            "Rebuild the DNBCScope analysis environment from "
            "requirements/analysis-lock.txt."
        ) from error
    return DeseqDataSet, DeseqStats


def resolve_h5ad(path):
    path = normalize_windows_path(path)
    if os.path.isfile(path) and path.lower().endswith(".h5ad"):
        return path
    # Keep the chosen h5ad stable across APFS and NTFS enumeration order.
    for name in sorted(os.listdir(path), key=lambda value: (value.casefold(), value)):
        if name.lower().endswith(".h5ad"):
            return os.path.join(path, name)
    raise RuntimeError(f"No h5ad file found in {path}")


def load_source(path, data_format):
    path = normalize_windows_path(path)
    import anndata as ad

    if data_format == "native-csr":
        from native_csr import load_native_csr
        return load_native_csr(path, include_barcodes=False)
    if data_format == "h5ad":
        return ad.read_h5ad(resolve_h5ad(path), backed="r")
    if data_format != "mtx":
        raise RuntimeError(f"Unsupported data format: {data_format}")
    # Use the shared 10x reader so differential expression has the same
    # matrix/feature/barcode validation and duplicate-symbol handling as the
    # main analysis and annotation pipelines.
    return read_10x_mtx_flexible(path)


def nonzero_fraction(matrix):
    import numpy as np

    if hasattr(matrix, "getnnz"):
        return np.asarray(matrix.getnnz(axis=0)).ravel() / matrix.shape[0]
    return np.count_nonzero(np.asarray(matrix), axis=0) / matrix.shape[0]


def _iter_filtered_ranked_rows(ranked, group, pct_group, pct_reference, filters):
    """Yield all rows that pass the user-facing marker filters.

    The table and heat-map panels intentionally consume this same filtered
    stream: both rank candidates by Scanpy's statistical score, so a marker
    kept for one view is also a candidate for the other.
    """
    import numpy as np

    names = ranked["names"][group]
    scores = ranked["scores"][group]
    logfc = ranked["logfoldchanges"][group]
    pvals_adj = ranked["pvals_adj"][group]
    for index, gene in enumerate(names):
        gene = str(gene)
        group_pct = float(pct_group.get(gene, 0.0))
        reference_pct = float(pct_reference.get(gene, 0.0))
        fold_change = float(logfc[index])
        adjusted_p = float(pvals_adj[index])
        score = float(scores[index])
        # A non-finite statistic is not a meaningful marker and would be
        # emitted as NaN/Infinity by the default Python JSON encoder.  Omit it
        # explicitly so the result remains strict JSON and the UI never shows
        # a fake marker with an unusable p-value.
        if not all(np.isfinite(value) for value in (score, fold_change, adjusted_p)):
            continue
        # A large fold-change can be produced by a tiny numerator when the
        # reference expression is even tinier.  Follow the usual marker
        # workflow (Scanpy/Seurat): require the population in which the gene
        # is enriched to meet the expression-fraction floor, then require a
        # minimum difference in detection fraction.  Using max(target, ref)
        # here would let a low-prevalence target pass merely because the
        # reference happens to be expressed in more cells.
        enriched_pct = group_pct if fold_change >= 0 else reference_pct
        if enriched_pct < filters["min_pct"]:
            continue
        if abs(group_pct - reference_pct) < filters.get("min_diff_pct", 0.0):
            continue
        if filters["only_positive"] and fold_change <= 0:
            continue
        if abs(fold_change) < filters["min_log2fc"] or adjusted_p > filters["max_p_value_adj"]:
            continue
        yield {
            "group": group,
            "gene": gene,
            "score": score,
            "log2_fold_change": fold_change,
            "p_value_adj": adjusted_p,
            "pct_target": group_pct,
            "pct_reference": reference_pct,
        }


def add_ranked_rows(rows, ranked, group, pct_group, pct_reference, filters):
    collected = list(_iter_filtered_ranked_rows(ranked, group, pct_group, pct_reference, filters))
    rows.extend(select_table_marker_rows(collected, filters["max_genes_per_group"]))


def select_table_marker_rows(collected, maximum_rows):
    """Keep the top score-ranked table rows for each group.

    The absolute score is used so both positively and negatively enriched
    markers survive the cap. Log2FC is deliberately not prioritised here: a
    large fold change can come from a gene detected in only a handful of
    cells, while the statistical score reflects both effect size and how
    many cells actually express the gene.
    """
    if maximum_rows <= 0 or len(collected) <= maximum_rows:
        return list(collected)
    ranked = sorted(collected, key=lambda row: (-abs(row["score"]), str(row["gene"])))
    return ranked[:maximum_rows]


def select_heatmap_genes(marker_rows, groups, markers_per_group=12, maximum_genes=60):
    """Choose a balanced, deterministic marker panel for the heat map.

    Markers are ranked by the statistical score, matching the table's default
    ordering. Score is preferred over Log2FC because a large fold change can
    be produced by a gene expressed in only a few cells, which makes a poor
    heat-map marker, while the score reflects both effect size and how many
    cells actually express the gene.
    """
    selected_genes = []
    seen = set()

    def priority(row):
        adjusted_p = row.get("p_value_adj")
        p_priority = (
            -math.log10(max(float(adjusted_p), 1e-300))
            if adjusted_p is not None and math.isfinite(float(adjusted_p))
            else 0.0
        )
        return (
            -float(row.get("score", 0.0)),
            -p_priority,
            str(row.get("gene", "")),
        )

    # Keep each group's marker block together. If a top gene was already used
    # by an earlier group, continue down this group's ranked list so its quota
    # is filled with the next unique score-ranked marker.
    for group in groups:
        group_rows = sorted(
            (row for row in marker_rows if row["group"] == group),
            key=priority,
        )
        selected_group_rows = []
        group_marker_count = 0
        for row in group_rows:
            if group_marker_count >= markers_per_group:
                break
            gene = row["gene"]
            if gene in seen:
                continue
            if len(selected_genes) + len(selected_group_rows) >= maximum_genes:
                break
            seen.add(gene)
            selected_group_rows.append(row)
            group_marker_count += 1
        selected_group_rows.sort(
            key=lambda row: (
                -float(row.get("score", 0.0)),
                (
                    math.log10(max(float(row["p_value_adj"]), 1e-300))
                    if row.get("p_value_adj") is not None and math.isfinite(float(row["p_value_adj"]))
                    else 0.0,
                ),
                str(row.get("gene", "")),
            ),
        )
        selected_genes.extend(row["gene"] for row in selected_group_rows)
        if len(selected_genes) >= maximum_genes:
            break
    return selected_genes


def select_heatmap_genes_from_ranked(
    ranked,
    groups,
    pct_by_group,
    pct_by_reference,
    filters,
    markers_per_group=12,
    maximum_genes=60,
):
    """Select heat-map genes from the complete ranked statistics.

    ``marker_rows`` is deliberately not used here: the table is capped by
    ``max_genes_per_group`` after score sorting, which could hide a stronger
    marker before the heat-map selector ever sees it. Process one group at a
    time so the complete ranked arrays do not become a second large in-memory
    result table. Candidates are ranked by the statistical score (not Log2FC)
    so markers detected in only a few cells do not crowd out robust ones.
    """
    selected_genes = []
    seen = set()

    def priority(row):
        adjusted_p = row.get("p_value_adj")
        p_priority = (
            -math.log10(max(float(adjusted_p), 1e-300))
            if adjusted_p is not None and math.isfinite(float(adjusted_p))
            else 0.0
        )
        return (
            -float(row.get("score", 0.0)),
            -p_priority,
            str(row.get("gene", "")),
        )

    for group in groups:
        candidates = list(
            _iter_filtered_ranked_rows(
                ranked,
                group,
                pct_by_group[group],
                pct_by_reference[group],
                filters,
            )
        )
        candidates.sort(key=priority)
        selected_group_rows = []
        group_marker_count = 0
        for row in candidates:
            if len(selected_genes) + len(selected_group_rows) >= maximum_genes or group_marker_count >= markers_per_group:
                break
            gene = row["gene"]
            if gene in seen:
                continue
            seen.add(gene)
            selected_group_rows.append(row)
            group_marker_count += 1
        selected_group_rows.sort(
            key=lambda row: (
                -float(row.get("score", 0.0)),
                (
                    math.log10(max(float(row["p_value_adj"]), 1e-300))
                    if row.get("p_value_adj") is not None and math.isfinite(float(row["p_value_adj"]))
                    else 0.0,
                ),
                str(row.get("gene", "")),
            ),
        )
        selected_genes.extend(row["gene"] for row in selected_group_rows)
        if len(selected_genes) >= maximum_genes:
            break
    return selected_genes


def build_pairwise_heatmap_rows(selected_genes, ranked, pct_target, pct_reference):
    """Add a reference column for a target-vs-reference analysis.

    Scanpy returns the target-vs-reference statistic once.  For the same two
    groups the reference effect is the sign-reversed target effect; the
    expression fractions are swapped and the two-sided P value is unchanged.
    Keeping this derived row beside the original result makes the heat map a
    true two-column comparison without running a second full DE pass.
    """
    import numpy as np

    names = [str(name) for name in ranked["names"]["target"]]
    indices = {gene: index for index, gene in enumerate(names)}
    scores = ranked["scores"]["target"]
    logfc = ranked["logfoldchanges"]["target"]
    pvals_adj = ranked["pvals_adj"]["target"]
    rows = []
    for gene in selected_genes:
        index = indices.get(gene)
        if index is None:
            continue
        score = float(scores[index])
        fold_change = float(logfc[index])
        adjusted_p = float(pvals_adj[index])
        if not np.isfinite(score) or not np.isfinite(fold_change):
            continue
        target = {
            "group": "target",
            "gene": gene,
            "score": score,
            "log2_fold_change": fold_change,
            "p_value_adj": adjusted_p if np.isfinite(adjusted_p) else None,
            "pct_target": float(pct_target.get(gene, 0.0)),
            "pct_reference": float(pct_reference.get(gene, 0.0)),
        }
        rows.append(target)
        rows.append(
            {
                "group": "reference",
                "gene": gene,
                "score": -float(target["score"]),
                "log2_fold_change": -float(target["log2_fold_change"]),
                "p_value_adj": target["p_value_adj"],
                "pct_target": float(pct_reference.get(gene, 0.0)),
                "pct_reference": float(pct_target.get(gene, 0.0)),
            }
        )
    return rows


def build_heatmap_rows(selected_genes, ranked, groups, pct_by_group, pct_by_reference):
    """Return complete one-vs-rest statistics for the genes selected for the heat map.

    Marker thresholds are intentionally used only to choose which genes enter the
    heat map. Once selected, every group gets its actual Scanpy statistic so a
    filtered/missing marker is never presented as a zero fold change.
    """
    import numpy as np

    rows = []
    for group in groups:
        names = [str(name) for name in ranked["names"][group]]
        indices = {gene: index for index, gene in enumerate(names)}
        scores = ranked["scores"][group]
        logfc = ranked["logfoldchanges"][group]
        pvals_adj = ranked["pvals_adj"][group]
        for gene in selected_genes:
            index = indices.get(gene)
            if index is None:
                continue
            score = float(scores[index])
            fold_change = float(logfc[index])
            adjusted_p = float(pvals_adj[index])
            # Undefined Scanpy statistics should remain absent and be rendered
            # as N/A by the client, rather than being converted into a false 0.
            if not np.isfinite(score) or not np.isfinite(fold_change):
                continue
            rows.append(
                {
                    "group": group,
                    "gene": gene,
                    "score": score,
                    "log2_fold_change": fold_change,
                    "p_value_adj": adjusted_p if np.isfinite(adjusted_p) else None,
                    "pct_target": float(pct_by_group[group].get(gene, 0.0)),
                    "pct_reference": float(pct_by_reference[group].get(gene, 0.0)),
                }
            )
    return rows


def benjamini_hochberg(p_values):
    """Return FDR-adjusted p values without adding a statsmodels dependency."""
    import numpy as np

    values = np.asarray(p_values, dtype=float)
    valid = np.isfinite(values)
    adjusted = np.full(values.shape, np.nan)
    if not valid.any():
        return adjusted
    order = np.argsort(values[valid])
    sorted_values = values[valid][order]
    count = len(sorted_values)
    corrected = sorted_values * count / np.arange(1, count + 1)
    corrected = np.minimum.accumulate(corrected[::-1])[::-1]
    indices = np.flatnonzero(valid)[order]
    adjusted[indices] = np.minimum(corrected, 1.0)
    return adjusted


def make_unique_names(names):
    """Keep DataFrame columns valid when a matrix carries duplicate symbols."""
    seen = {}
    result = []
    for name in names:
        base = str(name)
        count = seen.get(base, 0)
        seen[base] = count + 1
        result.append(base if count == 0 else f"{base}-{count}")
    return result


VALID_TEST_METHODS = ("t-test", "wilcoxon")


def coerce_cell_index(value, label="cell index"):
    """Accept only finite, integral numeric indices at the JSON boundary."""
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise RuntimeError(f"{label} must be a non-negative integer")
    if not math.isfinite(float(value)) or float(value) != math.floor(float(value)):
        raise RuntimeError(f"{label} must be a non-negative integer")
    index = int(value)
    if index < 0 or index > 0xFFFFFFFF:
        raise RuntimeError(f"{label} is outside the supported range")
    return index


def normalize_test_method(value):
    """Normalize public test-method values before calling Scanpy."""
    normalized = "wilcoxon" if value is None else str(value).strip().lower()
    normalized = normalized.replace("_", "-").replace(" ", "-")
    if normalized in {"ttest", "t-test"}:
        return "t-test"
    if normalized == "wilcoxon":
        return "wilcoxon"
    choices = ", ".join(VALID_TEST_METHODS)
    raise RuntimeError(f"Unsupported differential test method {value!r}; choose {choices}")


def parse_bool(value, default=False, name="boolean"):
    """Parse booleans without Python's surprising bool('false') behaviour."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise RuntimeError(f"{name} must be a boolean")


def parse_differential_filters(request):
    """Validate numeric filters at the Python boundary as well as in Rust."""
    def finite_float(name, default, minimum, maximum):
        try:
            value = float(request.get(name, default))
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"{name} must be a number") from error
        if not math.isfinite(value) or not minimum <= value <= maximum:
            raise RuntimeError(f"{name} must be between {minimum} and {maximum}")
        return value

    try:
        max_genes = int(request.get("max_genes_per_group", 200))
    except (TypeError, ValueError) as error:
        raise RuntimeError("max_genes_per_group must be an integer") from error
    if not 1 <= max_genes <= 10_000:
        raise RuntimeError("max_genes_per_group must be between 1 and 10000")
    return {
        "min_pct": finite_float("min_pct", 0.1, 0.0, 1.0),
        # Keep the Python default at zero for legacy JSON/binary requests that
        # predate this filter.  The current UI sends an explicit 0.10 default.
        "min_diff_pct": finite_float(
            "min_diff_pct",
            request.get("min_diff_pct", request.get("minDiffPct", 0.0)),
            0.0,
            1.0,
        ),
        "min_log2fc": finite_float("min_log2fc", 0.25, 0.0, 100.0),
        "max_p_value_adj": finite_float("max_p_value_adj", 0.05, 0.0, 1.0),
        "only_positive": parse_bool(request.get("only_positive"), default=False, name="only_positive"),
        "max_genes_per_group": max_genes,
    }


def load_differential_request(path):
    """Load legacy JSON or the compact DNDE binary request format."""
    import numpy as np

    with open(path, "rb") as handle:
        content = handle.read()
    if not content.startswith(b"DNDE"):
        request = json.loads(content.decode("utf-8"))
        if not isinstance(request, dict):
            raise RuntimeError("Differential JSON request must be an object")
        request = dict(request)
        request["test_method"] = normalize_test_method(
            request.get("test_method", request.get("testMethod", request.get("method")))
        )
        return request
    if len(content) < 16 or content[4] != 1:
        raise RuntimeError("Unsupported differential binary request")
    mode_code = content[5]
    metadata_size = int.from_bytes(content[8:12], "little")
    payload_size = int.from_bytes(content[12:16], "little")
    if len(content) != 16 + metadata_size + payload_size:
        raise RuntimeError("Truncated differential binary request")
    metadata = json.loads(content[16:16 + metadata_size].decode("utf-8"))
    if not isinstance(metadata, dict):
        raise RuntimeError("Differential binary metadata must be an object")
    payload = memoryview(content)[16 + metadata_size:]
    request = {
        "mode": {1: "pairwise", 2: "find_all", 3: "pseudobulk"}.get(mode_code),
        "min_pct": float(metadata.get("minPct", 0.1)),
        "min_diff_pct": float(metadata.get("minDiffPct", 0.0)),
        "min_log2fc": float(metadata.get("minLog2fc", 0.25)),
        "max_p_value_adj": float(metadata.get("maxPValueAdj", 0.05)),
        "only_positive": parse_bool(metadata.get("onlyPositive"), default=False, name="onlyPositive"),
        "test_method": normalize_test_method(
            metadata.get("testMethod", metadata.get("test_method"))
        ),
        "max_genes_per_group": 100,
        # The current binary UI consumes richer pairwise and pseudobulk
        # payloads; the legacy JSON command leaves this unset so its list
        # response remains backwards-compatible.
        "include_heatmap": mode_code in {1, 3},
    }
    if request["mode"] is None:
        raise RuntimeError(f"Unknown differential binary mode {mode_code}")
    if mode_code == 1:
        target_count = int(metadata["targetCount"])
        reference_count = int(metadata["referenceCount"])
        expected = (target_count + reference_count) * 4
        if len(payload) != expected:
            raise RuntimeError("Pairwise differential payload has an invalid length")
        indices = np.frombuffer(payload, dtype="<u4")
        request["target"] = indices[:target_count]
        request["reference"] = indices[target_count:]
    elif mode_code == 2:
        names = [str(name) for name in metadata["groupNames"]]
        counts = [int(count) for count in metadata["groupCounts"]]
        if len(names) != len(counts) or len(payload) != sum(counts) * 4:
            raise RuntimeError("Find-all differential payload has invalid group boundaries")
        indices = np.frombuffer(payload, dtype="<u4")
        groups = []
        offset = 0
        for name, count in zip(names, counts):
            groups.append({"name": name, "indices": indices[offset:offset + count]})
            offset += count
        request["groups"] = groups
    else:
        record_count = int(metadata["recordCount"])
        if len(payload) != record_count * 7:
            raise RuntimeError("Pseudobulk differential payload has an invalid length")
        source_bytes = record_count * 4
        sample_bytes = record_count * 2
        request.update(
            {
                "records_compact": {
                    "source_indices": np.frombuffer(payload[:source_bytes], dtype="<u4"),
                    "sample_ids": np.frombuffer(
                        payload[source_bytes:source_bytes + sample_bytes], dtype="<u2"
                    ),
                    "condition_ids": np.frombuffer(
                        payload[source_bytes + sample_bytes:], dtype=np.uint8
                    ),
                    "sample_names": [str(name) for name in metadata["sampleNames"]],
                    "condition_names": [str(name) for name in metadata["conditionNames"]],
                },
                "condition_a": str(metadata["conditionA"]),
                "condition_b": str(metadata["conditionB"]),
                "cell_type_name": str(metadata.get("cellTypeName", "")),
                "min_cells_per_sample": int(metadata.get("minCellsPerSample", 10)),
            }
        )
    return request


def pseudobulk_results(adata, records, request, filters, include_heatmap=False):
    """Aggregate count data per sample and run replicate-aware DESeq2.

    With fewer than two samples on either side, results remain useful as a
    normalized descriptive fold-change table, but statistical p values are
    intentionally omitted instead of treating cells as independent replicates.
    """
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    condition_a = str(request["condition_a"])
    condition_b = str(request["condition_b"])
    if not condition_a or not condition_b or condition_a == condition_b:
        raise RuntimeError("Choose two different experimental conditions")
    min_cells = max(1, int(request.get("min_cells_per_sample", 10)))
    grouped = {}
    compact = request.get("records_compact")
    if compact:
        compact_lengths = {
            len(compact.get("source_indices", [])),
            len(compact.get("sample_ids", [])),
            len(compact.get("condition_ids", [])),
        }
        if len(compact_lengths) != 1:
            raise RuntimeError("Pseudobulk compact columns must have equal lengths")
        sample_names = list(compact.get("sample_names", []))
        condition_names = list(compact.get("condition_names", []))
        for index, (sample_id, condition_id) in enumerate(
            zip(compact.get("sample_ids", []), compact.get("condition_ids", []))
        ):
            sample_id = int(sample_id)
            condition_id = int(condition_id)
            if sample_id < 0 or sample_id >= len(sample_names):
                raise RuntimeError(f"Pseudobulk sample id {sample_id} at index {index} is out of range")
            if condition_id < 0 or condition_id >= len(condition_names):
                raise RuntimeError(f"Pseudobulk condition id {condition_id} at index {index} is out of range")
        record_iter = (
            (
                coerce_cell_index(source_index, "Pseudobulk source index"),
                sample_names[int(sample_id)],
                condition_names[int(condition_id)],
            )
            for source_index, sample_id, condition_id in zip(
                compact["source_indices"],
                compact["sample_ids"],
                compact["condition_ids"],
            )
        )
    else:
        record_iter = (
            (
                coerce_cell_index(record["source_index"], "Pseudobulk source index"),
                str(record["sample"]).strip(),
                str(record["condition"]).strip(),
            )
            for record in records
        )
    seen_source_indices = set()
    for source_index, sample, condition in record_iter:
        if source_index < 0 or source_index >= adata.n_obs:
            raise RuntimeError("Pseudobulk record contains an out-of-range cell index")
        if source_index in seen_source_indices:
            raise RuntimeError("Pseudobulk records contain duplicate cell indices")
        seen_source_indices.add(source_index)
        if not sample or condition not in {condition_a, condition_b}:
            continue
        key = (sample, condition)
        grouped.setdefault(key, []).append(source_index)
    if not grouped:
        raise RuntimeError("No selected cells have both sample and condition metadata")

    sample_conditions = {}
    for sample, condition in grouped:
        if sample in sample_conditions and sample_conditions[sample] != condition:
            raise RuntimeError(
                f"Sample {sample!r} belongs to more than one selected condition; "
                "sample and condition must not be confounded."
            )
        sample_conditions[sample] = condition

    retained = [(sample, condition, indices) for (sample, condition), indices in grouped.items() if len(indices) >= min_cells]
    counts_by_condition = {
        condition_a: sum(1 for _sample, condition, _indices in retained if condition == condition_a),
        condition_b: sum(1 for _sample, condition, _indices in retained if condition == condition_b),
    }
    if not counts_by_condition[condition_a] or not counts_by_condition[condition_b]:
        raise RuntimeError(
            f"No valid pseudobulk samples remain after requiring {min_cells} cells per sample "
            "for the selected cell type."
        )

    # Backed h5ad fancy indexing requires increasing row indices. Keep one
    # sorted expression slice, then address its rows through a compact lookup
    # while aggregating each sample.
    selected_source_indices = sorted({
        index
        for _sample, _condition, sample_indices in retained
        for index in sample_indices
    })
    selected_row = {
        source_index: row_index
        for row_index, source_index in enumerate(selected_source_indices)
    }
    selected, is_counts = select_expression_adata(adata, selected_source_indices)
    if not is_counts:
        raise RuntimeError(
            "Multi-sample differential expression requires raw integer counts. "
            "This dataset only has processed expression values."
        )
    matrix = selected.X
    if not sp.issparse(matrix):
        matrix = sp.csr_matrix(matrix)
    else:
        matrix = matrix.tocsr()
    pseudo_counts = []
    sample_names = []
    conditions = []
    cell_counts = []
    for sample, condition, indices in retained:
        rows = [selected_row[index] for index in indices]
        pseudo_counts.append(np.asarray(matrix[rows].sum(axis=0)).ravel())
        sample_names.append(sample)
        conditions.append(condition)
        cell_counts.append(len(indices))
    counts = np.vstack(pseudo_counts).astype(np.int64, copy=False)
    feature_mask = counts.sum(axis=0) > 0
    if not feature_mask.any():
        raise RuntimeError("Selected pseudobulk samples contain no expressed genes")
    gene_names = make_unique_names(np.asarray([str(name) for name in selected.var_names])[feature_mask])
    counts = counts[:, feature_mask]
    library_sizes = counts.sum(axis=1).astype(float)
    if np.any(library_sizes <= 0):
        raise RuntimeError("A pseudobulk sample has zero total counts")
    normalized = counts / library_sizes[:, None] * 1e6
    condition_mask_a = np.asarray(conditions) == condition_a
    condition_mask_b = np.asarray(conditions) == condition_b
    mean_a = normalized[condition_mask_a].mean(axis=0)
    mean_b = normalized[condition_mask_b].mean(axis=0)
    log2fc = np.log2((mean_b + 1e-6) / (mean_a + 1e-6))

    # ``selected`` is materialized in ``selected_source_indices`` order, which
    # is deliberately sorted for backed h5ad fancy indexing.  The retained
    # sample groups are not necessarily in that same order (they follow the
    # first-seen sample order in the request), so concatenating their
    # conditions here would silently attach the wrong condition to rows and
    # corrupt pct_a/pct_b filtering.  Re-index the condition labels through the
    # same source-index ordering used by the matrix instead.
    condition_by_source = {
        index: condition
        for _sample, condition, indices in retained
        for index in indices
    }
    selected_conditions = np.asarray(
        [condition_by_source[index] for index in selected_source_indices], dtype=object
    )
    pct_a = np.asarray(matrix[selected_conditions == condition_a].getnnz(axis=0)).ravel()[feature_mask] / max(
        1, (selected_conditions == condition_a).sum()
    )
    pct_b = np.asarray(matrix[selected_conditions == condition_b].getnnz(axis=0)).ravel()[feature_mask] / max(
        1, (selected_conditions == condition_b).sum()
    )

    replicated = counts_by_condition[condition_a] >= 2 and counts_by_condition[condition_b] >= 2
    if replicated:
        DeseqDataSet, DeseqStats = load_pydeseq2()
        counts_frame = pd.DataFrame(counts, index=sample_names, columns=gene_names)
        metadata = pd.DataFrame({"condition": conditions}, index=sample_names)

        # Constant genes contain no between-sample signal and can make the
        # dispersion fit degenerate. Remove them before fitting; this is a
        # statistical pre-filter, not a workaround for the old AnnData shape
        # incompatibility (the runtime now pins PyDESeq2 >= 0.5).
        gene_variance = counts_frame.var(axis=0)
        variable_mask = gene_variance.values > 0
        if not variable_mask.all():
            counts_frame = counts_frame.loc[:, variable_mask]
            pct_a = pct_a[variable_mask]
            pct_b = pct_b[variable_mask]
        if counts_frame.shape[1] == 0:
            raise RuntimeError(
                "No genes vary across the selected pseudobulk samples; "
                "replicate-aware differential expression cannot be fitted."
            )
        de_gene_names = list(counts_frame.columns)

        dds = DeseqDataSet(
            counts=counts_frame,
            metadata=metadata,
            design="~condition",
            refit_cooks=True,
            n_cpus=1,
        )
        dds.deseq2()
        result = DeseqStats(dds, contrast=["condition", condition_b, condition_a], n_cpus=1)
        result.summary()
        result_frame = result.results_df.reindex(de_gene_names)
        stats = result_frame["stat"].to_numpy(dtype=float)
        log2fc_de = result_frame["log2FoldChange"].to_numpy(dtype=float)
        adjusted_p = result_frame["padj"].to_numpy(dtype=float)

    rows = []
    group = f"{request.get('cell_type_name', 'Selected cells')} · {condition_b} vs {condition_a}"
    iter_genes = de_gene_names if replicated else list(gene_names)
    for index, gene in enumerate(iter_genes):
        if replicated:
            p_value = adjusted_p[index]
            fold_change = float(log2fc_de[index])
            stat_val = float(stats[index]) if np.isfinite(stats[index]) else fold_change
        else:
            p_value = np.nan
            fold_change = float(log2fc[index])
            stat_val = fold_change
        if not np.isfinite(fold_change) or not np.isfinite(stat_val):
            # Keep the output contract strict even when a degenerate
            # pseudobulk fit returns an undefined fold change/statistic.
            continue
        pct_target = float(pct_b[index])
        pct_reference = float(pct_a[index])
        enriched_pct = pct_target if fold_change >= 0 else pct_reference
        if enriched_pct < filters["min_pct"]:
            continue
        if abs(pct_target - pct_reference) < filters.get("min_diff_pct", 0.0):
            continue
        if filters["only_positive"] and fold_change <= 0:
            continue
        if abs(fold_change) < filters["min_log2fc"]:
            continue
        if replicated and (not np.isfinite(p_value) or p_value > filters["max_p_value_adj"]):
            continue
        rows.append(
            {
                "group": group,
                "gene": str(gene),
                "score": stat_val,
                "log2_fold_change": fold_change,
                "p_value_adj": float(p_value) if replicated and np.isfinite(p_value) else None,
                "pct_target": float(pct_b[index]),
                "pct_reference": float(pct_a[index]),
            }
        )
    table_rows = select_table_marker_rows(rows, filters["max_genes_per_group"])
    if not include_heatmap:
        return table_rows

    # Keep the table's statistical-score ordering separate from the heat map:
    # the latter is selected by effect size so visually strong genes are not
    # lost merely because their test statistic is smaller.
    def heatmap_priority(row):
        adjusted_p = row.get("p_value_adj")
        p_priority = (
            -math.log10(max(float(adjusted_p), 1e-300))
            if adjusted_p is not None and math.isfinite(float(adjusted_p))
            else 0.0
        )
        return (row["log2_fold_change"], p_priority, str(row["gene"]))

    heatmap_rows = sorted(table_rows, key=heatmap_priority, reverse=True)[:HEATMAP_MARKERS_PER_GROUP]
    return {"results": table_rows, "heatmapResults": heatmap_rows}


def main():
    if len(sys.argv) != 4:
        raise RuntimeError("Usage: run_differential.py <source> <format> <request.json>")

    import numpy as np
    import pandas as pd
    import scanpy as sc

    source, data_format, request_path = sys.argv[1:]
    request = load_differential_request(request_path)
    filters = parse_differential_filters(request)

    groups = request.get("groups", [])
    mode = request.get("mode", "pairwise")
    if mode not in {"pairwise", "find_all", "pseudobulk"}:
        raise RuntimeError(f"Unsupported differential mode: {mode}")
    test_method = normalize_test_method(request.get("test_method", request.get("testMethod")))
    if mode == "pseudobulk":
        source_adata = load_source(source, data_format)
        try:
            result = pseudobulk_results(
                source_adata,
                request.get("records", []),
                request,
                filters,
                include_heatmap=bool(request.get("include_heatmap", False)),
            )
        finally:
            if getattr(source_adata, "isbacked", False):
                source_adata.file.close()
            del source_adata
        emit_json_result(result)
        return
    if mode == "pairwise":
        groups = [
            {"name": "target", "indices": request.get("target", [])},
            {"name": "reference", "indices": request.get("reference", [])},
        ]
    if len(groups) < 2:
        raise RuntimeError("Differential expression requires at least two groups")
    if any(len(group.get("indices", [])) < 2 for group in groups):
        raise RuntimeError("Differential expression requires at least two cells in each group")

    seen = set()
    indices = []
    labels = []
    for group in groups:
        name = str(group["name"])
        for source_index in group["indices"]:
            source_index = coerce_cell_index(source_index, "Differential cell index")
            if source_index in seen:
                raise RuntimeError("Differential groups must not overlap")
            seen.add(source_index)
            indices.append(source_index)
            labels.append(name)

    source_adata = load_source(source, data_format)
    try:
        source_cell_count = source_adata.n_obs
        if indices and (min(indices) < 0 or max(indices) >= source_cell_count):
            raise RuntimeError("Differential group contains an out-of-range cell index")
        subset, normalize_input = select_expression_adata(source_adata, indices)
    finally:
        if getattr(source_adata, "isbacked", False):
            source_adata.file.close()
        del source_adata
    subset.var_names_make_unique()
    category_order = [str(group["name"]) for group in groups]
    subset.obs["comparison"] = pd.Categorical(labels, categories=category_order)

    pct_by_group = {}
    pct_by_reference = {}
    names = [str(name) for name in subset.var_names]
    label_array = np.asarray(labels)
    for name in category_order:
        mask = label_array == name
        group_pct = nonzero_fraction(subset.X[mask])
        reference_pct = nonzero_fraction(subset.X[~mask])
        pct_by_group[name] = dict(zip(names, group_pct))
        pct_by_reference[name] = dict(zip(names, reference_pct))

    if normalize_input:
        sc.pp.normalize_total(subset, target_sum=1e4)
        sc.pp.log1p(subset)
    ranked_groups = ["target"] if mode == "pairwise" else category_order
    rank_kwargs = {
        "groups": ranked_groups,
        "reference": "reference" if mode == "pairwise" else "rest",
        "method": test_method,
        "pts": True,
        # Keep complete statistic arrays available to populate every group
        # value in the heat map. Candidate selection ranks those candidates
        # by the statistical score, consistent with the table's default.
        "n_genes": subset.n_vars,
    }
    sc.tl.rank_genes_groups(subset, "comparison", **rank_kwargs)

    rows = []
    ranked = subset.uns["rank_genes_groups"]
    for group in ranked_groups:
        add_ranked_rows(rows, ranked, group, pct_by_group[group], pct_by_reference[group], filters)
    # Select the heat-map panel from the complete Scanpy-ranked arrays rather
    # than the score-capped table rows: the cap could hide a strong marker
    # before the heat-map selector sees it. Both panels rank candidates by
    # the statistical score, so the heat map stays consistent with the table
    # while keeping its own complete candidate set.
    heatmap_genes = select_heatmap_genes_from_ranked(
        ranked,
        ranked_groups,
        pct_by_group,
        pct_by_reference,
        filters,
        markers_per_group=HEATMAP_MARKERS_PER_GROUP,
        maximum_genes=max(60, len(ranked_groups) * HEATMAP_MARKERS_PER_GROUP),
    )
    if mode == "find_all":
        heatmap_rows = build_heatmap_rows(
            heatmap_genes,
            ranked,
            ranked_groups,
            pct_by_group,
            pct_by_reference,
        )
        emit_json_result({"results": rows, "heatmapResults": heatmap_rows})
    elif mode == "pairwise":
        # Keep the table contract (target-ranked rows) while returning a
        # second, sign-reversed reference column for the heat map.  Older
        # projects may still have only the table rows persisted; the client
        # keeps a one-column fallback for those records.
        heatmap_rows = build_pairwise_heatmap_rows(
            heatmap_genes,
            ranked,
            pct_by_group["target"],
            pct_by_reference["target"],
        )
        if request.get("include_heatmap", True):
            emit_json_result({"results": rows, "heatmapResults": heatmap_rows})
        else:
            # Preserve the legacy non-binary command's array response.
            emit_json_result(rows)
    else:
        emit_json_result(rows)


if __name__ == "__main__":
    main()
