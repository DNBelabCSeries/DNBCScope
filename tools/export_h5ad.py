#!/usr/bin/env python3
"""Export DNBCScope project data to an AnnData h5ad file.

Usage: export_h5ad.py <source_path> <source_format> <output_path> [metadata_manifest]

Reads the original expression matrix (MTX directory or h5ad), attaches
obs/obsm metadata passed as JSON on stdin from the frontend, and writes a
complete h5ad file suitable for downstream analysis or sharing.
"""

import sys
import json
import os
from pathlib import Path

# Keep direct module loading and isolated Windows Python consistent with
# subprocess execution: native_csr.py is a sibling helper next to this script.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)


def _normalize_windows_path(path):
    """Drop a `\\?\\` prefix Rust may add to long Windows paths.

    h5py/HDF5 rejects the extended-length prefix on ordinary paths (WinError
    206).  Keep the prefix only when the logical path is still close to the
    legacy MAX_PATH boundary.  Mirrors expression_source.normalize_windows_path
    but stays dependency-free because this script runs standalone.
    """
    if os.name != "nt" or not isinstance(path, str) or not path.startswith("\\\\?\\"):
        return path
    if path.startswith("\\\\?\\UNC\\"):
        logical = "\\\\" + path[8:]
    else:
        logical = path[4:]
    # Count UTF-16 code units (not code points) to match Rust's
    # `encode_utf16().count()` and expression_source.normalize_windows_path, so
    # non-BMP characters do not cause the prefix to be stripped prematurely.
    windows_length = len(logical.encode("utf-16-le", errors="surrogatepass")) // 2
    return logical if windows_length < 240 else path


def main():
    if len(sys.argv) < 4:
        print(
            "Usage: export_h5ad.py <source_path> <source_format> <output_path>",
            file=sys.stderr,
        )
        sys.exit(1)

    source_path = _normalize_windows_path(sys.argv[1])
    source_format = sys.argv[2]
    output_path = _normalize_windows_path(sys.argv[3])

    try:
        import anndata as ad
        import numpy as np
        import pandas as pd
        import scipy.io
        import scipy.sparse
    except ImportError as e:
        print(
            "Missing dependencies. Install with: pip install anndata scipy pandas numpy\n"
            f"Error: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    binary_manifest = None
    if len(sys.argv) >= 5:
        manifest_path = Path(_normalize_windows_path(sys.argv[4]))
        try:
            binary_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            print(f"Invalid binary metadata manifest: {error}", file=sys.stderr)
            sys.exit(1)
        metadata = None
        print("Using memory-mapped binary export metadata...", file=sys.stderr)
    else:
        # Legacy compatibility path for older renderers and direct script use.
        metadata_raw = sys.stdin.read()
        if not metadata_raw:
            print("No metadata received on stdin", file=sys.stderr)
            sys.exit(1)
        print(f"Parsing metadata ({len(metadata_raw)} bytes)...", file=sys.stderr)
        try:
            metadata = json.loads(metadata_raw)
        except json.JSONDecodeError as e:
            print(f"Invalid metadata JSON: {e}", file=sys.stderr)
            sys.exit(1)

    print(f"Loading expression matrix from {source_path} (format={source_format})...", file=sys.stderr)
    try:
        adata = _load_expression(source_path, source_format)
    except Exception as e:
        print(f"Error loading expression matrix: {e}", file=sys.stderr)
        sys.exit(1)
    if adata is None:
        print(f"Failed to load expression matrix from {source_path}", file=sys.stderr)
        sys.exit(1)

    # Sub-projects and other filtered views store the full parent matrix as the
    # source, but their obs/obsm metadata is aligned to only the active subset
    # of cells. Restrict the matrix to those cells first so its cell count
    # matches the metadata arrays; otherwise the length checks in
    # _apply_metadata would silently drop every column and the exported h5ad
    # would contain no embeddings or cluster labels.
    if binary_manifest is not None:
        payload_path = manifest_path.with_name("metadata.bin")
        indices = _read_binary_array(payload_path, binary_manifest.get("activeCellIndices", {}))
        indices = np.asarray(indices, dtype=np.intp)
        if indices.size != int(binary_manifest.get("cellCount", -1)):
            raise ValueError("Binary active-cell index length does not match cellCount")
    else:
        active_cell_indices = metadata.get("active_cell_indices")
        indices = None
        if active_cell_indices is not None:
            if not isinstance(active_cell_indices, list):
                raise ValueError("active_cell_indices must be a list when provided")
            try:
                indices = [int(i) for i in active_cell_indices]
            except (TypeError, ValueError) as error:
                raise ValueError("active_cell_indices contains an invalid cell index") from error
    if indices is not None:
        index_array = np.asarray(indices)
        if index_array.size and (
            np.any(index_array < 0) or np.any(index_array >= adata.n_obs)
        ):
            raise ValueError(
                f"active_cell_indices contains a cell outside the source matrix (0–{adata.n_obs - 1})"
            )
        # Preserve an intentionally empty selection as an empty exported h5ad
        # rather than silently exporting the full source matrix.
        adata = adata[indices].copy()

    if binary_manifest is not None:
        _apply_binary_metadata(adata, binary_manifest, payload_path)
    else:
        _apply_metadata(adata, metadata)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing h5ad to {output}...", file=sys.stderr)
    # Commit atomically so cancelling the tracked exporter (or a disk-full
    # failure) cannot leave a truncated file at the user's chosen path.
    partial_output = output.parent / ".dnbc-export-partial.h5ad"
    try:
        adata.write_h5ad(str(partial_output), compression="lzf")
        os.replace(partial_output, output)
    finally:
        try:
            if partial_output.exists():
                partial_output.unlink()
        except OSError:
            pass

    print(
        f"Done: {adata.n_obs} cells, {adata.n_vars} genes -> {output}",
        file=sys.stderr,
    )
    print(str(output))


def _load_expression(source_path, source_format):
    """Load expression matrix based on source format. Returns AnnData or None."""
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scipy.io
    import scipy.sparse

    src = Path(source_path)

    if source_format == "native-csr":
        try:
            from native_csr import load_native_csr
            return load_native_csr(str(src))
        except (OSError, KeyError, TypeError, ValueError, RuntimeError, json.JSONDecodeError) as error:
            print(f"Invalid native CSR source: {error}", file=sys.stderr)
            return None

    if source_format == "h5ad":
        if not src.is_file():
            print(f"h5ad file not found: {src}", file=sys.stderr)
            return None
        return ad.read_h5ad(str(src))

    # MTX directory: read matrix.mtx(.gz) + features.tsv(.gz) + barcodes.tsv(.gz)
    if source_format in ("mtx", "dnbc", "dnbc-meta"):
        matrix_path = _find_file(src, ["matrix.mtx", "matrix.mtx.gz"])
        features_path = _find_file(src, ["features.tsv", "features.tsv.gz", "genes.tsv", "genes.tsv.gz"])
        barcodes_path = _find_file(src, ["barcodes.tsv", "barcodes.tsv.gz"])
        if not matrix_path:
            print(f"matrix.mtx(.gz) not found in {src}", file=sys.stderr)
            return None
        if not features_path:
            print(f"features.tsv(.gz) not found in {src}", file=sys.stderr)
            return None
        if not barcodes_path:
            print(f"barcodes.tsv(.gz) not found in {src}", file=sys.stderr)
            return None

        # Read matrix (cells x genes after transpose; MTX stores genes x cells)
        import gzip
        opener = gzip.open if str(matrix_path).endswith(".gz") else open
        with opener(matrix_path, "rb") as f:
            matrix = scipy.io.mmread(f, spmatrix=True).tocsr()
        # MTX is genes x cells -> transpose to cells x genes
        matrix = matrix.T.tocsr()

        # Read features
        feat_opener = gzip.open if str(features_path).endswith(".gz") else open
        with feat_opener(features_path, "rt", encoding="utf-8", errors="replace") as f:
            gene_names = []
            for line in f:
                parts = line.rstrip("\n").split("\t")
                # features.tsv format: ID \t Name \t Type (use Name if available)
                gene_names.append(parts[1] if len(parts) >= 2 else parts[0])

        # Read barcodes
        bc_opener = gzip.open if str(barcodes_path).endswith(".gz") else open
        with bc_opener(barcodes_path, "rt", encoding="utf-8", errors="replace") as f:
            barcodes = [line.strip() for line in f if line.strip()]

        n_cells = matrix.shape[0]
        n_genes = matrix.shape[1]
        if len(gene_names) < n_genes:
            gene_names.extend([f"Gene_{i}" for i in range(len(gene_names), n_genes)])
        elif len(gene_names) > n_genes:
            gene_names = gene_names[:n_genes]
        if len(barcodes) < n_cells:
            barcodes.extend([f"Cell_{i}" for i in range(len(barcodes), n_cells)])
        elif len(barcodes) > n_cells:
            barcodes = barcodes[:n_cells]

        adata = ad.AnnData(X=matrix)
        adata.var_names = pd.Index(gene_names)
        adata.obs_names = pd.Index(barcodes)
        return adata

    print(f"Unsupported source format: {source_format}", file=sys.stderr)
    return None


def _find_file(directory, candidates):
    for name in candidates:
        p = directory / name
        if p.exists():
            return p
    return None


def _read_binary_array(payload_path, descriptor):
    """Return a bounded view into the compact renderer payload."""
    import numpy as np

    dtype_names = {
        "u8": np.dtype("u1"),
        "u16": np.dtype("<u2"),
        "u32": np.dtype("<u4"),
        "f32": np.dtype("<f4"),
        "f64": np.dtype("<f8"),
    }
    dtype = dtype_names.get(str(descriptor.get("dtype", "")))
    if dtype is None:
        raise ValueError(f"Unsupported binary metadata dtype: {descriptor.get('dtype')}")
    try:
        offset = int(descriptor["offset"])
        length = int(descriptor["length"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Binary metadata descriptor is incomplete") from error
    if offset < 0 or length < 0:
        raise ValueError("Binary metadata descriptor contains a negative range")
    byte_count = length * dtype.itemsize
    payload_size = payload_path.stat().st_size
    if offset + byte_count > payload_size:
        raise ValueError(
            f"Binary metadata range exceeds payload ({offset}+{byte_count}>{payload_size})"
        )
    if length == 0:
        return np.empty(0, dtype=dtype)
    return np.memmap(payload_path, mode="r", dtype=dtype, offset=offset, shape=(length,))


def _apply_binary_metadata(adata, manifest, payload_path):
    """Attach typed metadata without materialising a giant JSON document."""
    import numpy as np
    import pandas as pd

    n_cells = adata.n_obs
    if int(manifest.get("cellCount", -1)) != n_cells:
        raise ValueError("Binary metadata cellCount does not match the selected matrix")
    for field in manifest.get("obs", []):
        name = str(field.get("name", "")).strip()
        if not name:
            raise ValueError("Binary obs field has no name")
        if field.get("sanitizeName"):
            name = name.replace(" ", "_").replace("(", "").replace(")", "")[:60]
        values = _read_binary_array(payload_path, field.get("data", {}))
        if len(values) != n_cells:
            raise ValueError(f"Binary obs field {name} has the wrong length")
        kind = field.get("kind")
        if kind == "numeric":
            adata.obs[name] = np.asarray(values)
        elif kind == "boolean":
            adata.obs[name] = pd.Categorical(np.asarray(values, dtype=bool))
        elif kind == "categorical":
            categories = field.get("categories", [])
            codes = np.asarray(values, dtype=np.int64)
            if codes.size and (codes.min() < 0 or codes.max() >= len(categories)):
                raise ValueError(f"Binary categorical field {name} contains an invalid code")
            adata.obs[name] = pd.Categorical.from_codes(codes, categories=categories)
        else:
            raise ValueError(f"Unsupported binary obs field kind: {kind}")

    for embedding in manifest.get("obsm", []):
        name = str(embedding.get("name", "")).strip()
        columns = int(embedding.get("columns", 0))
        values = _read_binary_array(payload_path, embedding.get("data", {}))
        if not name or columns <= 0 or len(values) != n_cells * columns:
            raise ValueError("Binary embedding descriptor is invalid")
        adata.obsm[name] = np.asarray(values).reshape(n_cells, columns)

    for key, value in manifest.get("uns", {}).items():
        adata.uns[key] = value


def _apply_metadata(adata, metadata):
    """Apply obs/obsm fields from metadata dict to AnnData object."""
    import numpy as np
    import pandas as pd

    n_cells = adata.n_obs

    # --- obs fields (per-cell metadata) ---
    obs_fields = metadata.get("obs", {})

    # Cluster assignment
    if "cluster" in obs_fields:
        clusters = obs_fields["cluster"]
        if len(clusters) == n_cells:
            adata.obs["cluster"] = pd.Categorical(clusters)

    # Cluster name (string label)
    if "cluster_name" in obs_fields:
        names = obs_fields["cluster_name"]
        if len(names) == n_cells:
            adata.obs["cluster_name"] = pd.Categorical(names)

    # Sample
    if "sample" in obs_fields:
        samples = obs_fields["sample"]
        if len(samples) == n_cells:
            adata.obs["sample"] = pd.Categorical(samples)

    # Group
    if "group" in obs_fields:
        groups = obs_fields["group"]
        if len(groups) == n_cells:
            adata.obs["group"] = pd.Categorical(groups)

    # QC metrics
    for field in ("n_genes", "total_counts", "mitochondrial_percent", "doublet_score"):
        if field in obs_fields:
            values = obs_fields[field]
            if len(values) == n_cells:
                adata.obs[field] = np.array(values, dtype=float)

    if "predicted_doublet" in obs_fields:
        values = obs_fields["predicted_doublet"]
        if len(values) == n_cells:
            adata.obs["predicted_doublet"] = pd.Categorical(
                [bool(v) if v is not None else False for v in values]
            )

    # VDJ receptor status
    if "receptor_status" in obs_fields:
        values = obs_fields["receptor_status"]
        if len(values) == n_cells:
            adata.obs["receptor_status"] = pd.Categorical(values)

    if "tcr_clonotype" in obs_fields:
        values = obs_fields["tcr_clonotype"]
        if len(values) == n_cells:
            adata.obs["tcr_clonotype"] = pd.Categorical(values)

    if "bcr_clonotype" in obs_fields:
        values = obs_fields["bcr_clonotype"]
        if len(values) == n_cells:
            adata.obs["bcr_clonotype"] = pd.Categorical(values)

    # Custom groups (each as a separate categorical obs column)
    custom_groups = obs_fields.get("custom_groups", [])
    for cg in custom_groups:
        name = cg.get("name", "custom_group")
        values = cg.get("values", [])
        if len(values) == n_cells:
            # Sanitize column name
            col = name.replace(" ", "_").replace("(", "").replace(")", "")[:60]
            adata.obs[col] = pd.Categorical(values)

    # --- obsm fields (embeddings) ---
    obsm_fields = metadata.get("obsm", {})

    if "X_umap" in obsm_fields:
        umap = obsm_fields["X_umap"]
        if len(umap) == n_cells and all(len(row) == 2 for row in umap):
            adata.obsm["X_umap"] = np.array(umap, dtype=float)

    if "X_tsne" in obsm_fields:
        tsne = obsm_fields["X_tsne"]
        if len(tsne) == n_cells and all(len(row) == 2 for row in tsne):
            adata.obsm["X_tsne"] = np.array(tsne, dtype=float)

    # --- uns fields (project-level metadata) ---
    uns_fields = metadata.get("uns", {})
    for key, value in uns_fields.items():
        adata.uns[key] = value


if __name__ == "__main__":
    main()
