#!/usr/bin/env python3
"""Convert an h5ad file into MTX + TSV files for DNBCScope to load."""

import sys
import os
from pathlib import Path


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
    if len(sys.argv) < 3:
        print("Usage: convert_h5ad.py <input.h5ad> <output_dir>", file=sys.stderr)
        sys.exit(1)

    h5ad_path = _normalize_windows_path(sys.argv[1])
    output_dir = Path(_normalize_windows_path(sys.argv[2]))
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import anndata as ad
        import scipy.io
        import scipy.sparse
    except ImportError as e:
        print(
            "Missing dependencies. Install with: pip install anndata scipy\n"
            f"Error: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Reading {h5ad_path}...", file=sys.stderr)
    adata = ad.read_h5ad(h5ad_path)

    if scipy.sparse.issparse(adata.X):
        matrix = adata.X.T.tocsc()
    else:
        matrix = scipy.sparse.csc_matrix(adata.X.T)

    matrix_path = output_dir / "matrix.mtx"
    features_path = output_dir / "features.tsv"
    barcodes_path = output_dir / "barcodes.tsv"
    partial_matrix = output_dir / ".dnbc-convert-matrix.mtx"
    partial_features = output_dir / ".dnbc-convert-features.tsv"
    partial_barcodes = output_dir / ".dnbc-convert-barcodes.tsv"
    print(f"Writing {matrix_path}...", file=sys.stderr)
    try:
        # Keep all conversion writes under hidden temporary names. A cancelled
        # conversion must not leave a plausible-looking but truncated MTX
        # directory that a later project open would accept.
        scipy.io.mmwrite(str(partial_matrix), matrix)

        # Pandas 2 may represent valid string indexes using ``StringDtype``
        # rather than ``object``. Checking the dtype used to replace genuine
        # gene names with 0, 1, 2 … during h5ad → MTX conversion. AnnData
        # already exposes the intended identifiers through var_names/obs_names;
        # stringify them without making assumptions about the pandas index dtype.
        gene_names = [str(name) for name in adata.var_names]
        with open(partial_features, "w", encoding="utf-8") as f:
            for i, name in enumerate(gene_names):
                f.write(f"Gene_{i}\t{name}\tGene Expression\n")

        barcode_names = [str(name) for name in adata.obs_names]
        with open(partial_barcodes, "w", encoding="utf-8") as f:
            for bc in barcode_names:
                f.write(bc + "\n")

        os.replace(partial_matrix, matrix_path)
        os.replace(partial_features, features_path)
        os.replace(partial_barcodes, barcodes_path)
    finally:
        for partial in (partial_matrix, partial_features, partial_barcodes):
            try:
                if partial.exists():
                    partial.unlink()
            except OSError:
                pass

    print(
        f"Done: {adata.n_obs} cells, {adata.n_vars} genes -> {output_dir}",
        file=sys.stderr,
    )
    print(str(output_dir))


if __name__ == "__main__":
    main()
