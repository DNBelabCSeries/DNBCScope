#!/usr/bin/env python3
"""
Generate scanpy plots (dotplot, violin, matrixplot) from expression data.

Reads JSON from stdin with:
  - expression: { "geneA": [val, val, ...], "geneB": [...] }
  - clusters: [0, 1, 0, 2, ...]  (cluster index per cell)
  - cluster_names: ["C0", "C1", ...]
  - group_by: "cluster" | "sample" | custom column name
  - samples: [0, 1, 0, ...]  (optional, sample index per cell)
  - sample_names: ["S1", "S2", ...]
  - groups: [0, 1, ...]  (optional)
  - group_names: ["G1", "G2", ...]
  - custom_columns: { "col_name": [0, 1, ...], ... }  (optional)
  - plot_type: "dotplot" | "violin" | "matrixplot"

Outputs JSON with: { "image": "data:image/png;base64,..." }
"""

import base64
import io
import json
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scanpy as sc
import numpy as np
import anndata as ad
import pandas as pd


def generate_plot_from_data(payload: dict) -> bytes:
    start = time.time()

    expression = payload["expression"]  # dict: gene_name -> list[float]
    clusters = np.array(payload["clusters"], dtype=int)
    cluster_names = payload["cluster_names"]
    plot_type = payload["plot_type"]
    group_by = payload.get("group_by", "cluster")

    # Build obs dataframe
    n_cells = len(clusters)
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_cells)])
    obs["cluster"] = pd.Categorical(
        [cluster_names[c] if c < len(cluster_names) else f"C{c}" for c in clusters]
    )

    # Add sample/group if provided
    if "samples" in payload and "sample_names" in payload:
        samples = np.array(payload["samples"], dtype=int)
        sample_names = payload["sample_names"]
        obs["sample"] = pd.Categorical(
            [sample_names[s] if s < len(sample_names) else f"S{s}" for s in samples]
        )

    if "groups" in payload and "group_names" in payload:
        groups = np.array(payload["groups"], dtype=int)
        group_names = payload["group_names"]
        obs["group"] = pd.Categorical(
            [group_names[g] if g < len(group_names) else f"G{g}" for g in groups]
        )

    if "custom_columns" in payload:
        for col_name, col_vals in payload["custom_columns"].items():
            unique_vals = sorted(set(col_vals))
            obs[col_name] = pd.Categorical(
                [str(v) for v in col_vals]
            )

    # Determine groupby column
    groupby_col = "cluster"
    if group_by == "sample" and "sample" in obs.columns:
        groupby_col = "sample"
    elif group_by == "group" and "group" in obs.columns:
        groupby_col = "group"
    elif group_by.startswith("custom:"):
        col_name = group_by.split(":", 1)[1]
        if col_name in obs.columns:
            groupby_col = col_name
    elif "cluster" in obs.columns:
        groupby_col = "cluster"

    # Build expression matrix (cells × genes)
    gene_names = list(expression.keys())
    X = np.array([expression[g] for g in gene_names], dtype=np.float32).T  # (n_cells, n_genes)

    var = pd.DataFrame(index=gene_names)
    adata = ad.AnnData(X=X, obs=obs, var=var)

    # Set up matplotlib
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 10,
        "font.family": "sans-serif",
    })

    n_genes = len(gene_names)

    if plot_type == "dotplot":
        fig_height = max(3, 1.5 + n_genes * 0.4)
        fig, ax = plt.subplots(figsize=(7, fig_height))
        sc.pl.dotplot(
            adata, var_names=gene_names, groupby=groupby_col,
            show=False, ax=ax, standard_scale="var",
        )
        fig.tight_layout()

    elif plot_type == "violin":
        ncols = min(n_genes, 4)
        nrows = (n_genes + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
        for i, gene in enumerate(gene_names):
            r, c = divmod(i, ncols)
            sc.pl.violin(
                adata, keys=gene, groupby=groupby_col,
                show=False, ax=axes[r][c], rotation=45,
            )
        for i in range(n_genes, nrows * ncols):
            r, c = divmod(i, ncols)
            axes[r][c].set_visible(False)
        fig.tight_layout()

    elif plot_type == "matrixplot":
        fig_height = max(3, 1.5 + n_genes * 0.4)
        fig, ax = plt.subplots(figsize=(7, fig_height))
        sc.pl.matrixplot(
            adata, var_names=gene_names, groupby=groupby_col,
            show=False, ax=ax, standard_scale="var",
        )
        fig.tight_layout()

    else:
        raise ValueError(f"Unknown plot type: {plot_type}")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    elapsed = time.time() - start
    print(f"[PLOT] {plot_type} generated in {elapsed:.2f}s ({n_genes} genes, groupby={groupby_col})", file=sys.stderr)

    return buf.read()


def main():
    payload = json.load(sys.stdin)
    png_bytes = generate_plot_from_data(payload)
    b64 = base64.b64encode(png_bytes).decode("ascii")
    print(json.dumps({"image": f"data:image/png;base64,{b64}"}))


if __name__ == "__main__":
    main()
