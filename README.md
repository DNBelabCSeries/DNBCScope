# DNBCScope

> **Beta test build** — feedback and testing are welcome.

Local single-cell RNA-seq analysis desktop app for QC, clustering, annotation,
differential expression, and figure export. **All data stays on your machine.**

---

## Features

- Create single-sample or multi-sample projects from Matrix Market data.
- Import processed `.h5ad` embeddings, metadata, categories, and expression sources.
- Run QC, Scrublet doublet detection, normalization, HVG, PCA, neighbors, Leiden, UMAP/t-SNE, and Harmony integration.
- Explore large cell counts with WebGL rendering and lazy binary loading.
- Select cells manually or by reusable category, expression, and QC rules.
- Annotate clusters with scType or bundled/downloadable CellTypist models.
- Run cell-level differential expression and multi-sample pseudobulk comparisons.
- Export project data, tables, and PNG/SVG figures.

## Getting started

Prebuilt installers are provided — **no compilation needed**.

### Where files live

| Item | Location |
| --- | --- |
| App | Installed like any desktop app (launcher: `DNBCScope`). |
| Project data | The folder you pick when creating/opening a project (metadata, cache, exports live there). |
| Recent list | Stored in the app's per-user local storage, not inside any project. |
| Analysis cache | Inside each project folder; safe to delete to reclaim space (raw input untouched). |
| Exports | Written to wherever you choose in the export dialog. |

If a project will not open, first check that its folder still exists at the path
shown in the recent-projects list and that you have read/write access to it.

## About the unsigned builds

The installers are **not code-signed** — expected for this beta test release, as
we have not yet applied for code signing (planned for a later stable release).
The warnings below do **not** mean the app is unsafe; they only mean the
publisher identity is not verified by the OS vendor.

**macOS (Gatekeeper)**
: warning: *“…cannot be opened because the developer cannot be verified.”*
: fix: Open **System Settings → Privacy & Security**, then click **Open Anyway**.
  This option appears only within ~1 hour of the block; if it is gone,
  re-trigger the popup by opening the app again.

**Windows (SmartScreen)**
: warning: *“Windows protected your PC.”*
: fix: Click **More info → Run anyway**.

## Privacy

All analysis runs locally on your computer. DNBCScope does not require an account
or network access to function, and no project data leaves your machine.

## Help

The full workflow, plotting, and export options are documented in the app's
built-in manual — open **Help** from the app menu.
