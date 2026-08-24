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

- macOS: [DNBCScope_0.1.0_aarch64.dmg](https://github.com/DNBelabCSeries/DNBCScope/releases/download/0.1.0/DNBCScope_0.1.0_aarch64.dmg)
- Windows: [DNBCScope_0.1.0_x64-setup.exe](https://github.com/DNBelabCSeries/DNBCScope/releases/download/0.1.0/DNBCScope_0.1.0_x64-setup.exe)

All installers are distributed via the [Releases](https://github.com/DNBelabCSeries/DNBCScope/releases) page. See *About the unsigned builds* below for launch steps.

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

### 关于未签名安装包（中文）

安装包**未经代码签名**——这是本测试版（beta）的预期情况，因为我们尚未申请代码签名（计划在后续稳定版中提供）。下面的警告**并不代表应用不安全**，它们只是表示操作系统厂商尚未验证发布者身份。

**macOS（Gatekeeper）**
: 警告：*“…无法打开，因为无法验证开发者。”*
: 解决方法：打开 **系统设置 → 隐私与安全性**，然后点击 **仍要打开**。该选项仅在被拦截后约 1 小时内出现；若已消失，再次打开应用即可重新触发弹窗。

**Windows（SmartScreen）**
: 警告：*“Windows 已保护你的电脑。”*
: 解决方法：点击 **更多信息 → 仍要运行**。

## Privacy

All analysis runs locally on your computer. DNBCScope does not require an account
or network access to function, and no project data leaves your machine.

## Help

The full workflow, plotting, and export options are documented in the app's
built-in manual — open **Help** from the app menu.
