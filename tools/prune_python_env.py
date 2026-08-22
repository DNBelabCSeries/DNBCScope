#!/usr/bin/env python3
"""Remove Python build-time artifacts before packaging DNBCScope.

The application only needs the interpreter and runtime packages. Test fixtures
and pip are build-time tools, while bytecode is deliberately retained by
default: the desktop app starts a fresh Python process for analysis tasks, and
precompiled modules make Scanpy/Scrublet cold starts substantially faster.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path


# ``.pyi`` looks development-only, but packages such as scikit-image use its
# stubs at runtime through lazy_loader. Keep all stubs unless a package-specific
# smoke test proves otherwise.
DEVELOPMENT_SUFFIXES = {
    ".a",
    ".c",
    ".cpp",
    ".exp",
    ".h",
    ".lib",
    ".pdb",
    ".pxd",
    ".pxi",
    ".pyx",
}
SAMPLE_DATA_SUFFIXES = {
    ".csv",
    ".dta",
    ".h5ad",
    ".jpg",
    ".jpeg",
    ".npy",
    ".npz",
    ".png",
    ".tif",
    ".tiff",
    ".xml",
}


def path_size(path: Path) -> int:
    if path.is_file() or path.is_symlink():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for child in path.rglob("*"):
        try:
            if child.is_file() or child.is_symlink():
                total += child.stat().st_size
        except OSError:
            pass
    return total


def remove(path: Path, dry_run: bool) -> int:
    size = path_size(path)
    if not dry_run:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
    return size


def runtime_python_path(env: Path) -> Path:
    """Resolve app-local Windows and Unix venv/standalone layouts."""
    if os.name == "nt":
        for candidate in (env / "python.exe", env / "Scripts" / "python.exe"):
            if candidate.is_file():
                return candidate
        return env / "python.exe"
    return env / "bin" / "python"


def site_packages_dirs(env: Path) -> list[Path]:
    candidates = [env / "Lib" / "site-packages"]
    candidates.extend(env.glob("lib/python*/site-packages"))
    return [path for path in candidates if path.is_dir()]


def add_development_targets(env: Path, targets: set[Path]) -> None:
    """Select compiler-only files without touching license metadata."""
    for path in (env / "include", env / "libs"):
        if path.exists():
            targets.add(path)
    for site_packages in site_packages_dirs(env):
        for path in site_packages.rglob("*"):
            if path.is_file() and path.suffix.lower() in DEVELOPMENT_SUFFIXES:
                # LICENSE/NOTICE files live in dist-info and use textual
                # suffixes, so they are intentionally outside this whitelist.
                targets.add(path)


def add_sample_data_targets(env: Path, targets: set[Path]) -> None:
    """Remove bundled library demos DNBCScope never exposes to users.

    Keep Python modules in these directories so importing scanpy/skimage and
    their public namespaces remains valid. Only known dataset payloads are
    selected; arbitrary package ``data`` directories may contain runtime
    resources and must not be pruned globally.
    """
    for site_packages in site_packages_dirs(env):
        sample_roots = (
            site_packages / "scanpy" / "datasets",
            site_packages / "skimage" / "data",
            site_packages / "statsmodels" / "datasets",
            site_packages / "matplotlib" / "mpl-data" / "sample_data",
        )
        for root in sample_roots:
            if not root.is_dir():
                continue
            for path in root.rglob("*"):
                if path.is_file() and path.suffix.lower() in SAMPLE_DATA_SUFFIXES:
                    targets.add(path)
        for relative in (
            Path("sklearn/datasets/data"),
            Path("sklearn/datasets/descr"),
            Path("sklearn/datasets/images"),
        ):
            path = site_packages / relative
            if path.is_dir():
                targets.add(path)


def add_build_tool_targets(env: Path, targets: set[Path]) -> None:
    """Select installers, compiler frontends and stdlib developer utilities."""
    for bin_dir in (env / "bin", env / "Scripts"):
        if not bin_dir.is_dir():
            continue
        for pattern in ("pip*", "cython*", "idle*", "pydoc*", "python*-config"):
            for path in bin_dir.glob(pattern):
                if path.is_file() or path.is_symlink():
                    targets.add(path)
    for stdlib in env.glob("lib/python*"):
        if not stdlib.is_dir():
            continue
        for name in ("ensurepip", "idlelib", "tkinter", "turtledemo", "pydoc_data"):
            path = stdlib / name
            if path.exists():
                targets.add(path)
        for path in stdlib.glob("config-*"):
            if path.is_dir():
                targets.add(path)
    pkgconfig = env / "lib" / "pkgconfig"
    if pkgconfig.is_dir():
        targets.add(pkgconfig)


def strip_macos_native_files(env: Path, dry_run: bool) -> tuple[int, int]:
    """Strip local symbols and restore an ad-hoc signature on each Mach-O.

    Wheels commonly ship already signed. ``strip`` invalidates that signature,
    and macOS then kills Python with SIGKILL while importing packages such as
    h5py. Re-sign the temporary file before replacing the working original; if
    either operation fails, leave the original untouched.
    """
    if platform.system() != "Darwin":
        return 0, 0
    strip = shutil.which("strip")
    codesign = shutil.which("codesign")
    if not strip or not codesign:
        print(
            "Warning: macOS strip/codesign is unavailable; "
            "native files were left unchanged."
        )
        return 0, 0

    candidates = [
        path
        for path in env.rglob("*")
        if path.is_file() and path.suffix.lower() in {".so", ".dylib"}
    ]
    if dry_run:
        # Dry-run cannot know the compressed symbol-table size without doing
        # the operation. Report candidates without overstating byte savings.
        return len(candidates), 0

    saved = 0
    stripped = 0
    with tempfile.TemporaryDirectory(prefix="dnbcscope-strip-") as temp_value:
        temp = Path(temp_value)
        for index, path in enumerate(candidates):
            before = path.stat().st_size
            candidate = temp / f"{index}{path.suffix}"
            shutil.copy2(path, candidate)
            result = subprocess.run(
                [strip, "-x", str(candidate)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if result.returncode != 0 or not candidate.is_file():
                continue
            sign_result = subprocess.run(
                [codesign, "--force", "--sign", "-", str(candidate)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if sign_result.returncode != 0:
                continue
            after = candidate.stat().st_size
            if after >= before:
                continue
            candidate.replace(path)
            saved += before - after
            stripped += 1
    return stripped, saved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", type=Path, required=True, help="Path to bundled python_env")
    parser.add_argument("--dry-run", action="store_true", help="Report reclaimable space without modifying files")
    parser.add_argument(
        "--remove-bytecode",
        action="store_true",
        help="Remove __pycache__ and .pyc files (not recommended for release builds)",
    )
    parser.add_argument(
        "--compile-bytecode",
        action="store_true",
        help="Compile runtime Python modules after pruning for consistent fast cold starts",
    )
    parser.add_argument(
        "--warm-scrublet-data",
        type=Path,
        help="Run Scrublet once on this bundled MTX dataset to cache only its real import path",
    )
    parser.add_argument(
        "--remove-build-tools",
        action="store_true",
        help="Also remove pip and Cython from the distributable runtime (not needed by DNBCScope)",
    )
    parser.add_argument(
        "--remove-development-files",
        action="store_true",
        help="Remove compiler headers, static libraries and Cython source files",
    )
    parser.add_argument(
        "--remove-sample-data",
        action="store_true",
        help="Remove curated third-party demo datasets that DNBCScope never loads",
    )
    parser.add_argument(
        "--strip-native",
        action="store_true",
        help="Strip local symbols from macOS native extensions (a no-op on other platforms)",
    )
    args = parser.parse_args()
    env = args.env.resolve()
    if not env.is_dir():
        raise SystemExit(f"Python environment does not exist: {env}")

    targets: set[Path] = set()
    if args.remove_bytecode:
        for path in env.rglob("__pycache__"):
            if path.is_dir():
                targets.add(path)
        for path in env.rglob("*.pyc"):
            if path.is_file():
                targets.add(path)
    for name in ("tests", "test"):
        for path in env.rglob(name):
            if path.is_dir():
                targets.add(path)

    if args.remove_build_tools:
        for path in env.rglob("pip"):
            if path.is_dir() and path.parent.name == "site-packages":
                targets.add(path)
        for path in env.rglob("pip-*.dist-info"):
            if path.is_dir() and path.parent.name == "site-packages":
                targets.add(path)
        for path in env.rglob("Cython"):
            if path.is_dir() and path.parent.name == "site-packages":
                targets.add(path)
        for pattern in ("cython.py", "cython-*.dist-info"):
            for path in env.rglob(pattern):
                if path.parent.name == "site-packages":
                    targets.add(path)
        add_build_tool_targets(env, targets)

    if args.remove_development_files:
        add_development_targets(env, targets)
    if args.remove_sample_data:
        add_sample_data_targets(env, targets)

    # Keep only root targets. A test directory can contain __pycache__ entries;
    # counting both would overstate dry-run savings and attempt duplicate work.
    selected: list[Path] = []
    for path in sorted(targets, key=lambda path: (len(path.parts), str(path))):
        if not any(parent in path.parents for parent in selected):
            selected.append(path)
    removed = 0
    removed_paths = 0
    for path in selected:
        if not path.exists():
            continue
        removed += remove(path, args.dry_run)
        removed_paths += 1

    mode = "Would remove" if args.dry_run else "Removed"
    print(f"{mode} {removed_paths} Python packaging artifacts ({removed / 1024 / 1024:.1f} MB).")
    if args.strip_native:
        stripped, strip_saved = strip_macos_native_files(env, args.dry_run)
        if args.dry_run:
            print(f"Would inspect {stripped} macOS native files for removable local symbols.")
        else:
            print(
                f"Stripped {stripped} macOS native files "
                f"({strip_saved / 1024 / 1024:.1f} MB)."
            )
    if args.compile_bytecode and not args.dry_run:
        print("Precompiling Python runtime bytecode for analysis cold starts...")
        runtime_python = runtime_python_path(env)
        if not runtime_python.is_file():
            raise SystemExit(f"Bundled Python executable is missing: {runtime_python}")
        subprocess.run(
            [str(runtime_python), "-m", "compileall", "-q", "-f", str(env)],
            check=True,
        )
    if args.warm_scrublet_data and not args.dry_run:
        data_path = args.warm_scrublet_data.resolve()
        if not data_path.is_dir():
            raise SystemExit(f"Scrublet warm-up data does not exist: {data_path}")
        runtime_python = runtime_python_path(env)
        if not runtime_python.is_file():
            raise SystemExit(f"Bundled Python executable is missing: {runtime_python}")
        project_root = Path(__file__).resolve().parent.parent
        bootstrap = (
            "from pathlib import Path; import sys; "
            "root = Path(sys.argv[2]); "
            "exec((root / 'tools' / 'expression_source.py').read_text(encoding='utf-8') + '\\n' "
            "+ (root / 'tools' / 'run_scrublet.py').read_text(encoding='utf-8'), "
            "{'__name__': '__main__', '__file__': str(root / 'tools' / 'run_scrublet.py')})"
        )
        print("Warming the Scrublet runtime with bundled PBMC3k data...")
        warm_env = os.environ.copy()
        warm_env.update(
            {
                "MPLCONFIGDIR": os.path.join(tempfile.gettempdir(), "dnbcscope-mpl"),
                "PYTHONUTF8": "1",
                "PYTHONIOENCODING": "utf-8",
            }
        )
        with subprocess.Popen(
            # Safe-path mode keeps unrelated directories in the build working
            # tree (for example a frontend coverage report) from being
            # mistaken for Python packages during the `-c` bootstrap.
            [str(runtime_python), "-P", "-c", bootstrap, str(data_path), str(project_root)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=warm_env,
        ) as process:
            _, stderr = process.communicate()
            if process.returncode != 0:
                raise SystemExit(f"Scrublet runtime warm-up failed:\n{stderr}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
