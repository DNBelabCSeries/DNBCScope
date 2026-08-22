#!/usr/bin/env python3
"""
DNBCScope — Package the Python scientific-computing environment into a
standalone directory that ships with the application.

The generated environment is placed under resources/python_env/ and embedded
inside the app by Tauri's bundle.resources; find_python discovers it at runtime.

Usage:
    # macOS (Apple Silicon)
    python3 tools/bundle_python.py

    # Windows
    python tools/bundle_python.py

The release builder version is pinned by .python-version. The bundled runtime
version and archive hashes are pinned by .python-version and
requirements/python-runtime.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.parse
import urllib.error
import urllib.request
import venv
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_LOCK = PROJECT_ROOT / "requirements" / "analysis-lock.txt"
PYTHON_VERSION_FILE = PROJECT_ROOT / ".python-version"
RUNTIME_CONFIG_FILE = PROJECT_ROOT / "requirements" / "python-runtime.json"
RUNTIME_MANIFEST = ".dnbcscope-runtime.json"


def load_runtime_config() -> tuple[str, dict]:
    version = PYTHON_VERSION_FILE.read_text(encoding="utf-8").strip()
    if not version or len(version.split(".")) != 3:
        raise RuntimeError(f"Invalid exact Python version in {PYTHON_VERSION_FILE}: {version!r}")
    config = json.loads(RUNTIME_CONFIG_FILE.read_text(encoding="utf-8"))
    asset_names = [
        str(config["windows"]["asset"]),
        *(str(asset["name"]) for asset in config["macos"]["assets"].values()),
    ]
    if any(version not in asset_name for asset_name in asset_names):
        raise RuntimeError(f"Runtime assets do not all match Python {version}: {asset_names}")
    checksums = [
        str(config["pip"]["sha256"]),
        str(config["windows"]["sha256"]),
        *(str(asset["sha256"]) for asset in config["macos"]["assets"].values()),
    ]
    if any(len(checksum) != 64 for checksum in checksums):
        raise RuntimeError(f"Runtime config contains an invalid SHA-256: {RUNTIME_CONFIG_FILE}")
    return version, config


PYTHON_VERSION, RUNTIME_CONFIG = load_runtime_config()
PIP_VERSION = str(RUNTIME_CONFIG["pip"]["version"])
PIP_ASSET = str(RUNTIME_CONFIG["pip"]["asset"])
PIP_URL = str(RUNTIME_CONFIG["pip"]["url"])
PIP_SHA256 = str(RUNTIME_CONFIG["pip"]["sha256"])
SETUPTOOLS_VERSION = str(RUNTIME_CONFIG["buildTools"]["setuptools"])
DEFAULT_PACKAGE_INDEX = str(RUNTIME_CONFIG["packageIndexes"]["primary"])
DEFAULT_PACKAGE_FALLBACK_INDEX = str(RUNTIME_CONFIG["packageIndexes"]["fallback"])
WINDOWS_PYTHON_VERSION = PYTHON_VERSION
WINDOWS_RUNTIME_KIND = str(RUNTIME_CONFIG["windows"]["runtimeKind"])
WINDOWS_ASSET = str(RUNTIME_CONFIG["windows"]["asset"])
WINDOWS_ASSET_URL = str(RUNTIME_CONFIG["windows"]["url"])
WINDOWS_ASSET_SHA256 = str(RUNTIME_CONFIG["windows"]["sha256"])
WINDOWS_ANNOY_WHEEL_SEARCH_ROOTS = tuple(
    str(path) for path in RUNTIME_CONFIG["windows"]["annoyWheelSearchRoots"]
)
WINDOWS_HARMONYPY_WHEEL_SEARCH_ROOTS = tuple(
    str(path) for path in RUNTIME_CONFIG["windows"]["harmonypyWheelSearchRoots"]
)
STANDALONE_RELEASE = str(RUNTIME_CONFIG["macos"]["standaloneRelease"])
STANDALONE_DOWNLOAD_BASE_URL = str(RUNTIME_CONFIG["macos"]["downloadBaseUrl"])
STANDALONE_FALLBACK_DOWNLOAD_BASE_URL = str(
    RUNTIME_CONFIG["macos"]["fallbackDownloadBaseUrl"]
)
STANDALONE_PYTHON_VERSION = PYTHON_VERSION
STANDALONE_MACOS_ASSETS = {
    architecture: (str(asset["name"]), str(asset["sha256"]))
    for architecture, asset in RUNTIME_CONFIG["macos"]["assets"].items()
}


def package_indexes() -> tuple[str, str]:
    return (
        os.environ.get("DNBCSCOPE_PYPI_INDEX", DEFAULT_PACKAGE_INDEX),
        os.environ.get("DNBCSCOPE_PYPI_FALLBACK_INDEX", DEFAULT_PACKAGE_FALLBACK_INDEX),
    )


def run_pip_with_fallback(py: str, args: list[str]) -> None:
    """Try only the fast mirror first; query official PyPI only after failure."""
    primary, fallback = package_indexes()
    for attempt, index in enumerate(dict.fromkeys((primary, fallback)), start=1):
        result = subprocess.run(
            [py, "-m", "pip", *args, "--index-url", index],
            check=False,
        )
        if result.returncode == 0:
            return
        if attempt == 1 and fallback != primary:
            log(f"镜像安装失败，改用备用源重试: {fallback}")
    raise subprocess.CalledProcessError(result.returncode, result.args)


def log(msg: str) -> None:
    print(msg, flush=True)


def python_executable(env_dir: Path) -> Path:
    if platform.system() == "Windows":
        app_local = env_dir / "python.exe"
        if app_local.is_file():
            return app_local
        return env_dir / "Scripts" / "python.exe"
    return env_dir / "bin" / "python"


def pip_executable(env_dir: Path) -> Path:
    if platform.system() == "Windows":
        return env_dir / "Scripts" / "pip.exe"
    return env_dir / "bin" / "pip"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def macos_standalone_asset(machine: str | None = None) -> tuple[str, str]:
    normalized = (machine or platform.machine()).lower()
    if normalized in ("arm64", "aarch64"):
        normalized = "aarch64"
    elif normalized in ("amd64", "x64"):
        normalized = "x86_64"
    try:
        return STANDALONE_MACOS_ASSETS[normalized]
    except KeyError as error:
        supported = ", ".join(sorted(STANDALONE_MACOS_ASSETS))
        raise RuntimeError(
            f"不支持的 macOS CPU 架构 {normalized!r}；当前支持: {supported}"
        ) from error


def runtime_download_cache() -> Path:
    override = os.environ.get("DNBCSCOPE_DOWNLOAD_CACHE")
    if override:
        return Path(override).expanduser().resolve()
    if platform.system() == "Darwin":
        return Path.home() / "Library" / "Caches" / "DNBCScope" / "python"
    if platform.system() == "Windows":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "DNBCScope" / "cache" / "python"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "dnbcscope" / "python"


def download_with_resume(url: str, destination: Path, attempts: int = 5) -> None:
    """Download to a persistent `.part` file and atomically publish it.

    GitHub release downloads redirect to a short-lived Azure URL. A transient
    reset during that redirect used to abort the entire release build and lose
    all progress. Range requests plus a persistent cache make retries cheap.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.part")
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        resume_from = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "DNBCScope-build/1.0"}
        if resume_from:
            headers["Range"] = f"bytes={resume_from}-"
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response:
                status = getattr(response, "status", response.getcode())
                append = resume_from > 0 and status == 206
                mode = "ab" if append else "wb"
                content_length = response.headers.get("Content-Length")
                expected_response_bytes = (
                    int(content_length) if content_length is not None else None
                )
                received = 0
                with partial.open(mode) as handle:
                    while chunk := response.read(1024 * 1024):
                        handle.write(chunk)
                        received += len(chunk)
                if (
                    expected_response_bytes is not None
                    and received != expected_response_bytes
                ):
                    raise OSError(
                        "download ended early: "
                        f"received {received} of {expected_response_bytes} bytes"
                    )
            partial.replace(destination)
            return
        except (OSError, TimeoutError, urllib.error.URLError) as error:
            last_error = error
            if attempt < attempts:
                delay = min(16, 2 ** (attempt - 1))
                log(f"下载中断，{delay} 秒后重试 ({attempt}/{attempts}): {error}")
                time.sleep(delay)

    # macOS always ships curl. It often succeeds in corporate networks where
    # Python's TLS/proxy configuration does not, and it can resume the same
    # persistent partial file.
    curl = shutil.which("curl")
    if curl:
        result = subprocess.run(
            [
                curl,
                "--fail",
                "--location",
                "--retry",
                "5",
                "--retry-all-errors",
                "--connect-timeout",
                "30",
                "--continue-at",
                "-",
                "--output",
                str(partial),
                url,
            ],
            check=False,
        )
        if result.returncode == 0 and partial.is_file():
            partial.replace(destination)
            return
    raise RuntimeError(f"download failed after {attempts} attempts: {last_error}")


def resolve_runtime_archive(
    asset_name: str,
    expected_sha256: str,
    archive_url: str,
    local_archive_env: str,
    platform_label: str,
    fallback_archive_urls: tuple[str, ...] = (),
) -> Path:
    local_archive = os.environ.get(local_archive_env)
    if local_archive:
        source = Path(local_archive).expanduser().resolve()
        if not source.is_file():
            raise RuntimeError(f"指定的 {platform_label} Python 压缩包不存在: {source}")
        actual_sha256 = file_sha256(source)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"{platform_label} Python 下载校验失败："
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        log(f"使用本地 {platform_label} Python 压缩包: {source}")
        return source

    cache_path = runtime_download_cache() / asset_name
    if cache_path.is_file():
        if file_sha256(cache_path) == expected_sha256:
            log(f"复用已校验的 Python 下载缓存: {cache_path}")
            return cache_path
        log(f"删除校验失败的 Python 下载缓存: {cache_path}")
        cache_path.unlink()
    urls = tuple(dict.fromkeys((archive_url, *fallback_archive_urls)))
    failures: list[str] = []
    for index, candidate_url in enumerate(urls):
        try:
            download_with_resume(candidate_url, cache_path)
        except RuntimeError as error:
            failures.append(f"{candidate_url}: {error}")
            if index + 1 < len(urls):
                log(f"{platform_label} Python 镜像下载失败，切换备用源...")
            continue
        actual_sha256 = file_sha256(cache_path)
        if actual_sha256 == expected_sha256:
            return cache_path
        failures.append(
            f"{candidate_url}: SHA-256 expected {expected_sha256}, got {actual_sha256}"
        )
        cache_path.unlink(missing_ok=True)
        if index + 1 < len(urls):
            log(f"{platform_label} Python 镜像校验失败，切换备用源...")
    raise RuntimeError(
        f"下载的 {platform_label} Python 未通过完整性校验：" + " | ".join(failures)
    )


def resolve_macos_runtime_archive(
    asset_name: str,
    expected_sha256: str,
    archive_url: str,
    fallback_archive_urls: tuple[str, ...] = (),
) -> Path:
    return resolve_runtime_archive(
        asset_name,
        expected_sha256,
        archive_url,
        "DNBCSCOPE_STANDALONE_PYTHON_ARCHIVE",
        "macOS",
        fallback_archive_urls,
    )


def create_macos_standalone_env(env_dir: Path) -> None:
    """Install a verified, relocatable CPython distribution on macOS.

    ``venv --copies`` only copies the launcher. The launcher and standard
    library still point at the build machine's Homebrew/Python.org framework,
    which makes the resulting app fail on a clean customer Mac. The stripped
    python-build-standalone archive contains the interpreter and stdlib and is
    explicitly designed to run after its directory is relocated.
    """
    asset_name, expected_sha256 = macos_standalone_asset()
    base_url = os.environ.get(
        "DNBCSCOPE_STANDALONE_PYTHON_BASE_URL", STANDALONE_DOWNLOAD_BASE_URL
    ).rstrip("/")
    archive_url = f"{base_url}/{STANDALONE_RELEASE}/{urllib.parse.quote(asset_name)}"
    fallback_base_url = STANDALONE_FALLBACK_DOWNLOAD_BASE_URL.rstrip("/")
    fallback_archive_url = (
        f"{fallback_base_url}/{STANDALONE_RELEASE}/{urllib.parse.quote(asset_name)}"
    )
    log(f"下载可重定位 macOS Python {STANDALONE_PYTHON_VERSION}...")
    try:
        archive = resolve_macos_runtime_archive(
            asset_name,
            expected_sha256,
            archive_url,
            (fallback_archive_url,),
        )
    except (OSError, RuntimeError) as error:
        raise RuntimeError(
            "无法下载 macOS standalone Python；已尝试断点续传和 curl 后备。"
            "请检查网络后重试，或通过 DNBCSCOPE_STANDALONE_PYTHON_ARCHIVE "
            f"指定官方压缩包。详细原因: {error}"
        ) from error
    actual_sha256 = file_sha256(archive)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "macOS Python 下载校验失败："
            f"expected {expected_sha256}, got {actual_sha256}"
        )

    with tempfile.TemporaryDirectory(prefix="dnbcscope-python-") as temp_dir_value:
        temp_dir = Path(temp_dir_value)
        extract_dir = temp_dir / "extract"
        extract_dir.mkdir()
        with tarfile.open(archive, "r:gz") as bundle:
            bundle.extractall(extract_dir, filter="data")
        runtime_root = extract_dir / "python"
        if not (runtime_root / "bin" / "python").is_file():
            raise RuntimeError("macOS standalone Python archive has an unexpected layout")
        shutil.move(str(runtime_root), env_dir)
    log("macOS standalone Python 已准备完成。")


def create_windows_nuget_env(env_dir: Path) -> None:
    """Extract the official CPython NuGet runtime into the application tree.

    Unlike the embeddable ZIP, CPython's NuGet package includes the standard
    library, development headers and import libraries required to compile
    packages such as Annoy. It is app-local and does not invoke Windows
    Installer, so an existing machine-wide Python cannot cause MSI error 1638.
    Release pruning removes build-only files after native dependencies have
    been installed and verified.
    """
    version = PYTHON_VERSION
    major_minor = ".".join(version.split(".")[:2])
    compact = major_minor.replace(".", "")

    log(f"下载 Windows app-local Python {version} (CPython NuGet)...")
    try:
        package = resolve_runtime_archive(
            WINDOWS_ASSET,
            WINDOWS_ASSET_SHA256,
            WINDOWS_ASSET_URL,
            "DNBCSCOPE_WINDOWS_PYTHON_ARCHIVE",
            "Windows",
        )
    except (OSError, RuntimeError) as error:
        raise RuntimeError(
            f"无法下载并校验 Windows Python {version} ({WINDOWS_ASSET_URL})。"
            "请检查网络，或通过 DNBCSCOPE_WINDOWS_PYTHON_ARCHIVE 指定官方 NuGet 包。"
        ) from error

    with tempfile.TemporaryDirectory(prefix="dnbcscope-python-") as temp_dir_value:
        extract_dir = Path(temp_dir_value) / "extract"
        extract_dir.mkdir()
        try:
            with zipfile.ZipFile(package) as bundle:
                bundle.extractall(extract_dir)
        except (OSError, zipfile.BadZipFile) as error:
            raise RuntimeError(f"Windows Python NuGet 包无法解压: {package}") from error

        runtime_root = extract_dir / "tools"
        if not (runtime_root / "python.exe").is_file():
            raise RuntimeError("Windows Python NuGet 包目录结构异常，缺少 tools/python.exe")
        shutil.move(str(runtime_root), env_dir)

    site_packages = env_dir / "Lib" / "site-packages"
    if not (site_packages / "pip").is_dir():
        # Current official packages already install pip. Keep a fallback for
        # a future CPython NuGet layout that only carries ensurepip's wheel.
        bundled_pips = sorted(
            (env_dir / "Lib" / "ensurepip" / "_bundled").glob("pip-*.whl")
        )
        if len(bundled_pips) != 1:
            raise RuntimeError("Windows Python NuGet 包缺少 pip 和唯一的 ensurepip wheel")
        site_packages.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(bundled_pips[0]) as pip_wheel:
            pip_wheel.extractall(site_packages)

    required = (
        env_dir / "python.exe",
        env_dir / f"python{compact}.dll",
        env_dir / "include" / "Python.h",
        env_dir / "libs" / f"python{compact}.lib",
        env_dir / "Lib" / "site-packages" / "pip",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("Windows app-local Python 不完整，缺少：\n  " + "\n  ".join(missing))
    log("Windows app-local Python（含开发头文件和 import library）已准备完成。")


def create_venv(env_dir: Path, base_python: str) -> None:
    if env_dir.exists():
        log(f"清理已有环境: {env_dir}")
        shutil.rmtree(env_dir)
    if platform.system() == "Windows":
        log(f"创建 Windows app-local Python 环境: {env_dir}")
        create_windows_nuget_env(env_dir)
        return

    if platform.system() == "Darwin":
        log(f"创建可重定位 macOS Python 环境: {env_dir}")
        create_macos_standalone_env(env_dir)
        return

    log(f"创建虚拟环境: {env_dir}")
    # Linux packaging is not currently a supported release target. Keep the
    # existing venv path for development, but do not describe it as a fully
    # relocatable distribution: copied launchers can retain base-prefix links.
    builder = venv.EnvBuilder(with_pip=True, upgrade_deps=True, symlinks=False)
    builder.create(str(env_dir))
    log("虚拟环境创建完成。")


def install_packages(env_dir: Path) -> None:
    # Do not invoke pip.exe to upgrade itself on Windows: the running
    # executable cannot be replaced in-place. Calling pip as a module works
    # on both platforms and also keeps the selected interpreter explicit.
    py = str(python_executable(env_dir))
    if not ANALYSIS_LOCK.is_file():
        raise FileNotFoundError(f"Analysis dependency lock is missing: {ANALYSIS_LOCK}")
    log("升级 pip...")
    run_pip_with_fallback(
        py,
        [
            "install",
            "--upgrade",
            "--disable-pip-version-check",
            "--no-cache-dir",
            f"pip=={PIP_VERSION}",
        ],
    )
    log("安装固定版本的构建工具...")
    run_pip_with_fallback(
        py,
        [
            "install",
            "--disable-pip-version-check",
            "--no-cache-dir",
            f"setuptools=={SETUPTOOLS_VERSION}",
        ],
    )
    requirements = [str(ANALYSIS_LOCK)]
    if platform.system() == "Windows":
        requirements = [str(windows_requirements_without_native_overrides())]
    log("安装科学计算依赖（这一步可能需要几分钟）...")
    run_pip_with_fallback(
        py,
        [
            "install",
            "--disable-pip-version-check",
            "--no-cache-dir",
            "--prefer-binary",
            "--progress-bar",
            "on",
            "--requirement",
            *requirements,
        ],
    )

    if platform.system() == "Windows":
        # The validated CPython 3.12 wheel avoids requiring MSVC/setuptools at
        # release-build time.  Do not silently fall back to a source build:
        # that was the cause of non-reproducible Windows packaging failures.
        annoy_wheel = find_windows_annoy_wheel()
        log(f"安装预构建的 Windows Annoy wheel: {annoy_wheel}")
        subprocess.run(
            [
                py,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-cache-dir",
                "--no-deps",
                "--force-reinstall",
                str(annoy_wheel),
            ],
            check=True,
        )

        harmony_wheel = find_windows_harmonypy_wheel()
        log(f"安装用户预构建的 Windows Harmony wheel: {harmony_wheel}")
        subprocess.run(
            [
                py,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-cache-dir",
                "--no-deps",
                "--force-reinstall",
                str(harmony_wheel),
            ],
            check=True,
        )


def windows_requirements_without_native_overrides() -> Path:
    """Leave Windows-only Annoy/Harmony installation to their explicit paths."""
    output = runtime_download_cache() / "analysis-lock-windows-no-native-overrides.txt"
    lines = ANALYSIS_LOCK.read_text(encoding="utf-8").splitlines()
    excluded = ("annoy==", "harmonypy==")
    filtered = [
        line
        for line in lines
        if not line.strip().lower().startswith(excluded)
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(filtered) + "\n", encoding="utf-8")
    return output


def find_existing_windows_harmonypy() -> Path | None:
    """Locate a previously built and validated harmonypy package on this
    machine so a rebuild can reuse it when no prebuilt wheel is available.

    Returns the package directory (site-packages/harmonypy) or None.
    """
    candidates = [
        PROJECT_ROOT
        / "src-tauri"
        / "target"
        / "release"
        / "resources"
        / "python_env"
        / "Lib"
        / "site-packages"
        / "harmonypy",
    ]
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        if not list(candidate.glob("_harmony_cpp*.pyd")):
            continue
        if not any(
            dll.name.lower() in {"libopenblas.dll", "openblas.dll"}
            for dll in candidate.iterdir()
        ):
            continue
        return candidate
    return None


def find_windows_annoy_wheel() -> Path:
    """Find the validated Annoy wheel for the pinned CPython x64 runtime."""
    return find_windows_native_wheel(
        distribution="annoy",
        version="1.17.3",
        search_roots=WINDOWS_ANNOY_WHEEL_SEARCH_ROOTS,
        override_name="DNBCSCOPE_WINDOWS_ANNOY_WHEEL",
        display_name="Annoy",
    )


def find_windows_native_wheel(
    *,
    distribution: str,
    version: str,
    search_roots: tuple[str, ...],
    override_name: str,
    display_name: str,
) -> Path:
    candidates: list[Path] = []
    override = os.environ.get(override_name)
    if override:
        candidates.append(Path(override).expanduser())
    for root_value in search_roots:
        root = Path(root_value)
        if not root.is_absolute() and not root.drive:
            root = PROJECT_ROOT / root
        if root.is_dir():
            candidates.extend(sorted(root.rglob(f"{distribution}-{version}-*.whl")))

    python_tag = f"cp{runtime_python_major_minor().replace('.', '')}"
    valid_name = re.compile(
        rf"^{re.escape(distribution)}-{re.escape(version)}(?:-[^-]+)?-"
        rf"{python_tag}-(?:{python_tag}|abi3)-win_amd64\.whl$",
        re.IGNORECASE,
    )
    valid = list(
        dict.fromkeys(
            candidate.resolve()
            for candidate in candidates
            if candidate.is_file() and valid_name.fullmatch(candidate.name)
        )
    )
    if len(valid) == 1:
        return valid[0]
    if len(valid) > 1:
        by_digest: dict[str, list[Path]] = {}
        for wheel in valid:
            by_digest.setdefault(file_sha256(wheel), []).append(wheel)
        if len(by_digest) == 1:
            # A build drive and the project wheelhouse often contain the same
            # validated artifact. Prefer search-root order in that case.
            return valid[0]
        raise RuntimeError(
            f"找到多个内容不同的 Windows {display_name} wheel，请用 "
            f"{override_name} 明确指定一个：\n  "
            + "\n  ".join(str(path) for path in valid)
        )
    searched = "\n  ".join(search_roots)
    raise RuntimeError(
        f"未找到预构建的 Windows {display_name} wheel；不会回退到源码编译。\n"
        f"需要 {distribution}-{version}-{python_tag}-{python_tag}-win_amd64.whl，"
        f"可设置 {override_name}，或放到以下位置：\n  {searched}"
    )


def find_windows_harmonypy_wheel() -> Path:
    """Find the user's prebuilt wheel for the pinned CPython x64 runtime.

    Source builds are intentionally forbidden here: the custom wheel carries
    the native Harmony extension and runtime DLLs already validated on the
    Windows build machine.
    """
    return find_windows_native_wheel(
        distribution="harmonypy",
        version="2.0.0",
        search_roots=WINDOWS_HARMONYPY_WHEEL_SEARCH_ROOTS,
        override_name="DNBCSCOPE_WINDOWS_HARMONYPY_WHEEL",
        display_name="Harmony",
    )


def repair_harmonypy_dlls(env_dir: Path) -> bool:
    """Copy native DLLs missing from harmonypy 2.x Windows wheels.

    harmonypy 2.0.0 may build its ``_harmony_cpp`` extension successfully but
    leave Armadillo/BLAS runtime DLLs outside the wheel. The lookup is
    deliberately limited to the package, the usual temporary build folders,
    and an optional LOCALAPPDATA cache so packaging does not scan all user
    files or silently copy unrelated DLLs.
    """
    if platform.system() != "Windows":
        return False

    hpy_dir = env_dir / "Lib" / "site-packages" / "harmonypy"
    if not hpy_dir.is_dir() or not list(hpy_dir.glob("_harmony_cpp*.pyd")):
        return False

    dll_names = {
        "libopenblas.dll",
        "liblapack.dll",
        "lapack.dll",
        "openblas.dll",
        "libgfortran-5.dll",
        "libgcc_s_seh-1.dll",
        "libquadmath-0.dll",
        "libwinpthread-1.dll",
    }
    candidate_dirs: list[Path] = [hpy_dir]
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        candidate_dirs.extend(
            [
                Path(local_app_data) / "Temp",
                Path(local_app_data) / "pip" / "Cache",
            ]
        )
    candidate_dirs.extend(
        [
            Path.home() / "AppData" / "Local" / "Temp",
            Path("G:/harmonypy_build_tmp/_deps/armadillo-src/examples/lib_win64"),
        ]
    )

    found: dict[str, Path] = {}
    for directory in candidate_dirs:
        if not directory.is_dir():
            continue
        try:
            for candidate in directory.rglob("*.dll"):
                name = candidate.name.lower()
                if name in dll_names and name not in found:
                    found[name] = candidate
        except OSError:
            continue

    if not found:
        log("警告: 未找到 harmonypy 的 BLAS/LAPACK DLL；请确认 Windows 环境中已成功编译 harmonypy。")
        return False

    copied = 0
    for name, source in found.items():
        destination = hpy_dir / source.name
        if destination.is_file():
            continue
        shutil.copy2(source, destination)
        copied += 1
    if copied:
        log(f"已补齐 harmonypy 运行库: {sorted(found)}")
    return True


def verify(env_dir: Path) -> bool:
    py = python_executable(env_dir)
    verify_script = (
        "import platform\n"
        "from importlib.metadata import version\n"
        "from packaging.version import Version\n"
        "import scanpy, numpy, leidenalg, harmonypy, scrublet, bbknn, anndata, pydeseq2, annoy\n"
        "from annoy import AnnoyIndex\n"
        f"expected_python = {PYTHON_VERSION!r}\n"
        f"expected_pip = {PIP_VERSION!r}\n"
        f"expected_setuptools = {SETUPTOOLS_VERSION!r}\n"
        "if platform.python_version() != expected_python:\n"
        "    raise RuntimeError(f'Python {platform.python_version()} does not match {expected_python}')\n"
        "if version('pip') != expected_pip:\n"
        "    raise RuntimeError(f\"pip {version('pip')} does not match {expected_pip}\")\n"
        "if version('setuptools') != expected_setuptools:\n"
        "    raise RuntimeError(f\"setuptools {version('setuptools')} does not match {expected_setuptools}\")\n"
        "if version('annoy') != '1.17.3':\n"
        "    raise RuntimeError(f\"annoy {version('annoy')} does not match 1.17.3\")\n"
        "annoy_index = AnnoyIndex(2, 'euclidean')\n"
        "annoy_index.add_item(0, [0.0, 0.0])\n"
        "annoy_index.add_item(1, [1.0, 1.0])\n"
        "annoy_index.build(2)\n"
        "if annoy_index.get_nns_by_item(0, 1) != [0]:\n"
        "    raise RuntimeError('annoy native index smoke test returned an invalid result')\n"
        "pydeseq2_version = Version(version('pydeseq2'))\n"
        "if pydeseq2_version < Version('0.5.4'):\n"
        "    raise RuntimeError(f'PyDESeq2 {pydeseq2_version} is too old; expected >=0.5.4')\n"
        "print(f'scanpy={scanpy.__version__}, numpy={numpy.__version__}, "
        "leidenalg={leidenalg.__version__}, anndata={anndata.__version__}, "
        "pydeseq2={pydeseq2_version}')"
    )
    result = subprocess.run(
        # Do not let repository-root folders shadow optional Python packages
        # while validating the relocatable runtime.
        [str(py), "-P", "-c", verify_script],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        log(f"验证失败:\n{result.stderr}")
        return False
    log(f"验证通过: {result.stdout.strip()}")
    return True


def prune_environment(env_dir: Path) -> None:
    """Remove only unambiguous build caches after dependency installation.

    Release pruning is handled by ``prune_python_env.py``. In particular, do
    not recursively delete ``*.txt``/``*.md`` here: Python distributions use
    those extensions for LICENSE and NOTICE files that must ship with the app.
    """
    log("清理缓存文件以减小体积...")

    # Remove __pycache__
    for cache_dir in env_dir.rglob("__pycache__"):
        shutil.rmtree(cache_dir, ignore_errors=True)

    # Remove .pyc files (keep .py sources)
    for pyc in env_dir.rglob("*.pyc"):
        pyc.unlink(missing_ok=True)

    # Remove test directories
    for test_dir in env_dir.rglob("tests"):
        if test_dir.is_dir():
            shutil.rmtree(test_dir, ignore_errors=True)

    # Remove pip cache
    cache_dir = env_dir / ".cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)

    log("清理完成。")


def env_size(env_dir: Path) -> str:
    total = 0
    for dirpath, _, files in os.walk(env_dir):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    mb = total / (1024 * 1024)
    return f"{mb / 1024:.2f} GB" if mb > 1024 else f"{mb:.0f} MB"


def runtime_manifest(env_dir: Path) -> Path:
    return env_dir / RUNTIME_MANIFEST


def lock_digest() -> str:
    return hashlib.sha256(ANALYSIS_LOCK.read_bytes()).hexdigest()


def runtime_python_version() -> str:
    if platform.system() in ("Windows", "Darwin"):
        return PYTHON_VERSION
    return platform.python_version()


def runtime_python_major_minor() -> str:
    return ".".join(runtime_python_version().split(".")[:2])


def runtime_matches_lock(env_dir: Path) -> bool:
    """Cheap, no-import check used by build scripts before reusing a venv."""
    executable = python_executable(env_dir)
    manifest_path = runtime_manifest(env_dir)
    if not executable.is_file() or not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    runtime_kind = manifest.get("runtimeKind")
    if platform.system() == "Windows":
        if runtime_kind != "windows-nuget":
            return False
        if (
            manifest.get("windowsRuntimeKind") != WINDOWS_RUNTIME_KIND
            or WINDOWS_RUNTIME_KIND != "nuget"
            or manifest.get("windowsAsset") != WINDOWS_ASSET
            or manifest.get("windowsSha256") != WINDOWS_ASSET_SHA256
        ):
            return False
        try:
            annoy_wheel = find_windows_annoy_wheel()
            harmony_wheel = find_windows_harmonypy_wheel()
        except RuntimeError:
            return False
        if (
            manifest.get("annoyWheel") != annoy_wheel.name
            or manifest.get("annoyWheelSha256") != file_sha256(annoy_wheel)
            or manifest.get("harmonypyWheel") != harmony_wheel.name
            or manifest.get("harmonypyWheelSha256") != file_sha256(harmony_wheel)
        ):
            return False
    elif platform.system() == "Darwin":
        if runtime_kind != "python-build-standalone":
            return False
        asset_name, asset_sha256 = macos_standalone_asset()
        if (
            manifest.get("standaloneRelease") != STANDALONE_RELEASE
            or manifest.get("standaloneAsset") != asset_name
            or manifest.get("standaloneSha256") != asset_sha256
        ):
            return False
    elif runtime_kind not in (None, "venv"):
        return False
    if platform.system() == "Windows":
        compact = runtime_python_major_minor().replace(".", "")
        if not (env_dir / f"python{compact}.dll").is_file():
            return False
    return (
        manifest.get("analysisLockSha256") == lock_digest()
        and manifest.get("platformSystem") == platform.system()
        and manifest.get("pythonMajorMinor") == runtime_python_major_minor()
        and manifest.get("pythonVersion") == runtime_python_version()
        and manifest.get("pipVersion") == PIP_VERSION
        and manifest.get("setuptoolsVersion") == SETUPTOOLS_VERSION
    )


def write_runtime_manifest(env_dir: Path) -> None:
    runtime_kind = "venv"
    standalone_metadata: dict[str, str] = {}
    if platform.system() == "Windows":
        runtime_kind = "windows-nuget"
        annoy_wheel = find_windows_annoy_wheel()
        harmony_wheel = find_windows_harmonypy_wheel()
        standalone_metadata = {
            "windowsRuntimeKind": WINDOWS_RUNTIME_KIND,
            "windowsAsset": WINDOWS_ASSET,
            "windowsSha256": WINDOWS_ASSET_SHA256,
            "annoyWheel": annoy_wheel.name,
            "annoyWheelSha256": file_sha256(annoy_wheel),
            "harmonypyWheel": harmony_wheel.name,
            "harmonypyWheelSha256": file_sha256(harmony_wheel),
        }
    elif platform.system() == "Darwin":
        runtime_kind = "python-build-standalone"
        asset_name, asset_sha256 = macos_standalone_asset()
        standalone_metadata = {
            "standaloneRelease": STANDALONE_RELEASE,
            "standaloneAsset": asset_name,
            "standaloneSha256": asset_sha256,
        }
    runtime_manifest(env_dir).write_text(
        json.dumps(
            {
                "analysisLockSha256": lock_digest(),
                "platformSystem": platform.system(),
                "pythonMajorMinor": runtime_python_major_minor(),
                "pythonVersion": runtime_python_version(),
                "pipVersion": PIP_VERSION,
                "setuptoolsVersion": SETUPTOOLS_VERSION,
                "runtimeKind": runtime_kind,
                **standalone_metadata,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build or verify DNBCScope's bundled Python runtime")
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 0 only when the existing runtime matches this platform and analysis lock",
    )
    args = parser.parse_args()
    resources_dir = PROJECT_ROOT / "src-tauri" / "resources"
    env_dir = resources_dir / "python_env"

    if args.check:
        current = runtime_matches_lock(env_dir)
        log("Bundled Python runtime is current." if current else "Bundled Python runtime is missing or stale.")
        return 0 if current else 1

    log("=" * 60)
    log("  DNBCScope Python 环境打包工具")
    log("=" * 60)
    log(f"目标平台  : {platform.system()} {platform.machine()}")
    log(f"基础 Python: {sys.executable} ({platform.python_version()})")
    log(f"输出目录   : {env_dir}")
    log("")

    resources_dir.mkdir(parents=True, exist_ok=True)

    create_venv(env_dir, sys.executable)
    install_packages(env_dir)

    if platform.system() == "Windows":
        repair_harmonypy_dlls(env_dir)

    if not verify(env_dir):
        log("环境验证失败，请检查上面的错误信息。")
        return 1

    prune_environment(env_dir)
    write_runtime_manifest(env_dir)

    log("")
    log("=" * 60)
    log("  Python 环境打包完成!")
    log("=" * 60)
    log(f"  位置: {env_dir}")
    log(f"  体积: {env_size(env_dir)}")
    log(f"  Python 可执行: {python_executable(env_dir)}")
    log("")
    log("接下来运行 npm run tauri build 即可将此环境打包进应用。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
