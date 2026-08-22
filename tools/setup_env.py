#!/usr/bin/env python3
"""
DNBCScope — Python analysis environment setup.

Creates a self-contained virtual-env with scanpy + clustering deps.
Designed to be invoked from the Tauri app OR run manually:

    # From the project root:
    python tools/setup_env.py                   # creates .dnbc-env/ next to this script
    python tools/setup_env.py /custom/path      # creates env at /custom/path

After setup the Tauri app will detect and use the bundled env automatically.
"""

import argparse
import json
import os
import platform
import subprocess
import time
import venv

DEFAULT_ENV_NAME = ".dnbc-env"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYSIS_LOCK = os.path.join(PROJECT_ROOT, "requirements", "analysis-lock.txt")
PYTHON_VERSION_FILE = os.path.join(PROJECT_ROOT, ".python-version")
RUNTIME_CONFIG_FILE = os.path.join(PROJECT_ROOT, "requirements", "python-runtime.json")
with open(PYTHON_VERSION_FILE, encoding="utf-8") as version_file:
    REQUIRED_PYTHON_VERSION = version_file.read().strip()
with open(RUNTIME_CONFIG_FILE, encoding="utf-8") as config_file:
    runtime_config = json.load(config_file)
    REQUIRED_PIP_VERSION = str(runtime_config["pip"]["version"])
    REQUIRED_SETUPTOOLS_VERSION = str(runtime_config["buildTools"]["setuptools"])
    DEFAULT_PACKAGE_INDEX = str(runtime_config["packageIndexes"]["primary"])
    DEFAULT_PACKAGE_FALLBACK_INDEX = str(runtime_config["packageIndexes"]["fallback"])

# Scanpy uses Leiden by default. Louvain is retained as an optional fallback,
# but it may require a slow source build on newer Python versions.
OPTIONAL_PACKAGES = ["louvain==0.8.2"]


def log(msg: str) -> None:
    print(msg, flush=True)


def get_env_dir(custom: str | None = None) -> str:
    if custom:
        return custom
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, DEFAULT_ENV_NAME)


def python_bin(env_dir: str) -> str:
    if platform.system() == "Windows":
        return os.path.join(env_dir, "Scripts", "python.exe")
    return os.path.join(env_dir, "bin", "python")


def pip_bin(env_dir: str) -> str:
    if platform.system() == "Windows":
        return os.path.join(env_dir, "Scripts", "pip.exe")
    return os.path.join(env_dir, "bin", "pip")


def tool_env(env_dir: str) -> dict[str, str]:
    env = os.environ.copy()
    cache_dir = os.path.join(env_dir, ".cache")
    env["PIP_CACHE_DIR"] = os.path.join(cache_dir, "pip")
    env["MPLCONFIGDIR"] = os.path.join(cache_dir, "matplotlib")
    os.makedirs(env["PIP_CACHE_DIR"], exist_ok=True)
    os.makedirs(env["MPLCONFIGDIR"], exist_ok=True)
    return env


def run_pip(
    pip: str,
    args: list[str],
    label: str,
    env_dir: str,
    required: bool = True,
) -> bool:
    log(f"\n[{label}]")
    log(f"  $ {pip} {' '.join(args)}")
    started = time.monotonic()
    primary = os.environ.get("DNBCSCOPE_PYPI_INDEX", DEFAULT_PACKAGE_INDEX)
    fallback = os.environ.get(
        "DNBCSCOPE_PYPI_FALLBACK_INDEX", DEFAULT_PACKAGE_FALLBACK_INDEX
    )
    result = None
    for attempt, index in enumerate(dict.fromkeys((primary, fallback)), start=1):
        result = subprocess.run(
            [pip, "--disable-pip-version-check", "--index-url", index] + args,
            env=tool_env(env_dir),
        )
        if result.returncode == 0:
            break
        if attempt == 1 and fallback != primary:
            log(f"  Mirror failed; retrying with fallback index: {fallback}")
    assert result is not None
    elapsed = time.monotonic() - started
    if result.returncode == 0:
        log(f"  Completed in {elapsed:.0f}s")
        return True
    if required:
        raise subprocess.CalledProcessError(result.returncode, [pip] + args)
    log(f"  Optional package skipped after {elapsed:.0f}s (exit code {result.returncode}).")
    return False


def create_venv(env_dir: str) -> None:
    if os.path.exists(env_dir):
        existing_python = python_bin(env_dir)
        result = subprocess.run(
            [existing_python, "-P", "-c", "import platform; print(platform.python_version())"],
            capture_output=True,
            text=True,
            check=False,
        ) if os.path.isfile(existing_python) else None
        existing_version = result.stdout.strip() if result and result.returncode == 0 else "unknown"
        if existing_version != REQUIRED_PYTHON_VERSION:
            raise RuntimeError(
                f"Existing environment uses Python {existing_version}; expected "
                f"{REQUIRED_PYTHON_VERSION}. Remove {env_dir} and run setup again."
            )
        log(f"Environment already exists at: {env_dir}")
        return
    log(f"Creating virtual environment at: {env_dir}")
    builder = venv.EnvBuilder(with_pip=True, upgrade_deps=True)
    builder.create(env_dir)
    log("Virtual environment created.")


def install_packages(env_dir: str, include_optional: bool = False) -> None:
    pip = pip_bin(env_dir)
    if not os.path.isfile(ANALYSIS_LOCK):
        raise FileNotFoundError(f"Analysis dependency lock is missing: {ANALYSIS_LOCK}")
    log(f"Required packages: locked by {ANALYSIS_LOCK}")
    optional_status = ", ".join(OPTIONAL_PACKAGES) if include_optional else "skipped (use --with-louvain)"
    log(f"Optional packages: {optional_status}")
    log("pip output will be shown below. Large scientific packages can take several minutes.")

    run_pip(
        pip,
        [
            "install",
            "--upgrade",
            "--progress-bar",
            "on",
            f"pip=={REQUIRED_PIP_VERSION}",
            f"setuptools=={REQUIRED_SETUPTOOLS_VERSION}",
        ],
        "1/2 Install pinned packaging tools",
        env_dir,
    )
    run_pip(
        pip,
        [
            "install",
            "--upgrade",
            "--prefer-binary",
            "--progress-bar",
            "on",
            "--requirement",
            ANALYSIS_LOCK,
        ],
        "2/2 Install required analysis packages",
        env_dir,
    )
    for package in OPTIONAL_PACKAGES if include_optional else []:
        run_pip(
            pip,
            ["install", "--upgrade", "--prefer-binary", "--progress-bar", "on", package],
            f"Optional: install {package}",
            env_dir,
            required=False,
        )
    log("\nRequired packages installed successfully.")


def verify_install(env_dir: str) -> None:
    py = python_bin(env_dir)
    verify_script = (
        "from importlib.metadata import version\n"
        "from packaging.version import Version\n"
        "import scanpy, numpy, leidenalg, harmonypy, scrublet, bbknn, pydeseq2\n"
        f"expected_pip = {REQUIRED_PIP_VERSION!r}\n"
        f"expected_setuptools = {REQUIRED_SETUPTOOLS_VERSION!r}\n"
        "if version('pip') != expected_pip:\n"
        "    raise RuntimeError(f\"pip {version('pip')} does not match {expected_pip}\")\n"
        "if version('setuptools') != expected_setuptools:\n"
        "    raise RuntimeError(f\"setuptools {version('setuptools')} does not match {expected_setuptools}\")\n"
        "pydeseq2_version = Version(version('pydeseq2'))\n"
        "if pydeseq2_version < Version('0.5.4'):\n"
        "    raise RuntimeError(f'PyDESeq2 {pydeseq2_version} is too old; expected >=0.5.4')\n"
        "print(f'scanpy={scanpy.__version__}, numpy={numpy.__version__}, "
        "leidenalg={leidenalg.__version__}, bbknn={bbknn.__version__}, "
        "pydeseq2={pydeseq2_version}')"
    )
    result = subprocess.run(
        [py, "-P", "-c", verify_script],
        capture_output=True,
        text=True,
        env=tool_env(env_dir),
    )
    if result.returncode != 0:
        log(f"WARNING: verification failed:\n{result.stderr}")
    else:
        log(f"Verified: {result.stdout.strip()}")


def estimate_size(env_dir: str) -> str:
    total = 0
    for dirpath, _dirs, files in os.walk(env_dir):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    mb = total / (1024 * 1024)
    if mb > 1024:
        return f"{mb / 1024:.1f} GB"
    return f"{mb:.0f} MB"


def main() -> None:
    parser = argparse.ArgumentParser(description="Set up the DNBCScope Python analysis environment.")
    parser.add_argument("env_dir", nargs="?", help="Custom virtual environment path")
    parser.add_argument(
        "--with-louvain",
        action="store_true",
        help="Also install optional Louvain support (may require a slow source build)",
    )
    args = parser.parse_args()
    current_python = platform.python_version()
    if current_python != REQUIRED_PYTHON_VERSION:
        raise SystemExit(
            f"DNBCScope requires Python {REQUIRED_PYTHON_VERSION} exactly; "
            f"this interpreter is {current_python}."
        )
    env_dir = get_env_dir(args.env_dir)

    log("=" * 50)
    log("  DNBCScope Python Environment Setup")
    log("=" * 50)

    create_venv(env_dir)
    install_packages(env_dir, include_optional=args.with_louvain)
    verify_install(env_dir)

    size = estimate_size(env_dir)
    log(f"\nSetup complete!")
    log(f"  Location : {env_dir}")
    log(f"  Size     : {size}")
    log(f"  Python   : {python_bin(env_dir)}")
    log(f"\nThe DNBCScope app will detect this environment automatically.")


if __name__ == "__main__":
    main()
