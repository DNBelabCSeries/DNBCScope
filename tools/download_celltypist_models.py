#!/usr/bin/env python3
"""Read the official CellTypist catalog and download one selected model."""

import hashlib
import http.client
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

MODELS_JSON_URL = "https://celltypist.cog.sanger.ac.uk/models/models.json"
USER_AGENT = "DNBCScope/1.0"
MAX_MODEL_BYTES = 128 * 1024 * 1024
MAX_RETRIES = 5
RETRIABLE_ERRORS = (urllib.error.URLError, ConnectionError, TimeoutError, http.client.IncompleteRead)


def _retry(fn, description):
    """Run fn with exponential backoff for transient network errors."""
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn()
        except RETRIABLE_ERRORS as error:
            last_error = error
            if attempt >= MAX_RETRIES:
                break
            wait = 2 ** (attempt - 1)
            print(
                json.dumps(
                    {
                        "event": "retry",
                        "attempt": attempt,
                        "maxRetries": MAX_RETRIES,
                        "waitSeconds": wait,
                        "error": str(error),
                    }
                ),
                file=sys.stderr,
                flush=True,
            )
            time.sleep(wait)
    raise last_error


def fetch_catalog():
    def do_fetch():
        request = urllib.request.Request(MODELS_JSON_URL, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))

    return _retry(do_fetch, "fetch catalog")


def _download_once(url, partial, catalog_size, progress):
    """Single download attempt. Resumes from .part file if present."""
    resume_from = 0
    if os.path.exists(partial):
        resume_from = os.path.getsize(partial)
        if resume_from > MAX_MODEL_BYTES:
            os.remove(partial)
            resume_from = 0

    headers = {"User-Agent": USER_AGENT}
    if resume_from > 0:
        headers["Range"] = f"bytes={resume_from}-"

    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        status = getattr(response, "status", None)
        if status is None:
            getcode = getattr(response, "getcode", None)
            status = getcode() if callable(getcode) else 200
        if resume_from > 0 and status == 206:
            # Server supports range request; append to existing .part.
            mode = "ab"
            downloaded = resume_from
            content_range = response.headers.get("Content-Range", "")
            if "/" in content_range:
                try:
                    total = int(content_range.rsplit("/", 1)[1])
                except ValueError:
                    total = catalog_size
            else:
                total = catalog_size
        else:
            # Server returned 200 (no range support) or no resume requested.
            mode = "wb"
            downloaded = 0
            content_length = response.headers.get("Content-Length")
            total = int(content_length) if content_length else catalog_size

        if total > MAX_MODEL_BYTES:
            raise RuntimeError("CellTypist model exceeds the download size limit")

        if progress:
            progress(
                downloaded,
                total,
                min(99, int(downloaded * 100 / total)) if total else 0,
            )

        with open(partial, mode) as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                downloaded += len(chunk)
                if downloaded > MAX_MODEL_BYTES:
                    raise RuntimeError("CellTypist model exceeds the download size limit")
                handle.write(chunk)
                if progress:
                    percent = min(99, int(downloaded * 100 / total)) if total else 0
                    progress(downloaded, total, percent)

    if downloaded < 1024:
        raise RuntimeError("Downloaded CellTypist model is unexpectedly small")
    if total and downloaded != total:
        raise ConnectionError(
            f"Incomplete CellTypist model download: expected {total} bytes, received {downloaded}"
        )
    return downloaded, total


def download_model(output_dir: str, model_name: str, progress=None):
    catalog = fetch_catalog()
    expected_filename = f"{model_name}.pkl"
    model = next(
        (item for item in catalog.get("models", []) if item.get("filename") == expected_filename),
        None,
    )
    if model is None:
        raise RuntimeError(f"Unknown CellTypist model: {model_name}")

    url = model.get("url", "")
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "celltypist.cog.sanger.ac.uk":
        raise RuntimeError("CellTypist catalog returned an untrusted download URL")

    os.makedirs(output_dir, exist_ok=True)
    destination = os.path.join(output_dir, expected_filename)
    partial = f"{destination}.part"
    partial_meta = f"{partial}.json"
    catalog_size = int(model.get("size") or 0)
    expected_md5 = model.get("md5")
    download_identity = {
        "url": url,
        "version": model.get("version", ""),
    }

    previous_identity = None
    if os.path.exists(partial_meta):
        try:
            with open(partial_meta, encoding="utf-8") as meta_handle:
                previous_identity = json.load(meta_handle)
        except (OSError, ValueError, TypeError):
            previous_identity = None
    if os.path.exists(partial) and previous_identity != download_identity:
        os.remove(partial)
    with open(partial_meta, "w", encoding="utf-8") as meta_handle:
        json.dump(download_identity, meta_handle)

    try:
        downloaded, total = _retry(
            lambda: _download_once(url, partial, catalog_size, progress),
            "download model",
        )
        if expected_md5:
            hasher = hashlib.md5()
            with open(partial, "rb") as verify_handle:
                for verify_chunk in iter(lambda: verify_handle.read(1024 * 1024), b""):
                    hasher.update(verify_chunk)
            if hasher.hexdigest().lower() != expected_md5.lower():
                os.remove(partial)
                if os.path.exists(partial_meta):
                    os.remove(partial_meta)
                raise RuntimeError("Downloaded CellTypist model failed checksum verification")
        os.replace(partial, destination)
        if os.path.exists(partial_meta):
            os.remove(partial_meta)
        if progress:
            progress(downloaded, total or downloaded, 100)
    except Exception:
        # Keep .part file so the next download attempt can resume; only clean
        # up when the user starts a different download or the file is complete.
        raise

    return {
        "name": model_name,
        "status": "downloaded",
        "path": destination,
        "size": downloaded,
        "version": model.get("version", ""),
        "cellTypes": model.get("No_celltypes", 0),
    }


def main():
    if len(sys.argv) == 2 and sys.argv[1] == "catalog":
        print(json.dumps(fetch_catalog()))
        return
    if len(sys.argv) == 4 and sys.argv[1] == "download":
        def report_progress(downloaded, total, percent):
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "downloaded": downloaded,
                        "total": total,
                        "percent": percent,
                    }
                ),
                flush=True,
            )

        print(json.dumps(download_model(sys.argv[2], sys.argv[3], report_progress)))
        return
    raise RuntimeError("Usage: download_celltypist_models.py catalog | download OUTPUT_DIR MODEL_NAME")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
