"""llmfit binary management and system info parsing.

Auto-downloads the llmfit binary from GitHub Releases, manages staleness
(24h re-check), provides PATH fallback, and parses ``llmfit --json system``
output for hardware-matched model recommendations.

Only uses stdlib + froggy.paths — no heavy third-party deps.
"""

from __future__ import annotations

import io
import json
import logging
import os
import platform
import shutil
import subprocess
import tarfile
import time
import urllib.request
from pathlib import Path

from froggy.paths import froggy_home

logger = logging.getLogger(__name__)

_LLMFIT_RELEASES_URL = (
    "https://api.github.com/repos/AlexsJones/llmfit/releases/latest"
)
_STALE_SECONDS = 24 * 60 * 60  # 24 hours


# ---------------------------------------------------------------------------
# Binary path
# ---------------------------------------------------------------------------

def llmfit_binary_path() -> Path:
    """Return the managed path for the llmfit binary."""
    return froggy_home() / "bin" / "llmfit"


# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------

def _platform_asset_name() -> str | None:
    """Map the current platform to a llmfit GitHub release asset name.

    Returns ``None`` for unsupported platforms.
    """
    system = platform.system()
    machine = platform.machine()

    if system == "Darwin":
        if machine == "arm64":
            return "aarch64-apple-darwin"
        if machine == "x86_64":
            return "x86_64-apple-darwin"
    elif system == "Linux":
        if machine == "x86_64":
            return "x86_64-unknown-linux-gnu"

    logger.debug("Unsupported platform for llmfit: %s/%s", system, machine)
    return None


# ---------------------------------------------------------------------------
# Staleness check
# ---------------------------------------------------------------------------

def _is_stale(path: Path) -> bool:
    """Return ``True`` if *path* does not exist or is older than 24 hours."""
    if not path.exists():
        return True
    age = time.time() - path.stat().st_mtime
    return age > _STALE_SECONDS


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _download_llmfit(dest: Path) -> bool:
    """Download the latest llmfit release binary to *dest*.

    Fetches the GitHub Releases API, finds the matching platform asset,
    downloads the tarball, extracts it, and moves the ``llmfit`` executable
    to *dest*.  Returns ``True`` on success, ``False`` on any error.
    """
    asset_name = _platform_asset_name()
    if asset_name is None:
        logger.debug("Cannot download llmfit: unsupported platform")
        return False

    try:
        # Fetch release metadata
        req = urllib.request.Request(
            _LLMFIT_RELEASES_URL,
            headers={"Accept": "application/vnd.github+json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            release = json.loads(resp.read().decode())

        # Find matching asset
        download_url: str | None = None
        for asset in release.get("assets", []):
            if asset_name in asset.get("name", ""):
                download_url = asset["browser_download_url"]
                break

        if download_url is None:
            logger.debug(
                "No matching llmfit asset for %s in release", asset_name
            )
            return False

        logger.debug("Downloading llmfit from %s", download_url)

        # Download tarball
        with urllib.request.urlopen(download_url, timeout=60) as resp:
            tarball_bytes = resp.read()

        # Extract and find the llmfit executable
        dest.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(tarball_bytes), mode="r:*") as tf:
            llmfit_member = None
            for member in tf.getmembers():
                if member.name.endswith("/llmfit") or member.name == "llmfit":
                    llmfit_member = member
                    break

            if llmfit_member is None:
                logger.debug("llmfit executable not found in tarball")
                return False

            extracted = tf.extractfile(llmfit_member)
            if extracted is None:
                logger.debug("Could not extract llmfit member from tarball")
                return False

            dest.write_bytes(extracted.read())

        os.chmod(dest, 0o755)
        logger.debug("llmfit installed to %s", dest)
        return True

    except Exception:
        logger.debug("Failed to download llmfit", exc_info=True)
        return False


# ---------------------------------------------------------------------------
# Ensure binary is available
# ---------------------------------------------------------------------------

def ensure_llmfit() -> Path | None:
    """Return a path to a usable llmfit binary, or ``None``.

    Checks the managed binary first (downloading or refreshing if needed),
    then falls back to ``llmfit`` on PATH.
    """
    path = llmfit_binary_path()

    if path.exists() and not _is_stale(path):
        logger.debug("Using cached llmfit at %s", path)
        return path

    if _download_llmfit(path):
        return path

    # Fallback: check PATH
    which = shutil.which("llmfit")
    if which is not None:
        logger.debug("Falling back to llmfit on PATH: %s", which)
        return Path(which)

    logger.debug("llmfit not available")
    return None


# ---------------------------------------------------------------------------
# System info
# ---------------------------------------------------------------------------

def llmfit_system_info(binary_path: Path | str) -> dict | None:
    """Run ``llmfit --json system`` and return the parsed system dict.

    Returns ``None`` on any error (subprocess failure, bad JSON, missing keys).
    """
    try:
        result = subprocess.run(
            [str(binary_path), "--json", "system"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            logger.debug(
                "llmfit exited with code %d: %s",
                result.returncode,
                result.stderr,
            )
            return None

        data = json.loads(result.stdout)
        system = data.get("system")
        if system is None:
            logger.debug("llmfit output missing 'system' key")
            return None

        return system

    except subprocess.TimeoutExpired:
        logger.debug("llmfit timed out after 30s")
        return None
    except (json.JSONDecodeError, OSError) as exc:
        logger.debug("llmfit system info error: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Recommend
# ---------------------------------------------------------------------------

def llmfit_recommend(
    binary_path: Path | str,
    limit: int = 5,
    use_case: str | None = None,
) -> list[dict]:
    """Run ``llmfit recommend --json`` and return a list of model dicts.

    Each dict has keys: ``name``, ``score``, ``best_quant``,
    ``estimated_tps``, ``fit_level``, ``run_mode``, ``memory_required_gb``.
    Returns ``[]`` on any error.
    """
    cmd = [str(binary_path), "recommend", "--json", "--limit", str(limit)]
    if use_case:
        cmd.extend(["--use-case", use_case])

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            logger.debug(
                "llmfit recommend exited with code %d: %s",
                result.returncode,
                result.stderr,
            )
            return []

        data = json.loads(result.stdout)
        raw_models = data.get("models", [])

        models: list[dict] = []
        for m in raw_models:
            models.append({
                "name": m.get("name", "unknown"),
                "score": m.get("score", 0),
                "best_quant": m.get("best_quant", "unknown"),
                "estimated_tps": m.get("estimated_tps", 0.0),
                "fit_level": m.get("fit_level", "unknown"),
                "run_mode": m.get("run_mode", "unknown"),
                "memory_required_gb": m.get("memory_required_gb", 0.0),
            })
        return models

    except subprocess.TimeoutExpired:
        logger.debug("llmfit recommend timed out after 30s")
        return []
    except (json.JSONDecodeError, OSError) as exc:
        logger.debug("llmfit recommend error: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Memory budget
# ---------------------------------------------------------------------------

def get_memory_budget() -> int | None:
    """Return a hardware-informed memory budget in bytes, or ``None``.

    Uses llmfit system info when available.  On Apple Silicon (unified
    memory) uses ``total_ram_gb``; on discrete GPU uses ``gpu_vram_gb``;
    on CPU-only uses ``total_ram_gb``.  The llmfit-reported values are
    used as-is (llmfit already accounts for overhead).

    Falls back to ``get_system_memory() * 0.7`` when llmfit is
    unavailable.  Returns ``None`` if both fail.
    """
    binary = ensure_llmfit()
    if binary is not None:
        info = llmfit_system_info(binary)
        if info is not None:
            if info.get("unified_memory"):
                gb = info.get("total_ram_gb") or 0
            elif (info.get("gpu_vram_gb") or 0) > 0:
                gb = info["gpu_vram_gb"]
            else:
                gb = info.get("total_ram_gb") or 0

            if gb and gb > 0:
                return int(gb * 1024 * 1024 * 1024)

    # Fallback: heuristic from system memory
    from froggy.download import get_system_memory

    sys_mem = get_system_memory()
    if sys_mem is not None:
        return int(sys_mem * 0.7)

    return None
