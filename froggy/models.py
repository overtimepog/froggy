"""Model management — list, inspect, and remove downloaded models.

Pure business-logic layer with no CLI dependencies (no Click, no Rich).
Extends ``froggy.discovery`` with GGUF-only directory scanning, directory
size computation, model deletion, and metadata aggregation.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from froggy.discovery import discover_models

# ── Helpers ──────────────────────────────────────────────────────


def _dir_size(path: Path) -> int:
    """Return total size in bytes of all files under *path* (recursive)."""
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _resolve_model_dir(name: str, models_path: Path) -> Path:
    """Resolve a model name to its directory on disk.

    Accepts both ``org/model`` (converted to ``org--model``) and the
    raw ``org--model`` directory name.  Raises ``ValueError`` if the
    directory does not exist.
    """
    # Normalise org/model → org--model
    dir_name = name.replace("/", "--")
    model_dir = models_path / dir_name
    if not model_dir.is_dir():
        raise ValueError(
            f"Model '{name}' not found in {models_path} "
            f"(looked for directory '{dir_name}')"
        )
    return model_dir


def _detect_format(path: Path) -> str:
    """Detect the weight format for a model directory."""
    if any(path.glob("*.gguf")):
        return "GGUF"
    if any(path.glob("*.safetensors")) or any(path.glob("model*.safetensors*")):
        return "SafeTensors"
    if any(path.glob("*.bin")):
        return "PyTorch"
    return "Unknown"


def _scan_gguf_only(models_path: Path) -> list[dict]:
    """Find directories containing ``.gguf`` files but **no** ``config.json``.

    These are GGUF-only downloads (e.g. from ``hf_hub_download``) that
    ``discover_models()`` cannot detect because it requires ``config.json``.
    """
    results: list[dict] = []
    if not models_path.is_dir():
        return results

    for child in sorted(models_path.iterdir()):
        if not child.is_dir():
            continue
        has_config = (child / "config.json").exists()
        has_gguf = any(child.glob("*.gguf"))
        if has_gguf and not has_config:
            results.append(
                {
                    "name": child.name,
                    "path": str(child),
                    "format": "GGUF",
                    "size": _dir_size(child),
                    "model_type": "unknown",
                    "architectures": [],
                    "modified": child.stat().st_mtime,
                }
            )
    return results


# ── Public API ───────────────────────────────────────────────────


def list_models(models_path: Path) -> list[dict]:
    """Return a list of all locally-available models.

    Combines results from :func:`discover_models` (standard HuggingFace
    layouts with ``config.json``) and :func:`_scan_gguf_only` (GGUF-only
    directories without ``config.json``).

    Each entry is a dict with keys:
    ``name``, ``path``, ``format``, ``size``, ``model_type``,
    ``architectures``, ``modified``.
    """
    # Standard models via discovery.py
    standard = discover_models(models_path)
    seen_names: set[str] = set()
    results: list[dict] = []

    for m in standard:
        seen_names.add(m.name)
        results.append(
            {
                "name": m.name,
                "path": str(m.path),
                "format": _detect_format(m.path),
                "size": _dir_size(m.path),
                "model_type": m.model_type,
                "architectures": list(m.architectures),
                "modified": m.path.stat().st_mtime,
            }
        )

    # GGUF-only models that discover_models misses
    for entry in _scan_gguf_only(models_path):
        if entry["name"] not in seen_names:
            results.append(entry)

    return results


def remove_model(name: str, models_path: Path) -> int:
    """Delete a model directory and return the number of bytes freed.

    Raises ``ValueError`` if the model directory does not exist.
    """
    model_dir = _resolve_model_dir(name, models_path)
    size = _dir_size(model_dir)
    shutil.rmtree(model_dir)
    return size


def model_info(name: str, models_path: Path) -> dict:
    """Return a detailed metadata dict for a single model.

    Raises ``ValueError`` if the model directory does not exist.

    Returned keys: ``name``, ``path``, ``format``, ``size``,
    ``model_type``, ``architectures``, ``file_count``, ``has_gguf``,
    ``has_lora``, ``modified``.
    """
    model_dir = _resolve_model_dir(name, models_path)

    has_gguf = any(model_dir.glob("*.gguf"))
    has_lora = (model_dir / "adapter_model.safetensors").exists() or (
        model_dir / "adapter_model.bin"
    ).exists()

    # Read config.json if available
    config_path = model_dir / "config.json"
    model_type = "unknown"
    architectures: list[str] = []
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            model_type = cfg.get("model_type", "unknown")
            architectures = cfg.get("architectures", [])
        except (json.JSONDecodeError, OSError):
            pass

    all_files = [f for f in model_dir.rglob("*") if f.is_file()]

    return {
        "name": model_dir.name,
        "path": str(model_dir),
        "format": _detect_format(model_dir),
        "size": sum(f.stat().st_size for f in all_files),
        "model_type": model_type,
        "architectures": architectures,
        "file_count": len(all_files),
        "has_gguf": has_gguf,
        "has_lora": has_lora,
        "modified": model_dir.stat().st_mtime,
    }
