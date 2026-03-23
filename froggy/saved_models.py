"""Saved model management — parse, normalize, store, and discover variants.

Users can add models from any source format:
- OpenRouter model ID: ``openai/gpt-4o``, ``anthropic/claude-3-opus``
- OpenRouter URL: ``https://openrouter.ai/models/openai/gpt-4o``
- HuggingFace repo ID: ``mlx-community/Llama-3-8B-4bit``
- HuggingFace URL: ``https://huggingface.co/TheBloke/Mistral-7B-GGUF``
- Ollama model name: ``ollama:llama3``, ``ollama:codellama``
- Bare name: ``llama3`` → searches across all platforms

Saved models are stored in ``~/.froggy/config.yaml`` under the ``models`` key.
"""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from .config import get_config, load_config, save_config
from .discovery import ModelInfo


def parse_model_source(source: str) -> dict:
    """Parse a model source string into a normalized model record.

    Returns a dict with keys:
        - name: the model ID (e.g. ``openai/gpt-4o``)
        - source: 'openrouter', 'huggingface', or 'ollama'
        - original: the raw input string

    Raises ValueError if the source can't be parsed.
    """
    source = source.strip()
    if not source:
        raise ValueError("Empty model source.")

    # Ollama explicit prefix: ollama:llama3
    if source.startswith("ollama:"):
        name = source[7:].strip()
        if not name:
            raise ValueError("Empty Ollama model name.")
        return {"name": name, "source": "ollama", "original": source}

    # OpenRouter URL: https://openrouter.ai/models/openai/gpt-4o
    if "openrouter.ai" in source:
        parsed = urlparse(source)
        path = parsed.path.strip("/")
        path = re.sub(r"^models/", "", path)
        if not path or "/" not in path:
            raise ValueError(f"Can't extract model ID from OpenRouter URL: {source}")
        return {"name": path, "source": "openrouter", "original": source}

    # HuggingFace URL: https://huggingface.co/TheBloke/Mistral-7B-GGUF
    if "huggingface.co" in source:
        parsed = urlparse(source)
        path = parsed.path.strip("/")
        path = re.sub(r"/blob/main/.*$", "", path)
        path = re.sub(r"/tree/.*$", "", path)
        if not path or "/" not in path:
            raise ValueError(f"Can't extract repo ID from HuggingFace URL: {source}")
        return {"name": path, "source": "huggingface", "original": source}

    # org/model format — determine platform by heuristics
    if "/" in source:
        hf_patterns = ["-GGUF", "-GPTQ", "-AWQ", "mlx-community/", "-4bit", "-8bit"]
        if any(p.lower() in source.lower() for p in hf_patterns):
            return {"name": source, "source": "huggingface", "original": source}
        return {"name": source, "source": "openrouter", "original": source}

    # Bare name without slash — could be anything
    return {"name": source, "source": "auto", "original": source}


# ---------------------------------------------------------------------------
# Variant discovery — find a model across platforms
# ---------------------------------------------------------------------------


def discover_variants(query: str) -> list[dict]:
    """Search for a model across OpenRouter, HuggingFace, and Ollama.

    Returns a list of variant dicts:
        {name, source, context_length, description}
    sorted by relevance.
    """
    variants: list[dict] = []
    query_lower = query.lower().replace("-", "").replace("_", "")

    # 1. Search OpenRouter
    try:
        variants.extend(_search_openrouter(query, query_lower))
    except Exception:
        pass

    # 2. Search HuggingFace
    try:
        variants.extend(_search_huggingface(query, query_lower))
    except Exception:
        pass

    # 3. Search Ollama
    try:
        variants.extend(_search_ollama(query, query_lower))
    except Exception:
        pass

    return variants


def _search_openrouter(query: str, query_lower: str) -> list[dict]:
    """Search OpenRouter's model catalog."""
    key = get_config("openrouter_api_key", "")
    if not key:
        import os
        key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        return []

    try:
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/models",
            headers={
                "Authorization": f"Bearer {key}",
                "HTTP-Referer": "https://github.com/overtimepog/froggy",
            },
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
    except OSError:
        return []

    results = []
    for entry in data.get("data", []):
        model_id = entry.get("id", "")
        if not model_id:
            continue
        # Match by substring in model ID
        id_normalized = model_id.lower().replace("-", "").replace("_", "")
        if query_lower in id_normalized or query.lower() in model_id.lower():
            ctx = entry.get("context_length")
            pricing = entry.get("pricing", {})
            prompt_price = pricing.get("prompt", "?")
            results.append({
                "name": model_id,
                "source": "openrouter",
                "context_length": ctx,
                "description": f"ctx:{ctx:,}" if ctx else "",
                "pricing": f"${float(prompt_price)*1_000_000:.2f}/M" if prompt_price and prompt_price != "?" else "",
            })
    return results


def _search_huggingface(query: str, query_lower: str) -> list[dict]:
    """Search HuggingFace Hub for matching repos."""
    try:
        req = urllib.request.Request(
            f"https://huggingface.co/api/models?search={query}&limit=10&sort=downloads&direction=-1",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
    except OSError:
        return []

    results = []
    for entry in data if isinstance(data, list) else []:
        model_id = entry.get("modelId", entry.get("id", ""))
        if not model_id:
            continue
        pipeline = entry.get("pipeline_tag", "")
        downloads = entry.get("downloads", 0)
        tags = entry.get("tags", [])
        fmt = ""
        if "gguf" in tags:
            fmt = "GGUF"
        elif any("mlx" in t for t in tags):
            fmt = "MLX"
        elif "safetensors" in tags:
            fmt = "SafeTensors"
        desc_parts = []
        if fmt:
            desc_parts.append(fmt)
        if pipeline:
            desc_parts.append(pipeline)
        if downloads:
            desc_parts.append(f"{downloads:,} downloads")
        results.append({
            "name": model_id,
            "source": "huggingface",
            "context_length": None,
            "description": " · ".join(desc_parts),
            "pricing": "free (local)",
        })
    return results


def _search_ollama(query: str, query_lower: str) -> list[dict]:
    """Check if a running Ollama server has matching models."""
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3) as resp:
            data = json.loads(resp.read())
    except OSError:
        return []

    results = []
    for entry in data.get("models", []):
        name = entry.get("name", "")
        if query.lower() in name.lower():
            details = entry.get("details", {})
            size = entry.get("size", 0)
            desc_parts = []
            if details.get("family"):
                desc_parts.append(details["family"])
            if size:
                desc_parts.append(f"{size / 1e9:.1f}GB")
            results.append({
                "name": name,
                "source": "ollama",
                "context_length": None,
                "description": " · ".join(desc_parts),
                "pricing": "free (local)",
            })
    return results


# ---------------------------------------------------------------------------
# Config persistence
# ---------------------------------------------------------------------------


def get_saved_models() -> list[dict]:
    """Return the list of saved models from config."""
    cfg = load_config()
    models = cfg.get("models", [])
    if not isinstance(models, list):
        return []
    return models


def add_saved_model(record: dict) -> bool:
    """Add a model record to the saved list. Returns False if already exists."""
    cfg = load_config()
    models = cfg.get("models", [])
    if not isinstance(models, list):
        models = []

    for m in models:
        if m.get("name") == record["name"] and m.get("source") == record["source"]:
            return False

    models.append(record)
    cfg["models"] = models
    save_config(cfg)
    return True


def remove_saved_model(name: str) -> bool:
    """Remove a model by name (or partial match). Returns False if not found."""
    cfg = load_config()
    models = cfg.get("models", [])
    if not isinstance(models, list):
        return False

    name_lower = name.lower()
    new_models = [m for m in models if name_lower not in m.get("name", "").lower()]
    if len(new_models) == len(models):
        return False

    cfg["models"] = new_models
    save_config(cfg)
    return True


def saved_models_as_model_info() -> list[ModelInfo]:
    """Convert saved models into ModelInfo objects for the chat selector."""
    saved = get_saved_models()
    infos = []
    for m in saved:
        source = m.get("source", "openrouter")
        name = m.get("name", "")
        ctx = m.get("context_length")
        if source == "openrouter":
            infos.append(ModelInfo(
                name=name,
                path=Path("https://openrouter.ai"),
                is_openrouter=True,
                context_length=ctx,
            ))
        elif source == "ollama":
            infos.append(ModelInfo(
                name=name,
                path=Path("http://localhost:11434"),
                is_ollama=True,
            ))
        elif source == "huggingface":
            # HuggingFace models need to be downloaded first
            # Check if already downloaded
            from .paths import models_dir as get_models_dir
            local_path = get_models_dir() / name.split("/")[-1]
            if local_path.is_dir():
                from .discovery import _try_add_model
                temp: list[ModelInfo] = []
                _try_add_model(local_path, temp)
                if temp:
                    infos.extend(temp)
                    continue
            # Not downloaded — show as placeholder that triggers download
            infos.append(ModelInfo(
                name=name,
                path=Path(name),
                model_type="huggingface (not downloaded)",
            ))
    return infos
