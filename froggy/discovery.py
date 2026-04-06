"""Model discovery — scans directories for HuggingFace models."""

import json
from dataclasses import dataclass, field
from pathlib import Path

SKIP_DIRS = {"base_model", "__pycache__", ".cache", ".git", "node_modules"}


@dataclass
class ModelInfo:
    name: str
    path: Path
    model_type: str = ""
    architectures: list[str] = field(default_factory=list)
    has_lora: bool = False
    lora_base_model: str | None = None
    has_gguf: bool = False
    has_jang: bool = False
    is_ollama: bool = False

    @property
    def label(self) -> str:
        if self.is_ollama:
            return f"{self.name} [bold blue]\\[Ollama][/]"
        if self.has_lora and self.lora_base_model:
            base_short = self.lora_base_model.split("/")[-1]
            return f"{base_short} + [bold magenta]{self.name}[/] [magenta]\\[LoRA][/]"
        tags: list[str] = []
        if self.has_gguf:
            tags.append("[bold green]\\[GGUF][/]")
        if self.has_jang:
            tags.append("[bold yellow]\\[JANG/vMLX][/]")
        suffix = f" {' '.join(tags)}" if tags else ""
        return f"{self.name}{suffix}"


def discover_models(search_dir: Path) -> list[ModelInfo]:
    """Scan one level deep for directories containing model weights."""
    models: list[ModelInfo] = []
    if not search_dir.is_dir():
        return models

    for child in sorted(search_dir.iterdir()):
        if not child.is_dir() or child.name in SKIP_DIRS:
            continue
        if (child / "config.json").exists():
            _try_add_model(child, models)
        else:
            # Check one level deeper
            for sub in child.iterdir():
                if sub.is_dir() and sub.name not in SKIP_DIRS and (sub / "config.json").exists():
                    _try_add_model(sub, models)

    return models


def _try_add_model(path: Path, models: list[ModelInfo]):
    try:
        cfg = json.loads((path / "config.json").read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return

    is_lora = (path / "adapter_model.safetensors").exists() or (path / "adapter_model.bin").exists()
    has_weights = (
        is_lora
        or any(path.glob("model*.safetensors"))
        or any(path.glob("*.bin"))
        or any(path.glob("*.gguf"))
        or (path / "model.safetensors").exists()
    )
    if not has_weights:
        return

    if any(m.name == path.name for m in models):
        return

    info = ModelInfo(
        name=path.name,
        path=path,
        model_type=cfg.get("model_type", "unknown"),
        architectures=cfg.get("architectures", []),
    )

    if is_lora:
        info.has_lora = True
        adapter_cfg_path = path / "adapter_config.json"
        if adapter_cfg_path.exists():
            try:
                acfg = json.loads(adapter_cfg_path.read_text(encoding="utf-8"))
                info.lora_base_model = acfg.get("base_model_name_or_path")
            except (json.JSONDecodeError, OSError):
                pass

    info.has_gguf = any(path.glob("*.gguf"))
    info.has_jang = (path / "jang_config.json").exists()
    models.append(info)


def discover_ollama_models(
    base_url: str = "http://localhost:11434",
) -> list[ModelInfo]:
    """Query a running Ollama server for available models."""
    import urllib.request

    try:
        with urllib.request.urlopen(f"{base_url}/api/tags") as resp:
            data = json.loads(resp.read())
    except OSError:
        return []

    models: list[ModelInfo] = []
    for entry in data.get("models", []):
        name = entry.get("name", "")
        details = entry.get("details", {})
        models.append(
            ModelInfo(
                name=name,
                path=Path(base_url),  # placeholder — Ollama manages files
                model_type=details.get("family", ""),
                is_ollama=True,
            )
        )
    return models
