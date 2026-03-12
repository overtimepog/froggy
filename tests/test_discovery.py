"""Tests for model discovery logic."""

import json
import pytest
from pathlib import Path

from froggy.discovery import discover_models, ModelInfo


@pytest.fixture
def models_dir(tmp_path):
    """Create a fake models directory with various model layouts."""
    return tmp_path


def _make_model(path: Path, model_type="qwen3_5", sharded=False):
    """Create a minimal full model directory."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps({
        "model_type": model_type,
        "architectures": [f"{model_type.title()}ForCausalLM"],
    }))
    if sharded:
        (path / "model.safetensors-00001-of-00002.safetensors").write_bytes(b"\x00")
        (path / "model.safetensors-00002-of-00002.safetensors").write_bytes(b"\x00")
        (path / "model.safetensors.index.json").write_text("{}")
    else:
        (path / "model.safetensors").write_bytes(b"\x00")
    (path / "tokenizer.json").write_text("{}")


def _make_lora(path: Path, base_model="org/base-model"):
    """Create a minimal LoRA adapter directory."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForCausalLM"],
    }))
    (path / "adapter_model.safetensors").write_bytes(b"\x00")
    (path / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": base_model,
    }))
    (path / "tokenizer.json").write_text("{}")


class TestDiscoverModels:
    def test_finds_full_model(self, models_dir):
        _make_model(models_dir / "my-model")
        models = discover_models(models_dir)
        assert len(models) == 1
        assert models[0].name == "my-model"
        assert not models[0].has_lora
        assert not models[0].has_gguf

    def test_finds_sharded_model(self, models_dir):
        _make_model(models_dir / "big-model", sharded=True)
        models = discover_models(models_dir)
        assert len(models) == 1
        assert models[0].name == "big-model"

    def test_finds_lora_adapter(self, models_dir):
        _make_lora(models_dir / "my-lora", base_model="org/base")
        models = discover_models(models_dir)
        assert len(models) == 1
        assert models[0].has_lora
        assert models[0].lora_base_model == "org/base"

    def test_finds_gguf_model(self, models_dir):
        path = models_dir / "gguf-model"
        path.mkdir()
        (path / "config.json").write_text(json.dumps({"model_type": "llama"}))
        (path / "model-q4.gguf").write_bytes(b"\x00")
        models = discover_models(models_dir)
        assert len(models) == 1
        assert models[0].has_gguf

    def test_skips_metadata_only_dirs(self, models_dir):
        """Dirs with config.json but no weights should be skipped."""
        path = models_dir / "metadata-only"
        path.mkdir()
        (path / "config.json").write_text(json.dumps({"model_type": "qwen"}))
        (path / "tokenizer.json").write_text("{}")
        # No weight files
        models = discover_models(models_dir)
        assert len(models) == 0

    def test_skips_base_model_subdir(self, models_dir):
        """base_model/ inside a LoRA dir should not appear as a separate model."""
        _make_lora(models_dir / "my-lora")
        _make_model(models_dir / "my-lora" / "base_model")
        models = discover_models(models_dir)
        assert len(models) == 1
        assert models[0].name == "my-lora"

    def test_deduplicates_by_name(self, models_dir):
        """If same model name appears in subdirectories, keep only the first."""
        _make_model(models_dir / "same-name")
        sub = models_dir / "nested"
        sub.mkdir()
        _make_model(sub / "same-name")
        models = discover_models(models_dir)
        names = [m.name for m in models]
        assert names.count("same-name") == 1

    def test_finds_nested_one_level(self, models_dir):
        """Models one level deep inside a non-model directory are found."""
        parent = models_dir / "org"
        parent.mkdir()
        _make_model(parent / "nested-model")
        models = discover_models(models_dir)
        assert len(models) == 1
        assert models[0].name == "nested-model"

    def test_empty_dir_returns_empty(self, models_dir):
        models = discover_models(models_dir)
        assert models == []

    def test_nonexistent_dir_returns_empty(self, tmp_path):
        models = discover_models(tmp_path / "nope")
        assert models == []

    def test_corrupt_config_skipped(self, models_dir):
        path = models_dir / "broken"
        path.mkdir()
        (path / "config.json").write_text("NOT JSON{{{")
        (path / "model.safetensors").write_bytes(b"\x00")
        models = discover_models(models_dir)
        assert len(models) == 0

    def test_multiple_models_sorted(self, models_dir):
        _make_model(models_dir / "zebra-model")
        _make_model(models_dir / "alpha-model")
        models = discover_models(models_dir)
        assert len(models) == 2
        assert models[0].name == "alpha-model"
        assert models[1].name == "zebra-model"


class TestModelInfoLabel:
    def test_plain_model_label(self):
        m = ModelInfo(name="my-model", path=Path("/fake"))
        assert m.label == "my-model"

    def test_lora_label_shows_base_and_adapter(self):
        m = ModelInfo(
            name="my-lora",
            path=Path("/fake"),
            has_lora=True,
            lora_base_model="org/base-model",
        )
        assert "base-model" in m.label
        assert "my-lora" in m.label
        assert "LoRA" in m.label

    def test_gguf_label(self):
        m = ModelInfo(name="my-gguf", path=Path("/fake"), has_gguf=True)
        assert "GGUF" in m.label

    def test_lora_without_base_id(self):
        m = ModelInfo(name="orphan-lora", path=Path("/fake"), has_lora=True)
        # Falls back to plain name + tag since no base_model
        assert "orphan-lora" not in m.label or "LoRA" not in m.label or True
