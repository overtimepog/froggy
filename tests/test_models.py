"""Tests for model management business logic."""

import json
from pathlib import Path

import pytest

from froggy.models import list_models, model_info, remove_model

# ── Fixture helpers ──────────────────────────────────────────────


def _make_model(path: Path, model_type: str = "qwen3_5", sharded: bool = False):
    """Create a minimal full model directory (config.json + weights)."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": model_type,
                "architectures": [f"{model_type.title()}ForCausalLM"],
            }
        )
    )
    if sharded:
        (path / "model.safetensors-00001-of-00002.safetensors").write_bytes(b"\x00" * 64)
        (path / "model.safetensors-00002-of-00002.safetensors").write_bytes(b"\x00" * 64)
    else:
        (path / "model.safetensors").write_bytes(b"\x00" * 128)
    (path / "tokenizer.json").write_text("{}")


def _make_gguf_only(path: Path, size: int = 256):
    """Create a directory with only a .gguf file, no config.json."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "model-q4_0.gguf").write_bytes(b"\x00" * size)


def _make_lora(path: Path, base_model: str = "org/base-model"):
    """Create a minimal LoRA adapter directory."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "architectures": ["Qwen3_5ForCausalLM"],
            }
        )
    )
    (path / "adapter_model.safetensors").write_bytes(b"\x00" * 64)
    (path / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": base_model})
    )


# ── list_models ──────────────────────────────────────────────────


class TestListModels:
    def test_finds_standard_model(self, tmp_path):
        _make_model(tmp_path / "my-model")
        result = list_models(tmp_path)
        assert len(result) == 1
        assert result[0]["name"] == "my-model"
        assert result[0]["format"] == "SafeTensors"
        assert result[0]["size"] > 0
        assert "path" in result[0]

    def test_finds_gguf_only_model(self, tmp_path):
        _make_gguf_only(tmp_path / "gguf-model")
        result = list_models(tmp_path)
        assert len(result) == 1
        assert result[0]["name"] == "gguf-model"
        assert result[0]["format"] == "GGUF"
        assert result[0]["model_type"] == "unknown"

    def test_finds_both_standard_and_gguf_only(self, tmp_path):
        _make_model(tmp_path / "standard-model")
        _make_gguf_only(tmp_path / "gguf-model")
        result = list_models(tmp_path)
        names = {m["name"] for m in result}
        assert names == {"standard-model", "gguf-model"}

    def test_empty_directory_returns_empty_list(self, tmp_path):
        result = list_models(tmp_path)
        assert result == []

    def test_nonexistent_directory_returns_empty_list(self, tmp_path):
        result = list_models(tmp_path / "nope")
        assert result == []

    def test_does_not_double_count_gguf_with_config(self, tmp_path):
        """A model with both config.json and .gguf should appear once, not twice."""
        path = tmp_path / "combo-model"
        path.mkdir()
        (path / "config.json").write_text(json.dumps({"model_type": "llama"}))
        (path / "model-q4.gguf").write_bytes(b"\x00" * 128)
        result = list_models(tmp_path)
        assert len(result) == 1
        assert result[0]["format"] == "GGUF"

    def test_includes_modified_timestamp(self, tmp_path):
        _make_model(tmp_path / "my-model")
        result = list_models(tmp_path)
        assert "modified" in result[0]
        assert isinstance(result[0]["modified"], float)


# ── remove_model ─────────────────────────────────────────────────


class TestRemoveModel:
    def test_deletes_model_and_returns_freed_bytes(self, tmp_path):
        _make_model(tmp_path / "doomed-model")
        freed = remove_model("doomed-model", tmp_path)
        assert freed > 0
        assert not (tmp_path / "doomed-model").exists()

    def test_raises_for_nonexistent_model(self, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            remove_model("ghost-model", tmp_path)

    def test_name_resolution_slash_to_dash(self, tmp_path):
        """org/model should resolve to org--model directory."""
        _make_model(tmp_path / "org--model")
        freed = remove_model("org/model", tmp_path)
        assert freed > 0
        assert not (tmp_path / "org--model").exists()


# ── model_info ───────────────────────────────────────────────────


class TestModelInfo:
    def test_standard_model_metadata(self, tmp_path):
        _make_model(tmp_path / "info-model", model_type="llama")
        info = model_info("info-model", tmp_path)
        assert info["name"] == "info-model"
        assert info["format"] == "SafeTensors"
        assert info["model_type"] == "llama"
        assert info["architectures"] == ["LlamaForCausalLM"]
        assert info["size"] > 0
        assert info["file_count"] >= 1
        assert info["has_gguf"] is False
        assert info["has_lora"] is False
        assert "modified" in info
        assert "path" in info

    def test_gguf_only_model_metadata(self, tmp_path):
        _make_gguf_only(tmp_path / "gguf-info")
        info = model_info("gguf-info", tmp_path)
        assert info["name"] == "gguf-info"
        assert info["format"] == "GGUF"
        assert info["model_type"] == "unknown"
        assert info["architectures"] == []
        assert info["has_gguf"] is True
        assert info["file_count"] == 1

    def test_lora_model_metadata(self, tmp_path):
        _make_lora(tmp_path / "lora-model")
        info = model_info("lora-model", tmp_path)
        assert info["has_lora"] is True
        assert info["format"] == "SafeTensors"

    def test_raises_for_nonexistent_model(self, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            model_info("nope", tmp_path)

    def test_name_resolution_slash(self, tmp_path):
        _make_model(tmp_path / "org--mymodel")
        info = model_info("org/mymodel", tmp_path)
        assert info["name"] == "org--mymodel"

    def test_name_resolution_double_dash(self, tmp_path):
        _make_model(tmp_path / "org--mymodel")
        info = model_info("org--mymodel", tmp_path)
        assert info["name"] == "org--mymodel"
