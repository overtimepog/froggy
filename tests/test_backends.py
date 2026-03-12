"""Tests for backend selection and interface."""

import pytest
from pathlib import Path

from froggy.discovery import ModelInfo
from froggy.backends import pick_backend, TransformersBackend, LlamaCppBackend


class TestPickBackend:
    def test_gguf_gets_llamacpp(self):
        m = ModelInfo(name="test", path=Path("/fake"), has_gguf=True)
        backend = pick_backend(m)
        assert isinstance(backend, LlamaCppBackend)

    def test_safetensors_gets_transformers(self):
        m = ModelInfo(name="test", path=Path("/fake"))
        backend = pick_backend(m)
        assert isinstance(backend, TransformersBackend)

    def test_lora_gets_transformers(self):
        m = ModelInfo(name="test", path=Path("/fake"), has_lora=True)
        backend = pick_backend(m)
        assert isinstance(backend, TransformersBackend)


class TestLlamaCppBackend:
    def test_not_implemented(self):
        backend = LlamaCppBackend()
        m = ModelInfo(name="test", path=Path("/fake"), has_gguf=True)
        with pytest.raises(NotImplementedError):
            backend.load(m, "cpu")

    def test_unload_is_safe(self):
        backend = LlamaCppBackend()
        backend.unload()  # should not raise


class TestTransformersBackend:
    def test_initial_state(self):
        backend = TransformersBackend()
        assert backend.model is None
        assert backend.tokenizer is None
        assert backend.name == "transformers"

    def test_unload_when_empty(self):
        backend = TransformersBackend()
        backend.unload()  # should not raise
