"""Tests for MLXBackend — Apple Silicon inference via mlx-lm."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from froggy.backends import MLXBackend, _is_apple_silicon, _mlx_available, pick_backend
from froggy.discovery import ModelInfo

# ── helpers ────────────────────────────────────────────────────────────────


def _make_model_info(**kwargs) -> ModelInfo:
    defaults = dict(name="Llama-3-8B-4bit", path=Path("/fake/models/Llama-3-8B-4bit"))
    defaults.update(kwargs)
    return ModelInfo(**defaults)


# ── pick_backend routing ──────────────────────────────────────────────────


class TestPickBackendMLX:
    def test_mlx_preferred_on_apple_silicon(self):
        """On Apple Silicon with mlx-lm installed, safetensors models get MLX."""
        m = _make_model_info()
        with patch("froggy.backends._is_apple_silicon", return_value=True):
            with patch("froggy.backends._mlx_available", return_value=True):
                backend = pick_backend(m)
        assert isinstance(backend, MLXBackend)

    def test_falls_back_to_transformers_without_mlx(self):
        """Without mlx-lm, safetensors models get Transformers."""
        from froggy.backends import TransformersBackend

        m = _make_model_info()
        with patch("froggy.backends._is_apple_silicon", return_value=True):
            with patch("froggy.backends._mlx_available", return_value=False):
                backend = pick_backend(m)
        assert isinstance(backend, TransformersBackend)

    def test_falls_back_to_transformers_on_non_apple(self):
        """On non-Apple-Silicon, safetensors models get Transformers."""
        from froggy.backends import TransformersBackend

        m = _make_model_info()
        with patch("froggy.backends._is_apple_silicon", return_value=False):
            backend = pick_backend(m)
        assert isinstance(backend, TransformersBackend)

    def test_gguf_still_gets_llamacpp_on_apple_silicon(self):
        """GGUF models should still use llama.cpp even on Apple Silicon."""
        from froggy.backends import LlamaCppBackend

        m = _make_model_info(has_gguf=True)
        with patch("froggy.backends._is_apple_silicon", return_value=True):
            with patch("froggy.backends._mlx_available", return_value=True):
                backend = pick_backend(m)
        assert isinstance(backend, LlamaCppBackend)

    def test_ollama_still_gets_ollama_on_apple_silicon(self):
        """Ollama models should still use Ollama even on Apple Silicon."""
        from froggy.backends import OllamaBackend

        m = _make_model_info(is_ollama=True)
        with patch("froggy.backends._is_apple_silicon", return_value=True):
            with patch("froggy.backends._mlx_available", return_value=True):
                backend = pick_backend(m)
        assert isinstance(backend, OllamaBackend)


# ── MLXBackend basics ────────────────────────────────────────────────────


class TestMLXBackendInit:
    def test_initial_state(self):
        backend = MLXBackend()
        assert backend.model is None
        assert backend.tokenizer is None

    def test_name(self):
        assert MLXBackend().name == "mlx"


# ── load() ───────────────────────────────────────────────────────────────


class TestMLXLoad:
    def test_load_sets_model_and_tokenizer(self):
        backend = MLXBackend()
        m = _make_model_info()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_mlx_load = MagicMock(return_value=(mock_model, mock_tokenizer))

        with patch.dict("sys.modules", {"mlx_lm": MagicMock(load=mock_mlx_load)}):
            backend.load(m, "auto")

        assert backend.model is mock_model
        assert backend.tokenizer is mock_tokenizer
        mock_mlx_load.assert_called_once_with(str(m.path))


# ── generate_stream() ────────────────────────────────────────────────────


class TestMLXGenerateStream:
    def _loaded_backend(self) -> MLXBackend:
        b = MLXBackend()
        b.model = MagicMock()
        b.tokenizer = MagicMock()
        b.tokenizer.apply_chat_template.return_value = "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
        return b

    def test_not_loaded_raises(self):
        backend = MLXBackend()
        with pytest.raises(RuntimeError, match="not loaded"):
            list(backend.generate_stream([], 0.7, 100))

    def _make_responses(self, texts):
        """Create mock GenerationResponse objects."""
        responses = []
        for t in texts:
            resp = MagicMock()
            resp.text = t
            responses.append(resp)
        return responses

    def test_yields_tokens(self):
        backend = self._loaded_backend()
        responses = self._make_responses(["Hello", " world", "!"])
        mock_stream = MagicMock(return_value=iter(responses))
        mock_make_sampler = MagicMock(return_value=MagicMock())

        with patch.dict("sys.modules", {
            "mlx_lm": MagicMock(stream_generate=mock_stream),
            "mlx_lm.sample_utils": MagicMock(make_sampler=mock_make_sampler),
        }):
            tokens = list(backend.generate_stream(
                [{"role": "user", "content": "Hi"}], 0.7, 100
            ))
        assert tokens == ["Hello", " world", "!"]

    def test_skips_empty_text(self):
        backend = self._loaded_backend()
        responses = self._make_responses(["Hello", "", "!"])
        mock_stream = MagicMock(return_value=iter(responses))
        mock_make_sampler = MagicMock(return_value=MagicMock())

        with patch.dict("sys.modules", {
            "mlx_lm": MagicMock(stream_generate=mock_stream),
            "mlx_lm.sample_utils": MagicMock(make_sampler=mock_make_sampler),
        }):
            tokens = list(backend.generate_stream(
                [{"role": "user", "content": "Hi"}], 0.7, 100
            ))
        assert tokens == ["Hello", "!"]

    def test_applies_chat_template(self):
        backend = self._loaded_backend()
        responses = self._make_responses(["ok"])
        mock_stream = MagicMock(return_value=iter(responses))
        mock_make_sampler = MagicMock(return_value=MagicMock())

        with patch.dict("sys.modules", {
            "mlx_lm": MagicMock(stream_generate=mock_stream),
            "mlx_lm.sample_utils": MagicMock(make_sampler=mock_make_sampler),
        }):
            list(backend.generate_stream(
                [{"role": "user", "content": "test"}], 0.5, 200
            ))
        backend.tokenizer.apply_chat_template.assert_called_once()

    def test_passes_temperature_and_max_tokens(self):
        backend = self._loaded_backend()
        responses = self._make_responses(["ok"])
        mock_stream = MagicMock(return_value=iter(responses))
        mock_make_sampler = MagicMock(return_value=MagicMock())
        mock_sampler = mock_make_sampler.return_value

        with patch.dict("sys.modules", {
            "mlx_lm": MagicMock(stream_generate=mock_stream),
            "mlx_lm.sample_utils": MagicMock(make_sampler=mock_make_sampler),
        }):
            list(backend.generate_stream(
                [{"role": "user", "content": "test"}], 0.3, 256
            ))
        # Temperature is passed via make_sampler
        mock_make_sampler.assert_called_once_with(temp=0.3)
        # max_tokens and sampler are passed to stream_generate
        call_kwargs = mock_stream.call_args
        assert call_kwargs.kwargs["max_tokens"] == 256
        assert call_kwargs.kwargs["sampler"] is mock_sampler

    def test_fallback_prompt_without_chat_template(self):
        backend = self._loaded_backend()
        del backend.tokenizer.apply_chat_template  # Remove the method
        responses = self._make_responses(["ok"])
        mock_stream = MagicMock(return_value=iter(responses))
        mock_make_sampler = MagicMock(return_value=MagicMock())

        with patch.dict("sys.modules", {
            "mlx_lm": MagicMock(stream_generate=mock_stream),
            "mlx_lm.sample_utils": MagicMock(make_sampler=mock_make_sampler),
        }):
            list(backend.generate_stream(
                [{"role": "user", "content": "Hi"}], 0.7, 100
            ))
        # Should still work with fallback prompt formatting
        call_kwargs = mock_stream.call_args
        prompt = call_kwargs.kwargs["prompt"]
        assert "user: Hi" in prompt
        assert "assistant:" in prompt


# ── unload() ──────────────────────────────────────────────────────────────


class TestMLXUnload:
    def test_unload_clears_state(self):
        backend = MLXBackend()
        backend.model = MagicMock()
        backend.tokenizer = MagicMock()
        backend.unload()
        assert backend.model is None
        assert backend.tokenizer is None

    def test_unload_when_empty(self):
        backend = MLXBackend()
        backend.unload()  # should not raise


# ── platform detection ────────────────────────────────────────────────────


class TestPlatformDetection:
    def test_apple_silicon_detected(self):
        with patch("platform.system", return_value="Darwin"):
            with patch("platform.machine", return_value="arm64"):
                assert _is_apple_silicon() is True

    def test_intel_mac_not_detected(self):
        with patch("platform.system", return_value="Darwin"):
            with patch("platform.machine", return_value="x86_64"):
                assert _is_apple_silicon() is False

    def test_linux_not_detected(self):
        with patch("platform.system", return_value="Linux"):
            with patch("platform.machine", return_value="aarch64"):
                assert _is_apple_silicon() is False

    def test_mlx_available_when_importable(self):
        with patch.dict("sys.modules", {"mlx_lm": MagicMock()}):
            assert _mlx_available() is True

    def test_mlx_unavailable_when_not_importable(self):
        with patch.dict("sys.modules", {"mlx_lm": None}):
            assert _mlx_available() is False
