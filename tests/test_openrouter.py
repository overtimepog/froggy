"""Tests for the OpenRouter backend and discovery."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from froggy.backends import OpenRouterBackend, pick_backend
from froggy.discovery import ModelInfo, discover_openrouter_models


class TestOpenRouterBackend:
    def test_name(self):
        backend = OpenRouterBackend(api_key="test-key")
        assert backend.name == "openrouter"

    @patch.dict("os.environ", {}, clear=False)
    def test_load_without_api_key(self):
        # Remove env var if set
        import os
        os.environ.pop("OPENROUTER_API_KEY", None)
        backend = OpenRouterBackend(api_key="")
        model = ModelInfo(
            name="openai/gpt-4o",
            path=Path("https://openrouter.ai"),
            is_openrouter=True,
        )
        with patch("froggy.config.get_config", return_value=""):
            with pytest.raises(ValueError, match="API key not set"):
                backend.load(model, "auto")

    def test_load_with_api_key(self):
        backend = OpenRouterBackend(api_key="sk-test-123")
        model = ModelInfo(
            name="openai/gpt-4o",
            path=Path("https://openrouter.ai"),
            is_openrouter=True,
            context_length=128_000,
        )
        backend.load(model, "auto")
        assert backend._model_name == "openai/gpt-4o"
        assert backend._context_length == 128_000

    def test_generate_stream_not_loaded(self):
        backend = OpenRouterBackend(api_key="sk-test")
        with pytest.raises(RuntimeError, match="not loaded"):
            list(backend.generate_stream([], 0.7, 100))

    def test_unload(self):
        backend = OpenRouterBackend(api_key="sk-test")
        backend._model_name = "test"
        backend.unload()
        assert backend._model_name is None

    @patch("urllib.request.urlopen")
    def test_generate_stream_parses_sse(self, mock_urlopen):
        """Verify SSE stream parsing."""
        backend = OpenRouterBackend(api_key="sk-test")
        backend._model_name = "openai/gpt-4o"

        # Simulate SSE response
        chunks = [
            b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n',
            b'data: {"choices":[{"delta":{"content":" world"}}]}\n',
            b'data: [DONE]\n',
        ]
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.__iter__ = MagicMock(return_value=iter(chunks))
        mock_urlopen.return_value = mock_resp

        result = list(backend.generate_stream(
            [{"role": "user", "content": "Hi"}], 0.7, 100
        ))
        assert result == ["Hello", " world"]


class TestPickBackendOpenRouter:
    def test_openrouter_model_gets_openrouter_backend(self):
        model = ModelInfo(
            name="anthropic/claude-3-opus",
            path=Path("https://openrouter.ai"),
            is_openrouter=True,
        )
        backend = pick_backend(model)
        assert isinstance(backend, OpenRouterBackend)

    def test_openrouter_takes_priority_over_ollama(self):
        model = ModelInfo(
            name="test",
            path=Path("https://openrouter.ai"),
            is_openrouter=True,
            is_ollama=True,  # hypothetical conflict
        )
        backend = pick_backend(model)
        assert isinstance(backend, OpenRouterBackend)


class TestModelInfoOpenRouter:
    def test_label_openrouter(self):
        model = ModelInfo(
            name="openai/gpt-4o",
            path=Path("https://openrouter.ai"),
            is_openrouter=True,
            context_length=128_000,
        )
        label = model.label
        assert "OpenRouter" in label
        assert "128,000" in label

    def test_label_openrouter_no_context(self):
        model = ModelInfo(
            name="openai/gpt-4o",
            path=Path("https://openrouter.ai"),
            is_openrouter=True,
        )
        label = model.label
        assert "OpenRouter" in label


class TestDiscoverOpenRouterModels:
    @patch("urllib.request.urlopen")
    def test_discover_success(self, mock_urlopen):
        response_data = {
            "data": [
                {
                    "id": "openai/gpt-4o",
                    "context_length": 128000,
                    "architecture": {"modality": "text->text"},
                },
                {
                    "id": "anthropic/claude-3-opus",
                    "context_length": 200000,
                    "architecture": {"modality": "text->text"},
                },
            ]
        }
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = json.dumps(response_data).encode()
        mock_urlopen.return_value = mock_resp

        models = discover_openrouter_models(api_key="sk-test")
        assert len(models) == 2
        assert models[0].name == "openai/gpt-4o"
        assert models[0].is_openrouter is True
        assert models[0].context_length == 128000
        assert models[1].name == "anthropic/claude-3-opus"

    def test_discover_no_api_key(self):
        """Should return empty list when no API key."""
        import os
        env = os.environ.copy()
        env.pop("OPENROUTER_API_KEY", None)
        with patch.dict("os.environ", env, clear=True):
            with patch("froggy.config.get_config", return_value=""):
                models = discover_openrouter_models(api_key="")
                assert models == []

    @patch("urllib.request.urlopen")
    def test_discover_network_error(self, mock_urlopen):
        mock_urlopen.side_effect = OSError("Connection refused")
        models = discover_openrouter_models(api_key="sk-test")
        assert models == []
