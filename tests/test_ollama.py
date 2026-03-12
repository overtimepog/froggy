"""Tests for OllamaBackend — written before implementation (TDD)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from froggy.backends import OllamaBackend, pick_backend
from froggy.discovery import ModelInfo

# ── helpers ────────────────────────────────────────────────────────────────


def _make_model_info(**kwargs) -> ModelInfo:
    defaults = dict(name="llama3", path=Path("/fake"), is_ollama=True)
    defaults.update(kwargs)
    return ModelInfo(**defaults)


def _fake_tags_response(model_names: list[str]) -> dict:
    """Build a fake GET /api/tags JSON response."""
    return {
        "models": [
            {
                "name": name,
                "model": name,
                "size": 4_000_000_000,
                "details": {
                    "family": "llama",
                    "parameter_size": "8B",
                    "quantization_level": "Q4_0",
                },
            }
            for name in model_names
        ]
    }


def _fake_chat_chunks(tokens: list[str]) -> list[bytes]:
    """Build fake streaming NDJSON lines from POST /api/chat."""
    lines = []
    for tok in tokens:
        lines.append(
            json.dumps({"message": {"role": "assistant", "content": tok}, "done": False}).encode()
            + b"\n"
        )
    lines.append(
        json.dumps({"message": {"role": "assistant", "content": ""}, "done": True}).encode()
        + b"\n"
    )
    return lines


# ── pick_backend routing ──────────────────────────────────────────────────


class TestPickBackendOllama:
    def test_ollama_model_gets_ollama_backend(self):
        m = _make_model_info()
        backend = pick_backend(m)
        assert isinstance(backend, OllamaBackend)

    def test_local_model_does_not_get_ollama(self):
        m = ModelInfo(name="test", path=Path("/fake"))
        backend = pick_backend(m)
        assert not isinstance(backend, OllamaBackend)


# ── OllamaBackend basics ─────────────────────────────────────────────────


class TestOllamaBackendInit:
    def test_default_base_url(self):
        backend = OllamaBackend()
        assert backend.base_url == "http://localhost:11434"

    def test_custom_base_url(self):
        backend = OllamaBackend(base_url="http://myhost:9999")
        assert backend.base_url == "http://myhost:9999"

    def test_name(self):
        assert OllamaBackend().name == "ollama"


# ── load() ────────────────────────────────────────────────────────────────


class TestOllamaLoad:
    def test_load_sets_model_name(self):
        backend = OllamaBackend()
        m = _make_model_info(name="mistral:7b")
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(
                _fake_tags_response(["mistral:7b"])
            ).encode()
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp
            backend.load(m, "auto")
        assert backend._model_name == "mistral:7b"

    def test_load_fails_if_server_unreachable(self):
        backend = OllamaBackend()
        m = _make_model_info()
        with patch("urllib.request.urlopen", side_effect=ConnectionError("refused")):
            with pytest.raises(ConnectionError):
                backend.load(m, "auto")

    def test_load_fails_if_model_not_on_server(self):
        backend = OllamaBackend()
        m = _make_model_info(name="nonexistent:latest")
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(
                _fake_tags_response(["llama3:latest"])
            ).encode()
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp
            with pytest.raises(ValueError, match="not found"):
                backend.load(m, "auto")


# ── generate_stream() ────────────────────────────────────────────────────


class TestOllamaGenerateStream:
    def _loaded_backend(self) -> OllamaBackend:
        b = OllamaBackend()
        b._model_name = "llama3:latest"
        return b

    def test_yields_tokens_from_stream(self):
        backend = self._loaded_backend()
        chunks = _fake_chat_chunks(["Hello", " world", "!"])

        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.__iter__ = lambda s: iter(chunks)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            tokens = list(backend.generate_stream(
                [{"role": "user", "content": "Hi"}], 0.7, 100
            ))
        assert tokens == ["Hello", " world", "!"]

    def test_passes_temperature_and_max_tokens(self):
        backend = self._loaded_backend()
        chunks = _fake_chat_chunks(["ok"])
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.__iter__ = lambda s: iter(chunks)

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
            list(backend.generate_stream(
                [{"role": "user", "content": "test"}], 0.3, 256
            ))
            # Inspect the request body that was sent
            call_args = mock_open.call_args
            req = call_args[0][0]
            body = json.loads(req.data)
            assert body["options"]["temperature"] == 0.3
            assert body["options"]["num_predict"] == 256

    def test_sends_messages_in_body(self):
        backend = self._loaded_backend()
        chunks = _fake_chat_chunks(["hi"])
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.__iter__ = lambda s: iter(chunks)

        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hello"},
        ]
        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
            list(backend.generate_stream(messages, 0.7, 100))
            req = mock_open.call_args[0][0]
            body = json.loads(req.data)
            assert body["messages"] == messages
            assert body["model"] == "llama3:latest"

    def test_not_loaded_raises(self):
        backend = OllamaBackend()
        with pytest.raises(RuntimeError, match="not loaded"):
            list(backend.generate_stream([], 0.7, 100))


# ── unload() ─────────────────────────────────────────────────────────────


class TestOllamaUnload:
    def test_unload_clears_state(self):
        backend = OllamaBackend()
        backend._model_name = "test"
        backend.unload()
        assert backend._model_name is None

    def test_unload_when_empty(self):
        backend = OllamaBackend()
        backend.unload()  # should not raise


# ── discover_ollama_models() ─────────────────────────────────────────────


class TestDiscoverOllamaModels:
    def test_returns_model_infos(self):
        from froggy.discovery import discover_ollama_models

        tags = _fake_tags_response(["llama3:latest", "mistral:7b"])
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(tags).encode()
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp
            models = discover_ollama_models()

        assert len(models) == 2
        assert models[0].name == "llama3:latest"
        assert models[0].is_ollama is True
        assert models[1].name == "mistral:7b"

    def test_returns_empty_when_server_down(self):
        from froggy.discovery import discover_ollama_models

        with patch("urllib.request.urlopen", side_effect=OSError("refused")):
            models = discover_ollama_models()
        assert models == []

    def test_label_shows_ollama_tag(self):
        m = _make_model_info(name="llama3:latest")
        assert "Ollama" in m.label
