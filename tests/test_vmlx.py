from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from froggy.backends import VMLXBackend, _vmlx_available, pick_backend
from froggy.discovery import ModelInfo, discover_models
from froggy.download import download_model


def _make_model_info(**kwargs) -> ModelInfo:
    defaults = dict(
        name="Gemma-4-31B-JANG_4M-CRACK",
        path=Path("/fake/models/Gemma-4-31B-JANG_4M-CRACK"),
        has_jang=True,
    )
    defaults.update(kwargs)
    return ModelInfo(**defaults)


class TestPickBackendVMLX:
    def test_jang_models_get_vmlx_backend(self):
        backend = pick_backend(_make_model_info())
        assert isinstance(backend, VMLXBackend)


class TestVmlxAvailability:
    def test_available_when_vmlx_on_path(self):
        with patch("shutil.which", return_value="/opt/homebrew/bin/vmlx"):
            assert _vmlx_available() is True

    def test_unavailable_when_missing(self):
        with patch("shutil.which", return_value=None):
            assert _vmlx_available() is False


class TestDiscoveryJang:
    def test_discovers_jang_models(self, tmp_path):
        model_dir = tmp_path / "dealignai--Gemma-4-31B-JANG_4M-CRACK"
        model_dir.mkdir()
        (model_dir / "config.json").write_text('{"model_type": "gemma4", "architectures": ["Gemma4ForConditionalGeneration"]}')
        (model_dir / "jang_config.json").write_text('{"format": "jang"}')
        (model_dir / "model-00001-of-00005.safetensors").write_bytes(b"x")

        models = discover_models(tmp_path)

        assert len(models) == 1
        assert models[0].has_jang is True


class TestVMLXBackend:
    def test_load_starts_local_vmlx_server(self):
        backend = VMLXBackend()
        proc = MagicMock()
        model = _make_model_info(path=Path("/models/jang"))

        with patch("froggy.backends._is_apple_silicon", return_value=True), \
             patch("froggy.backends._find_vmlx_cli", return_value="/opt/homebrew/bin/vmlx"), \
             patch.object(backend, "_find_free_port", return_value=8123), \
             patch.object(backend, "_wait_until_ready"), \
             patch("subprocess.Popen", return_value=proc) as mock_popen:
            backend.load(model, "auto")

        assert backend._process is proc
        assert backend._port == 8123
        assert backend._model_name == model.name
        cmd = mock_popen.call_args.args[0]
        assert cmd[:2] == ["/opt/homebrew/bin/vmlx", "serve"]
        assert str(model.path) in cmd
        assert "--port" in cmd
        assert "8123" in cmd
        assert "--served-model-name" in cmd
        assert model.name in cmd

    def test_load_adds_gemma4_parser_flags(self):
        backend = VMLXBackend()
        proc = MagicMock()
        model = _make_model_info(
            path=Path("/models/jang"),
            model_type="gemma4",
            architectures=["Gemma4ForConditionalGeneration"],
        )

        with patch("froggy.backends._is_apple_silicon", return_value=True), \
             patch("froggy.backends._find_vmlx_cli", return_value="/opt/homebrew/bin/vmlx"), \
             patch.object(backend, "_find_free_port", return_value=8123), \
             patch.object(backend, "_wait_until_ready"), \
             patch("subprocess.Popen", return_value=proc) as mock_popen:
            backend.load(model, "auto")

        cmd = mock_popen.call_args.args[0]
        assert "--tool-call-parser" in cmd
        assert "--reasoning-parser" in cmd
        assert cmd[cmd.index("--tool-call-parser") + 1] == "gemma4"
        assert cmd[cmd.index("--reasoning-parser") + 1] == "gemma4"

    def test_load_requires_apple_silicon(self):
        backend = VMLXBackend()
        with patch("froggy.backends._is_apple_silicon", return_value=False):
            with pytest.raises(RuntimeError, match="Apple Silicon"):
                backend.load(_make_model_info(), "auto")

    def test_generate_stream_parses_openai_sse(self):
        backend = VMLXBackend()
        backend._model_name = "dealignai/Gemma-4-31B-JANG_4M-CRACK"
        backend._base_url = "http://127.0.0.1:8123/v1"

        payloads = [
            b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n',
            b'\n',
            b'data: {"choices":[{"delta":{"content":" world"}}]}\n',
            b'data: [DONE]\n',
        ]
        response = io.BytesIO(b"".join(payloads))

        with patch("urllib.request.urlopen", return_value=response):
            chunks = list(backend.generate_stream([
                {"role": "user", "content": "Hi"},
            ], temperature=0.2, max_tokens=64))

        assert chunks == ["Hello", " world"]


class TestDownloadForJang:
    @patch("froggy.download.ensure_froggy_home")
    @patch("froggy.download.models_dir")
    @patch("froggy.download.snapshot_download")
    @patch("froggy.download.find_mlx_repo", return_value=None)
    @patch("froggy.download.list_gguf_files", return_value=[])
    @patch("froggy.download.detect_platform", return_value="mlx")
    def test_safetensors_download_keeps_chat_template_jinja(
        self, _detect, _list_gguf, _find_mlx, mock_snapshot, mock_models_dir, _home
    ):
        mock_models_dir.return_value = Path("/tmp/test_models")

        download_model("dealignai/Gemma-4-31B-JANG_4M-CRACK")

        allow_patterns = mock_snapshot.call_args.kwargs["allow_patterns"]
        assert "*.jinja" in allow_patterns
