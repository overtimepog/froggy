"""Tests for backend selection and interface."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from froggy.backends import (
    LlamaCppBackend,
    TransformersBackend,
    _find_llama_cli,
    _format_chat_prompt,
    pick_backend,
)
from froggy.discovery import ModelInfo


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
    def test_initial_state(self):
        backend = LlamaCppBackend()
        assert backend._exe is None
        assert backend._gguf_path is None
        assert backend._process is None
        assert backend.name == "llama.cpp"

    def test_load_no_executable(self, tmp_path):
        """load() raises FileNotFoundError when llama-cli is not found."""
        gguf_file = tmp_path / "model.gguf"
        gguf_file.write_bytes(b"\x00" * 64)
        m = ModelInfo(name="test", path=tmp_path, has_gguf=True)
        backend = LlamaCppBackend()
        with patch("froggy.backends._find_llama_cli", return_value=None):
            with pytest.raises(FileNotFoundError, match="llama-cli"):
                backend.load(m, "cpu")

    def test_load_no_gguf_files(self, tmp_path):
        """load() raises FileNotFoundError when directory has no GGUF files."""
        m = ModelInfo(name="test", path=tmp_path, has_gguf=True)
        backend = LlamaCppBackend()
        with patch("froggy.backends._find_llama_cli", return_value="/usr/bin/llama-cli"):
            with pytest.raises(FileNotFoundError, match="No .gguf files"):
                backend.load(m, "cpu")

    def test_load_success(self, tmp_path):
        """load() succeeds when llama-cli and GGUF file are available."""
        gguf_file = tmp_path / "model-q4.gguf"
        gguf_file.write_bytes(b"\x00" * 1024)
        m = ModelInfo(name="test", path=tmp_path, has_gguf=True)
        backend = LlamaCppBackend()
        with patch("froggy.backends._find_llama_cli", return_value="/usr/bin/llama-cli"):
            with patch("subprocess.run"):
                backend.load(m, "cpu")
        assert backend._exe == "/usr/bin/llama-cli"
        assert backend._gguf_path == gguf_file
        assert backend._gpu is False

    def test_load_picks_first_gguf_alphabetically(self, tmp_path):
        """When multiple GGUF files exist, picks the first one sorted."""
        (tmp_path / "b-model.gguf").write_bytes(b"\x00" * 64)
        (tmp_path / "a-model.gguf").write_bytes(b"\x00" * 64)
        m = ModelInfo(name="test", path=tmp_path, has_gguf=True)
        backend = LlamaCppBackend()
        with patch("froggy.backends._find_llama_cli", return_value="/usr/bin/llama-cli"):
            with patch("subprocess.run"):
                backend.load(m, "cpu")
        assert backend._gguf_path.name == "a-model.gguf"

    def test_load_gpu_flag(self, tmp_path):
        """load() sets GPU offload when device is not 'cpu'."""
        (tmp_path / "model.gguf").write_bytes(b"\x00" * 64)
        m = ModelInfo(name="test", path=tmp_path, has_gguf=True)
        backend = LlamaCppBackend()
        with patch("froggy.backends._find_llama_cli", return_value="/usr/bin/llama-cli"):
            with patch("subprocess.run"):
                backend.load(m, "auto")
        assert backend._gpu is True

    def test_generate_stream_not_loaded(self):
        """generate_stream() raises RuntimeError if load() wasn't called."""
        backend = LlamaCppBackend()
        with pytest.raises(RuntimeError, match="not loaded"):
            list(backend.generate_stream([], 0.7, 100))

    def test_generate_stream_reads_stdout(self, tmp_path):
        """generate_stream() yields characters from llama-cli stdout."""
        (tmp_path / "model.gguf").write_bytes(b"\x00" * 64)
        backend = LlamaCppBackend()
        backend._exe = "/usr/bin/llama-cli"
        backend._gguf_path = tmp_path / "model.gguf"
        backend._gpu = False

        mock_proc = MagicMock()
        mock_proc.stdout.read = MagicMock(side_effect=list("Hello!") + [""])
        mock_proc.wait = MagicMock()

        with patch("subprocess.Popen", return_value=mock_proc):
            result = "".join(backend.generate_stream(
                [{"role": "user", "content": "Hi"}], 0.7, 100
            ))
        assert result == "Hello!"

    def test_unload_when_empty(self):
        backend = LlamaCppBackend()
        backend.unload()  # should not raise

    def test_unload_kills_process(self):
        """unload() terminates a running process."""
        backend = LlamaCppBackend()
        mock_proc = MagicMock()
        backend._process = mock_proc
        backend.unload()
        mock_proc.terminate.assert_called_once()
        assert backend._process is None


class TestFormatChatPrompt:
    def test_single_user_message(self):
        prompt = _format_chat_prompt([{"role": "user", "content": "Hello"}])
        assert "<|im_start|>user\nHello<|im_end|>" in prompt
        assert prompt.endswith("<|im_start|>assistant\n")

    def test_system_and_user(self):
        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hi"},
        ]
        prompt = _format_chat_prompt(messages)
        assert "<|im_start|>system\nBe helpful<|im_end|>" in prompt
        assert "<|im_start|>user\nHi<|im_end|>" in prompt

    def test_multi_turn(self):
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"},
        ]
        prompt = _format_chat_prompt(messages)
        assert prompt.count("<|im_start|>") == 4  # 3 messages + assistant prompt
        assert prompt.count("<|im_end|>") == 3


class TestFindLlamaCli:
    def test_found_on_path(self):
        with patch("shutil.which", return_value="/usr/bin/llama-cli"):
            assert _find_llama_cli() == "/usr/bin/llama-cli"

    def test_not_found(self):
        with patch("shutil.which", return_value=None):
            with patch("pathlib.Path.is_file", return_value=False):
                assert _find_llama_cli() is None


class TestTransformersBackend:
    def test_initial_state(self):
        backend = TransformersBackend()
        assert backend.model is None
        assert backend.tokenizer is None
        assert backend.name == "transformers"

    def test_unload_when_empty(self):
        backend = TransformersBackend()
        backend.unload()  # should not raise
