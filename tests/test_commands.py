"""Tests for slash command handling."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from froggy.discovery import ModelInfo
from froggy.session import ChatSession, handle_command


@pytest.fixture
def session():
    """Create a ChatSession with a mock backend."""
    backend = MagicMock()
    backend.name = "mock"
    model_info = ModelInfo(name="test-model", path=Path("/fake"))
    return ChatSession(backend, model_info, "cpu")


class TestHandleCommand:
    def test_quit(self, session):
        assert handle_command("/quit", session) == "quit"

    def test_exit(self, session):
        assert handle_command("/exit", session) == "quit"

    def test_q(self, session):
        assert handle_command("/q", session) == "quit"

    def test_model_switch(self, session):
        assert handle_command("/model", session) == "switch"

    def test_clear(self, session):
        session.messages = [{"role": "user", "content": "hi"}]
        handle_command("/clear", session)
        assert session.messages == []

    def test_system_set(self, session):
        handle_command("/system You are a pirate.", session)
        assert session.system_prompt == "You are a pirate."

    def test_system_show(self, session):
        result = handle_command("/system", session)
        assert result is None  # no action, just prints

    def test_temp_valid(self, session):
        handle_command("/temp 0.5", session)
        assert session.temperature == 0.5

    def test_temp_zero(self, session):
        handle_command("/temp 0.0", session)
        assert session.temperature == 0.0

    def test_temp_boundary(self, session):
        handle_command("/temp 2.0", session)
        assert session.temperature == 2.0

    def test_temp_out_of_range(self, session):
        original = session.temperature
        handle_command("/temp 5.0", session)
        assert session.temperature == original

    def test_temp_no_arg(self, session):
        result = handle_command("/temp", session)
        assert result is None

    def test_tokens_valid(self, session):
        handle_command("/tokens 2048", session)
        assert session.max_tokens == 2048

    def test_tokens_zero(self, session):
        original = session.max_tokens
        handle_command("/tokens 0", session)
        assert session.max_tokens == original

    def test_tokens_negative(self, session):
        original = session.max_tokens
        handle_command("/tokens -10", session)
        assert session.max_tokens == original

    def test_tokens_no_arg(self, session):
        result = handle_command("/tokens", session)
        assert result is None

    def test_help(self, session):
        result = handle_command("/help", session)
        assert result is None

    def test_info(self, session):
        result = handle_command("/info", session)
        assert result is None

    def test_unknown_command(self, session):
        result = handle_command("/foobar", session)
        assert result is None

    def test_case_insensitive(self, session):
        assert handle_command("/QUIT", session) == "quit"
        assert handle_command("/Quit", session) == "quit"

    def test_extra_whitespace(self, session):
        assert handle_command("  /quit  ", session) == "quit"
        handle_command("  /temp 0.3  ", session)
        assert session.temperature == 0.3


class TestChatSession:
    def test_initial_state(self, session):
        assert session.messages == []
        assert session.temperature == 0.7
        assert session.max_tokens == 1024
        assert session.system_prompt == "You are a helpful assistant."

    def test_clear(self, session):
        session.messages = [{"role": "user", "content": "test"}]
        session.clear()
        assert session.messages == []
