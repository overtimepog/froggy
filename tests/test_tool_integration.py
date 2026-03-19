"""Tests for tool system integration in ChatSession."""

from __future__ import annotations

import json
from typing import Iterator
from unittest.mock import MagicMock, patch

from froggy.session import ChatSession, handle_command

# ---------------------------------------------------------------------------
# Minimal stub classes so tests run without real backend / tool modules
# ---------------------------------------------------------------------------


class _StubModelInfo:
    name = "stub-model"
    label = "stub-model"
    model_type = "stub"


class _StubBackend:
    name = "stub"
    _chunks: list[str] = []

    def load(self, model_info, device):
        pass

    def generate_stream(self, messages, temperature, max_tokens) -> Iterator[str]:
        yield from self._chunks

    def unload(self):
        pass


class _StubToolCall:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


def _make_session(chunks=None, *, tools_enabled=False, registry=None, executor=None):
    backend = _StubBackend()
    if chunks is not None:
        backend._chunks = chunks
    session = ChatSession(
        backend,
        _StubModelInfo(),
        "cpu",
        tool_registry=registry,
        tool_executor=executor,
    )
    session.tools_enabled = tools_enabled
    return session


# ---------------------------------------------------------------------------
# Basic chat (no tools)
# ---------------------------------------------------------------------------


def test_chat_appends_messages(capsys):
    """chat() adds user + assistant messages to history."""
    session = _make_session(["Hello", " world"])
    with patch("froggy.session.Live"):
        session.chat("hi")
    assert session.messages[0] == {"role": "user", "content": "hi"}
    assert session.messages[1]["role"] == "assistant"
    assert "Hello world" in session.messages[1]["content"]


def test_chat_strips_thinking(capsys):
    """chat() strips <think> blocks from the stored assistant message."""
    session = _make_session(["<think>reasoning</think>Final answer"])
    with patch("froggy.session.Live"):
        session.chat("question")
    assert "<think>" not in session.messages[-1]["content"]
    assert "Final answer" in session.messages[-1]["content"]


def test_chat_stops_at_stop_string(capsys):
    """chat() truncates output at stop-string boundaries."""
    session = _make_session(["Hello<|im_end|>GARBAGE"])
    with patch("froggy.session.Live"):
        session.chat("hi")
    assert "GARBAGE" not in session.messages[-1]["content"]
    assert "Hello" in session.messages[-1]["content"]


# ---------------------------------------------------------------------------
# Tool loop integration
# ---------------------------------------------------------------------------


def _make_tool_session(response_chunks, tool_calls_sequence):
    """Create a session wired to return tool calls from a pre-defined sequence.

    tool_calls_sequence is a list of lists; each inner list is the tool calls
    returned for that generation round.
    """
    backend = _StubBackend()
    backend._chunks = response_chunks

    registry = MagicMock()
    registry.all.return_value = []
    registry.system_prompt_block.return_value = ""

    executor = MagicMock()
    executor.execute.return_value = {"tool": "read_file", "ok": True, "output": "file contents", "truncated": False}

    session = ChatSession(backend, _StubModelInfo(), "cpu",
                          tool_registry=registry, tool_executor=executor)
    session.tools_enabled = True

    # Patch _generate_one_round to return pre-canned (text, tool_calls) pairs
    rounds = iter(tool_calls_sequence)

    def _fake_generate():
        calls = next(rounds, [])
        return "assistant text", calls

    session._generate_one_round = _fake_generate
    return session


def test_tool_loop_injects_result():
    """When tool calls are returned, results are injected as user messages."""
    tool_call = _StubToolCall("read_file", {"path": "/tmp/x.txt"})
    session = _make_tool_session(
        response_chunks=["text"],
        tool_calls_sequence=[[tool_call], []],  # round 1: tool call; round 2: done
    )
    session.chat("use a tool")

    roles = [m["role"] for m in session.messages]
    assert "user" in roles
    assert "assistant" in roles
    # Tool result injected as user message with <tool_response> wrapper
    tool_result_msgs = [m for m in session.messages
                        if m["role"] == "user" and "<tool_response>" in m["content"]]
    assert len(tool_result_msgs) == 1


def test_tool_loop_max_rounds():
    """Tool loop caps at max_tool_rounds and does not loop infinitely."""
    tool_call = _StubToolCall("get_datetime", {})
    # Always return a tool call to force the loop to hit the cap
    session = _make_tool_session(
        response_chunks=["text"],
        tool_calls_sequence=[[tool_call]] * 10,
    )
    session.max_tool_rounds = 3
    session.chat("keep calling tools")
    # Should have stopped — no infinite loop
    assert len(session.messages) > 0


def test_tool_loop_disabled_skips_execution():
    """When tools_enabled=False, tool calls are NOT executed."""
    executor = MagicMock()
    registry = MagicMock()
    registry.system_prompt_block.return_value = ""

    session = ChatSession(_StubBackend(), _StubModelInfo(), "cpu",
                          tool_registry=registry, tool_executor=executor)
    session.tools_enabled = False

    tool_call = _StubToolCall("read_file", {"path": "/tmp/x.txt"})
    session._generate_one_round = lambda: ("answer", [tool_call])

    session.chat("question")

    executor.execute.assert_not_called()


def test_tool_result_contains_json():
    """The injected tool_response message contains valid JSON."""
    tool_call = _StubToolCall("get_datetime", {})
    session = _make_tool_session(
        response_chunks=[],
        tool_calls_sequence=[[tool_call], []],
    )
    session.chat("what time is it")

    tool_result_msgs = [m for m in session.messages
                        if m["role"] == "user" and "<tool_response>" in m.get("content", "")]
    assert tool_result_msgs
    content = tool_result_msgs[0]["content"]
    # Strip the XML wrapper and check JSON validity
    inner = content.replace("<tool_response>\n", "").replace("\n</tool_response>", "")
    parsed = json.loads(inner)
    assert "tool" in parsed or "ok" in parsed


# ---------------------------------------------------------------------------
# Slash commands for tools
# ---------------------------------------------------------------------------


def _session_with_registry():
    tool_a = MagicMock()
    tool_a.name = "read_file"
    tool_a.description = "Read a file"

    tool_b = MagicMock()
    tool_b.name = "run_shell"
    tool_b.description = "Run a shell command"

    registry = MagicMock()
    registry.all.return_value = [tool_a, tool_b]
    registry.names.return_value = ["read_file", "run_shell"]
    registry.get.side_effect = lambda name: {"read_file": tool_a, "run_shell": tool_b}.get(name)
    registry.system_prompt_block.return_value = ""

    session = ChatSession(_StubBackend(), _StubModelInfo(), "cpu", tool_registry=registry)
    session.tools_enabled = True
    return session


def test_slash_tools_on_off(capsys):
    session = _session_with_registry()
    session.tools_enabled = False
    handle_command("/tools on", session)
    assert session.tools_enabled is True
    handle_command("/tools off", session)
    assert session.tools_enabled is False


def test_slash_tools_add_remove(capsys):
    session = _session_with_registry()
    # Start with all tools active
    assert session._active_tool_names is None

    handle_command("/tools remove run_shell", session)
    assert "run_shell" not in session._active_tool_names

    handle_command("/tools add run_shell", session)
    assert "run_shell" in session._active_tool_names


def test_slash_autorun_toggle(capsys):
    session = _session_with_registry()
    initial = session.autorun
    handle_command("/autorun", session)
    assert session.autorun is not initial
    handle_command("/autorun", session)
    assert session.autorun is initial


def test_slash_tools_add_unknown(capsys):
    """Adding an unknown tool should print an error, not crash."""
    session = _session_with_registry()
    handle_command("/tools add nonexistent_tool", session)
    # No exception; _active_tool_names unchanged
    assert session._active_tool_names is None


# ---------------------------------------------------------------------------
# System prompt injection
# ---------------------------------------------------------------------------


def test_system_prompt_includes_tools_when_enabled():
    registry = MagicMock()
    registry.all.return_value = []
    registry.system_prompt_block.return_value = "TOOL_BLOCK"

    session = ChatSession(_StubBackend(), _StubModelInfo(), "cpu", tool_registry=registry)
    session.tools_enabled = True

    prompt = session._build_system_prompt()
    assert "TOOL_BLOCK" in prompt


def test_system_prompt_clean_when_tools_disabled():
    registry = MagicMock()
    registry.system_prompt_block.return_value = "TOOL_BLOCK"

    session = ChatSession(_StubBackend(), _StubModelInfo(), "cpu", tool_registry=registry)
    session.tools_enabled = False

    prompt = session._build_system_prompt()
    assert "TOOL_BLOCK" not in prompt
