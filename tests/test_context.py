"""Tests for the context management module."""

import os
import tempfile
from pathlib import Path

from froggy.context import (
    PROFILES,
    ContextManager,
    TokenUsage,
    build_tool_instructions,
    context_limit_for_model,
    estimate_messages_tokens,
    estimate_tokens,
    read_file_for_injection,
    summarize_messages,
)


class TestEstimateTokens:
    def test_empty_string(self):
        assert estimate_tokens("") == 1  # max(1, 0)

    def test_short_string(self):
        result = estimate_tokens("hello")
        assert result >= 1

    def test_longer_string(self):
        text = "a" * 400
        result = estimate_tokens(text)
        assert 80 <= result <= 120

    def test_model_param_accepted(self):
        result = estimate_tokens("hello world", model="gpt-4")
        assert result >= 1


class TestEstimateMessagesTokens:
    def test_empty_messages(self):
        assert estimate_messages_tokens([]) == 0

    def test_single_message(self):
        msgs = [{"role": "user", "content": "hello"}]
        result = estimate_messages_tokens(msgs)
        assert result >= 5

    def test_multiple_messages(self):
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi there!"},
            {"role": "assistant", "content": "Hello!"},
        ]
        result = estimate_messages_tokens(msgs)
        assert result > 12


class TestTokenUsage:
    def test_initial_state(self):
        usage = TokenUsage()
        assert usage.prompt_tokens == 0
        assert usage.total_tokens == 0
        assert usage.requests == 0

    def test_record(self):
        usage = TokenUsage()
        usage.record(100, 50)
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150
        assert usage.requests == 1

    def test_record_accumulates(self):
        usage = TokenUsage()
        usage.record(100, 50)
        usage.record(200, 100)
        assert usage.total_tokens == 450
        assert usage.requests == 2

    def test_summary(self):
        usage = TokenUsage()
        usage.record(100, 50)
        s = usage.summary()
        assert "100" in s
        assert "50" in s
        assert "requests: 1" in s

    def test_elapsed(self):
        usage = TokenUsage()
        assert usage.elapsed >= 0


class TestContextLimitForModel:
    def test_explicit_limit(self):
        assert context_limit_for_model("anything", 32000) == 32000

    def test_gpt4(self):
        assert context_limit_for_model("gpt-4") == 8_192

    def test_gpt4o(self):
        assert context_limit_for_model("gpt-4o-mini") == 128_000

    def test_claude(self):
        assert context_limit_for_model("claude-3-opus") == 200_000

    def test_llama(self):
        assert context_limit_for_model("llama-3.1-8b") == 8_192

    def test_unknown_model(self):
        assert context_limit_for_model("totally-unknown-model-xyz") == 4_096

    def test_nemotron(self):
        assert context_limit_for_model("nemotron-nano-9b") == 128_000


class TestBuildToolInstructions:
    def test_includes_tool_block(self):
        result = build_tool_instructions("my tools here")
        assert "my tools here" in result
        assert "Tool System" in result

    def test_with_examples(self):
        result = build_tool_instructions("tools", include_examples=True)
        assert "read_file" in result
        assert "Format Reference" in result

    def test_without_examples(self):
        result = build_tool_instructions("tools", include_examples=False)
        assert "Format Reference" not in result
        assert "tools" in result


class TestReadFileForInjection:
    def test_read_existing_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello world")
            f.flush()
            result = read_file_for_injection(f.name)
            assert result is not None
            assert "hello world" in result
            os.unlink(f.name)

    def test_nonexistent_file(self):
        result = read_file_for_injection("/nonexistent/file.txt")
        assert result is None

    def test_truncation(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("x" * 100_000)
            f.flush()
            result = read_file_for_injection(f.name, max_tokens=100)
            assert result is not None
            assert "truncated" in result
            assert len(result) < 100_000
            os.unlink(f.name)


class TestSummarizeMessages:
    def test_basic_summary(self):
        msgs = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language."},
        ]
        result = summarize_messages(msgs)
        assert "User:" in result
        assert "Assistant:" in result

    def test_tool_call_compressed(self):
        msgs = [
            {"role": "assistant", "content": '<tool_call>{"name": "read_file", "arguments": {"path": "foo.py"}}</tool_call>'},
        ]
        result = summarize_messages(msgs)
        assert "read_file" in result

    def test_tool_response_compressed(self):
        msgs = [
            {"role": "user", "content": '<tool_response>\n{"ok": true, "output": "lots of content..."}\n</tool_response>'},
        ]
        result = summarize_messages(msgs)
        assert "successfully" in result.lower()

    def test_system_messages_skipped(self):
        msgs = [
            {"role": "system", "content": "You are a bot."},
            {"role": "user", "content": "Hi"},
        ]
        result = summarize_messages(msgs)
        assert "bot" not in result
        assert "User:" in result

    def test_truncation(self):
        msgs = [{"role": "user", "content": "x" * 10000}]
        result = summarize_messages(msgs, max_tokens=50)
        assert len(result) < 10000


class TestProfiles:
    def test_all_profiles_exist(self):
        assert "minimal" in PROFILES
        assert "standard" in PROFILES
        assert "full" in PROFILES

    def test_minimal_has_no_examples(self):
        assert PROFILES["minimal"]["inject_tool_examples"] is False

    def test_full_has_no_auto_summarize(self):
        assert PROFILES["full"]["summarize_after_turns"] == 0

    def test_standard_has_examples(self):
        assert PROFILES["standard"]["inject_tool_examples"] is True


class TestContextManager:
    def _make_messages(self, n: int, content_size: int = 100) -> list[dict]:
        msgs = [{"role": "system", "content": "You are helpful."}]
        for i in range(n):
            role = "user" if i % 2 == 0 else "assistant"
            msgs.append({"role": role, "content": "x" * content_size})
        return msgs

    def test_no_trim_when_within_budget(self):
        mgr = ContextManager(context_limit=100_000)
        msgs = self._make_messages(3)
        original_len = len(msgs)
        result = mgr.trim_if_needed(msgs)
        assert len(result) == original_len
        assert mgr._trim_count == 0

    def test_trim_when_over_budget(self):
        mgr = ContextManager(context_limit=100, profile="full")
        msgs = self._make_messages(20, content_size=50)
        original_len = len(msgs)
        result = mgr.trim_if_needed(msgs)
        assert len(result) < original_len

    def test_trim_preserves_system_message(self):
        mgr = ContextManager(context_limit=100, profile="full")
        msgs = self._make_messages(20, content_size=50)
        result = mgr.trim_if_needed(msgs)
        assert result[0]["role"] == "system"

    def test_trim_respects_minimum_messages(self):
        mgr = ContextManager(context_limit=10, profile="full")
        msgs = self._make_messages(2, content_size=50)
        result = mgr.trim_if_needed(msgs)
        assert len(result) >= 3

    def test_status(self):
        mgr = ContextManager(context_limit=10_000)
        msgs = self._make_messages(5)
        status = mgr.status(msgs)
        assert "current_tokens" in status
        assert "profile" in status
        assert status["profile"] == "standard"
        assert "injected_items" in status
        assert "summarizations" in status

    def test_available_tokens(self):
        mgr = ContextManager(context_limit=10_000, reserve_ratio=0.15)
        assert mgr.available_tokens == 8500

    # -- Injection tests --

    def test_inject_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("project context here")
            f.flush()
            mgr = ContextManager(context_limit=10_000)
            assert mgr.inject_file(f.name) is True
            items = mgr.list_injected()
            assert len(items) == 1
            os.unlink(f.name)

    def test_inject_file_nonexistent(self):
        mgr = ContextManager(context_limit=10_000)
        assert mgr.inject_file("/nonexistent/path.txt") is False

    def test_remove_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("content")
            f.flush()
            mgr = ContextManager(context_limit=10_000)
            mgr.inject_file(f.name)
            assert mgr.remove_file(f.name) is True
            assert len(mgr.list_injected()) == 0
            os.unlink(f.name)

    def test_inject_block(self):
        mgr = ContextManager(context_limit=10_000)
        mgr.inject_block("Coding Standards", "Always use type hints.")
        items = mgr.list_injected()
        assert "[block] Coding Standards" in items

    def test_remove_block(self):
        mgr = ContextManager(context_limit=10_000)
        mgr.inject_block("Rules", "Be concise.")
        assert mgr.remove_block("Rules") is True
        assert len(mgr.list_injected()) == 0

    # -- Profile tests --

    def test_set_profile(self):
        mgr = ContextManager(context_limit=10_000)
        assert mgr.set_profile("minimal") is True
        assert mgr.profile_name == "minimal"
        assert mgr.profile["inject_tool_examples"] is False

    def test_set_invalid_profile(self):
        mgr = ContextManager(context_limit=10_000)
        assert mgr.set_profile("nonexistent") is False
        assert mgr.profile_name == "standard"  # unchanged

    # -- Build system context --

    def test_build_system_context_basic(self):
        mgr = ContextManager(context_limit=10_000)
        result = mgr.build_system_context("You are helpful.")
        assert "You are helpful." in result

    def test_build_system_context_with_tools(self):
        mgr = ContextManager(context_limit=10_000)
        result = mgr.build_system_context("You are helpful.", tool_block="[tool defs]")
        assert "Tool System" in result
        assert "[tool defs]" in result
        assert "Format Reference" in result  # standard profile includes examples

    def test_build_system_context_minimal_no_examples(self):
        mgr = ContextManager(context_limit=10_000, profile="minimal")
        result = mgr.build_system_context("You are helpful.", tool_block="[tool defs]")
        assert "Tool System" in result
        assert "Format Reference" not in result

    def test_build_system_context_with_injected(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("important project info")
            f.flush()
            mgr = ContextManager(context_limit=10_000)
            mgr.inject_file(f.name)
            result = mgr.build_system_context("Base prompt.")
            assert "important project info" in result
            os.unlink(f.name)

    def test_build_system_context_with_block(self):
        mgr = ContextManager(context_limit=10_000)
        mgr.inject_block("Rules", "Never use eval().")
        result = mgr.build_system_context("Base prompt.")
        assert "Never use eval()" in result

    def test_build_system_context_with_summary(self):
        mgr = ContextManager(context_limit=10_000)
        mgr._conversation_summary = "User asked about Python. Assistant explained basics."
        result = mgr.build_system_context("Base prompt.")
        assert "Earlier Conversation Summary" in result
        assert "Python" in result

    # -- Summarization --

    def test_maybe_summarize_below_threshold(self):
        mgr = ContextManager(context_limit=100_000)
        msgs = self._make_messages(3)
        result = mgr.maybe_summarize(msgs)
        assert len(result) == len(msgs)  # no change
        assert mgr._summarize_count == 0

    def test_maybe_summarize_above_threshold(self):
        mgr = ContextManager(context_limit=100_000, profile="minimal")
        # minimal profile: summarize_after_turns = 4
        msgs = [{"role": "system", "content": "system"}]
        for i in range(6):
            msgs.append({"role": "user", "content": f"message {i}"})
            msgs.append({"role": "assistant", "content": f"reply {i}"})
        original_len = len(msgs)
        result = mgr.maybe_summarize(msgs)
        assert len(result) < original_len
        assert mgr._summarize_count == 1
        assert mgr._conversation_summary is not None

    def test_maybe_summarize_disabled_in_full_profile(self):
        mgr = ContextManager(context_limit=100_000, profile="full")
        msgs = [{"role": "system", "content": "system"}]
        for i in range(20):
            msgs.append({"role": "user", "content": f"message {i}"})
            msgs.append({"role": "assistant", "content": f"reply {i}"})
        result = mgr.maybe_summarize(msgs)
        assert len(result) == len(msgs)  # no change — full profile doesn't summarize

    # -- Fresh context --

    def test_fresh_context(self):
        mgr = ContextManager(context_limit=10_000)
        mgr._conversation_summary = "old summary"
        mgr._trim_count = 3
        mgr._summarize_count = 2
        mgr.inject_block("Rules", "Be nice.")

        mgr.fresh_context()
        assert mgr._conversation_summary is None
        assert mgr._trim_count == 0
        assert mgr._summarize_count == 0
        # Injected items are preserved
        assert len(mgr.list_injected()) == 1

    # -- Auto-inject project --

    def test_auto_inject_project(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake pyproject.toml
            (Path(tmpdir) / "pyproject.toml").write_text("[project]\nname = 'test'\n")
            mgr = ContextManager(context_limit=10_000)
            count = mgr.auto_inject_project(tmpdir)
            assert count >= 1
            items = mgr.list_injected()
            assert any("pyproject.toml" in k for k in items)

    def test_auto_inject_project_respects_profile(self):
        mgr = ContextManager(context_limit=10_000, profile="minimal")
        count = mgr.auto_inject_project("/nonexistent")
        assert count == 0  # minimal profile disables project injection

    def test_multiple_trims_increment_counter(self):
        # Use full profile to disable summarization so trimming kicks in
        mgr = ContextManager(context_limit=200, profile="full")
        msgs = self._make_messages(20, content_size=50)
        mgr.trim_if_needed(msgs)
        assert mgr._trim_count >= 1

        for i in range(20):
            msgs.append({"role": "user", "content": "x" * 50})
        mgr.trim_if_needed(msgs)
        assert mgr._trim_count >= 2
