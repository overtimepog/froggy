"""Tests for the context management module."""

from froggy.context import (
    ContextManager,
    TokenUsage,
    context_limit_for_model,
    estimate_messages_tokens,
    estimate_tokens,
)


class TestEstimateTokens:
    def test_empty_string(self):
        assert estimate_tokens("") == 1  # max(1, 0)

    def test_short_string(self):
        # "hello" = 5 chars → ~1 token with fallback
        result = estimate_tokens("hello")
        assert result >= 1

    def test_longer_string(self):
        text = "a" * 400
        result = estimate_tokens(text)
        # ~100 tokens with 4-char heuristic
        assert 80 <= result <= 120

    def test_model_param_accepted(self):
        # Should not crash with model param
        result = estimate_tokens("hello world", model="gpt-4")
        assert result >= 1


class TestEstimateMessagesTokens:
    def test_empty_messages(self):
        assert estimate_messages_tokens([]) == 0

    def test_single_message(self):
        msgs = [{"role": "user", "content": "hello"}]
        result = estimate_messages_tokens(msgs)
        # 4 overhead + ~1 for "hello"
        assert result >= 5

    def test_multiple_messages(self):
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi there!"},
            {"role": "assistant", "content": "Hello!"},
        ]
        result = estimate_messages_tokens(msgs)
        assert result > 12  # 3 messages × 4 overhead = 12 min


class TestTokenUsage:
    def test_initial_state(self):
        usage = TokenUsage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
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
        assert usage.prompt_tokens == 300
        assert usage.completion_tokens == 150
        assert usage.total_tokens == 450
        assert usage.requests == 2

    def test_summary(self):
        usage = TokenUsage()
        usage.record(100, 50)
        s = usage.summary()
        assert "100" in s
        assert "50" in s
        assert "150" in s
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


class TestContextManager:
    def _make_messages(self, n: int, content_size: int = 100) -> list[dict]:
        """Create n messages with the given content size."""
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
        # Very small context limit to force trimming
        mgr = ContextManager(context_limit=100)
        msgs = self._make_messages(20, content_size=50)
        original_len = len(msgs)
        result = mgr.trim_if_needed(msgs)
        assert len(result) < original_len
        assert mgr._trim_count == 1

    def test_trim_preserves_system_message(self):
        mgr = ContextManager(context_limit=100)
        msgs = self._make_messages(20, content_size=50)
        result = mgr.trim_if_needed(msgs)
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are helpful."

    def test_trim_inserts_marker(self):
        mgr = ContextManager(context_limit=100)
        msgs = self._make_messages(20, content_size=50)
        result = mgr.trim_if_needed(msgs)
        # Should have a trim marker after the system message
        has_marker = any(
            m["content"].startswith("[Earlier conversation")
            for m in result
            if m["role"] == "system"
        )
        assert has_marker

    def test_trim_respects_minimum_messages(self):
        mgr = ContextManager(context_limit=10)  # absurdly small
        msgs = self._make_messages(2, content_size=50)
        result = mgr.trim_if_needed(msgs)
        assert len(result) >= 3  # _MIN_MESSAGES

    def test_status(self):
        mgr = ContextManager(context_limit=10_000)
        msgs = self._make_messages(5)
        status = mgr.status(msgs)
        assert "current_tokens" in status
        assert "available_tokens" in status
        assert "context_limit" in status
        assert status["context_limit"] == 10_000
        assert "utilization" in status
        assert "messages" in status
        assert status["messages"] == 6  # 1 system + 5

    def test_available_tokens(self):
        mgr = ContextManager(context_limit=10_000, reserve_ratio=0.15)
        assert mgr.available_tokens == 8500

    def test_count_tokens(self):
        mgr = ContextManager(context_limit=10_000)
        msgs = [{"role": "user", "content": "hello world"}]
        count = mgr.count_tokens(msgs)
        assert count > 0

    def test_multiple_trims_increment_counter(self):
        mgr = ContextManager(context_limit=200)
        msgs = self._make_messages(20, content_size=50)
        mgr.trim_if_needed(msgs)
        assert mgr._trim_count == 1

        # Add more messages and trim again
        for i in range(20):
            msgs.append({"role": "user", "content": "x" * 50})
        mgr.trim_if_needed(msgs)
        assert mgr._trim_count == 2
