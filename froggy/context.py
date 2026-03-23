"""Context window management — token tracking and auto-trimming.

Provides a ContextManager that wraps the raw message list and enforces
a token budget.  When the conversation approaches the model's context
limit, the oldest user/assistant turns are summarized or dropped to make
room for new content.

Token counting uses a lightweight character-based estimator by default
(~4 chars ≈ 1 token).  When *tiktoken* is installed, a proper BPE
tokenizer is used instead.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def _try_tiktoken():
    """Try to import tiktoken for accurate token counting."""
    try:
        import tiktoken
        return tiktoken
    except ImportError:
        return None


def estimate_tokens(text: str, model: str | None = None) -> int:
    """Return an approximate token count for *text*.

    Uses tiktoken when available (with cl100k_base encoding), otherwise
    falls back to a character-based heuristic (~4 chars per token).
    """
    tk = _try_tiktoken()
    if tk is not None:
        try:
            enc = tk.encoding_for_model(model or "gpt-4")
        except KeyError:
            enc = tk.get_encoding("cl100k_base")
        return len(enc.encode(text))
    # Fallback: ~4 characters per token (reasonable for English)
    return max(1, len(text) // 4)


def estimate_messages_tokens(messages: list[dict], model: str | None = None) -> int:
    """Estimate total tokens across a list of chat messages."""
    total = 0
    for msg in messages:
        # ~4 tokens overhead per message (role, separators)
        total += 4
        total += estimate_tokens(msg.get("content", ""), model)
    return total


# ---------------------------------------------------------------------------
# Usage tracking
# ---------------------------------------------------------------------------

@dataclass
class TokenUsage:
    """Cumulative token usage for a session."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    requests: int = 0
    start_time: float = field(default_factory=time.time)

    def record(self, prompt: int, completion: int) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += prompt + completion
        self.requests += 1

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    def summary(self) -> str:
        mins = self.elapsed / 60
        return (
            f"Tokens — prompt: {self.prompt_tokens:,}  "
            f"completion: {self.completion_tokens:,}  "
            f"total: {self.total_tokens:,}  "
            f"requests: {self.requests}  "
            f"elapsed: {mins:.1f}m"
        )


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

# Default context limits by model family (conservative estimates)
# Order matters — more specific prefixes must come before general ones.
_DEFAULT_CONTEXT_LIMITS: list[tuple[str, int]] = [
    ("gpt-4o", 128_000),
    ("gpt-4-turbo", 128_000),
    ("gpt-4", 8_192),
    ("gpt-3.5-turbo", 16_385),
    ("claude", 200_000),
    ("deepseek", 64_000),
    ("mistral", 32_768),
    ("qwen", 32_768),
    ("llama", 8_192),
    ("gemma", 8_192),
    ("phi", 4_096),
]

# Percentage of context to reserve for the response
_RESPONSE_RESERVE = 0.15

# Minimum messages to keep (system + at least 1 user/assistant pair)
_MIN_MESSAGES = 3


def context_limit_for_model(model_name: str, explicit_limit: int | None = None) -> int:
    """Determine the context window size for a model.

    If *explicit_limit* is given, use that.  Otherwise, try to match
    *model_name* against known families.  Falls back to 4096.
    """
    if explicit_limit is not None:
        return explicit_limit

    name_lower = model_name.lower()
    for key, limit in _DEFAULT_CONTEXT_LIMITS:
        if key in name_lower:
            return limit
    return 4_096


class ContextManager:
    """Manages a conversation's message list within a token budget.

    Wraps a list of messages and provides:
    - Token counting per message and total
    - Auto-trimming when the context window is near full
    - Usage tracking across the session

    Parameters
    ----------
    context_limit : int
        Maximum tokens for the model's context window.
    reserve_ratio : float
        Fraction of context_limit reserved for the model's response.
    model : str or None
        Model name passed to the token estimator.
    """

    def __init__(
        self,
        context_limit: int = 4096,
        reserve_ratio: float = _RESPONSE_RESERVE,
        model: str | None = None,
    ):
        self.context_limit = context_limit
        self.reserve_ratio = reserve_ratio
        self.model = model
        self.usage = TokenUsage()
        self._trim_count = 0  # how many times we've auto-trimmed

    @property
    def available_tokens(self) -> int:
        """Max tokens available for prompt (context minus response reserve)."""
        return int(self.context_limit * (1 - self.reserve_ratio))

    def count_tokens(self, messages: list[dict]) -> int:
        """Count tokens in a message list."""
        return estimate_messages_tokens(messages, self.model)

    def trim_if_needed(self, messages: list[dict]) -> list[dict]:
        """Trim the message list if it exceeds the available token budget.

        Removes the oldest user/assistant pairs (preserving the system
        message at index 0) until the total fits within budget.  Inserts
        a brief "[earlier conversation trimmed]" marker so the model
        knows context was lost.

        Returns the (possibly trimmed) message list.  Modifies in place.
        """
        budget = self.available_tokens
        current = self.count_tokens(messages)

        if current <= budget:
            return messages

        # Find the system message (should be index 0)
        system_msg = None
        if messages and messages[0]["role"] == "system":
            system_msg = messages[0]

        # Remove oldest non-system messages until we fit
        trimmed = False
        while len(messages) > _MIN_MESSAGES and self.count_tokens(messages) > budget:
            # Remove the first non-system message
            idx = 1 if system_msg else 0
            messages.pop(idx)
            trimmed = True

        if trimmed:
            self._trim_count += 1
            # Insert a context-loss marker after the system message
            marker_idx = 1 if system_msg else 0
            # Only insert if there isn't already a marker
            if not (len(messages) > marker_idx and
                    messages[marker_idx].get("content", "").startswith("[Earlier conversation")):
                messages.insert(marker_idx, {
                    "role": "system",
                    "content": (
                        "[Earlier conversation trimmed to fit context window. "
                        f"Trim #{self._trim_count}]"
                    ),
                })

        return messages

    def status(self, messages: list[dict]) -> dict:
        """Return a status dict with current context usage."""
        current = self.count_tokens(messages)
        budget = self.available_tokens
        return {
            "current_tokens": current,
            "available_tokens": budget,
            "context_limit": self.context_limit,
            "utilization": current / budget if budget > 0 else 0,
            "messages": len(messages),
            "trims": self._trim_count,
            "usage": self.usage.summary(),
        }
