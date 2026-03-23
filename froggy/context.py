"""Context window management — injection, tracking, compression, and trimming.

Provides a ContextManager that programmatically builds and manages
the conversation context.  Inspired by gsd-2's dispatch pipeline where
every generation gets exactly the context it needs — no more, no less.

Key capabilities:
- **Context injection** — inject files, instructions, and tool defs into context
- **Context profiles** — minimal / standard / full control over what's inlined
- **Token tracking** — count tokens per message and across the session
- **Auto-trimming** — compress or drop old messages when approaching the limit
- **Summarization** — condense earlier conversation into a compact summary
- **Fresh context** — reset to a clean state with injected artifacts only

Token counting uses a lightweight character-based estimator by default
(~4 chars ≈ 1 token).  When *tiktoken* is installed, a proper BPE
tokenizer is used instead.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

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
# Context profiles
# ---------------------------------------------------------------------------

PROFILES = {
    "minimal": {
        "description": "Bare minimum — system prompt + tool defs only",
        "inject_tool_examples": False,
        "inject_project_files": False,
        "max_injected_file_tokens": 0,
        "summarize_after_turns": 4,
        "max_prior_summary_tokens": 500,
    },
    "standard": {
        "description": "Balanced — tools with examples, project context on request",
        "inject_tool_examples": True,
        "inject_project_files": True,
        "max_injected_file_tokens": 4000,
        "summarize_after_turns": 10,
        "max_prior_summary_tokens": 2000,
    },
    "full": {
        "description": "Everything inlined — max context, no compression",
        "inject_tool_examples": True,
        "inject_project_files": True,
        "max_injected_file_tokens": 16000,
        "summarize_after_turns": 0,  # 0 = never auto-summarize
        "max_prior_summary_tokens": 8000,
    },
}


# ---------------------------------------------------------------------------
# Context injection helpers
# ---------------------------------------------------------------------------

def build_tool_instructions(tool_block: str, include_examples: bool = True) -> str:
    """Build clear tool-use instructions to inject into context.

    This is the key to making tool use reliable across models — we don't
    just list the tools, we show the model exactly how to format calls
    and what to expect back.
    """
    instructions = (
        "# Tool Use Instructions\n\n"
        "You have tools available. To call a tool, output EXACTLY this format:\n\n"
        '<tool_call>{"name": "TOOL_NAME_HERE", "arguments": {"param": "value"}}</tool_call>\n\n'
        "Rules:\n"
        "- Replace TOOL_NAME_HERE with the actual tool name from the list below\n"
        "- Output the <tool_call> tag with valid JSON inside — no other format\n"
        "- Wait for the <tool_response> before continuing\n"
        "- You can call multiple tools in sequence across turns\n"
        "- Always use the exact parameter names from the tool definitions\n"
        "- If a tool returns an error, explain the issue and try a different approach\n\n"
    )

    if include_examples:
        instructions += (
            "## Format Examples (do NOT call these — they show the format only)\n\n"
            "Example of reading a file:\n"
            '<tool_call>{"name": "read_file", "arguments": {"path": "src/main.py"}}</tool_call>\n\n'
            "Example of running a shell command:\n"
            '<tool_call>{"name": "run_shell", "arguments": {"cmd": "ls -la"}}</tool_call>\n\n'
            "Example of searching the web:\n"
            '<tool_call>{"name": "web_search", "arguments": {"query": "python asyncio tutorial"}}</tool_call>\n\n'
            "Example of running Python code:\n"
            '<tool_call>{"name": "python_eval", "arguments": {"code": "import math; math.factorial(10)"}}</tool_call>\n\n'
        )

    instructions += "## Available Tools\n\n" + tool_block
    return instructions


def read_file_for_injection(path: str, max_tokens: int = 4000) -> str | None:
    """Read a file and truncate to fit within a token budget.

    Returns None if the file doesn't exist or can't be read.
    """
    try:
        p = Path(path).expanduser()
        if not p.is_file():
            return None
        content = p.read_text(encoding="utf-8", errors="replace")
        # Rough truncation: 4 chars per token
        max_chars = max_tokens * 4
        if len(content) > max_chars:
            content = content[:max_chars] + "\n\n[... truncated to fit context window]"
        return content
    except OSError:
        return None


def summarize_messages(messages: list[dict], max_tokens: int = 500) -> str:
    """Create a compressed summary of a conversation chunk.

    This is a deterministic heuristic summarizer — no LLM calls.
    Extracts the key information from each message and compresses it.
    """
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")

        if role == "system":
            continue  # skip system messages in summary

        if "<tool_call>" in content:
            # Extract just the tool name
            import re
            match = re.search(r'"name"\s*:\s*"([^"]+)"', content)
            tool_name = match.group(1) if match else "unknown"
            parts.append(f"[Called tool: {tool_name}]")
        elif "<tool_response>" in content:
            # Compress tool responses heavily
            if '"ok": true' in content or '"ok":true' in content:
                parts.append("[Tool returned successfully]")
            else:
                parts.append("[Tool returned an error]")
        elif role == "user":
            # Keep user messages but truncate
            short = content[:200].strip()
            if len(content) > 200:
                short += "..."
            parts.append(f"User: {short}")
        elif role == "assistant":
            # Keep assistant messages but truncate
            short = content[:300].strip()
            if len(content) > 300:
                short += "..."
            parts.append(f"Assistant: {short}")

    summary = "\n".join(parts)

    # Final truncation to token budget
    max_chars = max_tokens * 4
    if len(summary) > max_chars:
        summary = summary[:max_chars] + "\n[... summary truncated]"

    return summary


# ---------------------------------------------------------------------------
# Context limit detection
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
    ("nemotron", 128_000),
    ("llama", 8_192),
    ("gemma", 8_192),
    ("phi", 4_096),
]

# Percentage of context to reserve for the response
_RESPONSE_RESERVE = 0.15

# Minimum messages to keep (system + at least 1 user/assistant pair)
_MIN_MESSAGES = 3


def context_limit_for_model(model_name: str, explicit_limit: int | None = None) -> int:
    """Determine the context window size for a model."""
    if explicit_limit is not None:
        return explicit_limit

    name_lower = model_name.lower()
    for key, limit in _DEFAULT_CONTEXT_LIMITS:
        if key in name_lower:
            return limit
    return 4_096


# ---------------------------------------------------------------------------
# ContextManager
# ---------------------------------------------------------------------------


class ContextManager:
    """Manages a conversation's context window programmatically.

    Builds context like gsd-2's dispatch pipeline: each generation gets
    a clean, purpose-built context window with exactly the artifacts it
    needs.  Supports:

    - Injecting files, instructions, and tool definitions
    - Context profiles (minimal / standard / full)
    - Token counting and budget tracking
    - Auto-summarization of old turns
    - Auto-trimming as a last resort
    - Fresh context resets

    Parameters
    ----------
    context_limit : int
        Maximum tokens for the model's context window.
    reserve_ratio : float
        Fraction of context_limit reserved for the model's response.
    model : str or None
        Model name passed to the token estimator.
    profile : str
        Context profile — 'minimal', 'standard', or 'full'.
    """

    def __init__(
        self,
        context_limit: int = 4096,
        reserve_ratio: float = _RESPONSE_RESERVE,
        model: str | None = None,
        profile: str = "standard",
    ):
        self.context_limit = context_limit
        self.reserve_ratio = reserve_ratio
        self.model = model
        self.profile_name = profile
        self.profile = PROFILES.get(profile, PROFILES["standard"])
        self.usage = TokenUsage()
        self._trim_count = 0
        self._summarize_count = 0

        # Injected context blocks — rebuilt on each generation
        self._injected_files: dict[str, str] = {}  # path → content
        self._injected_blocks: dict[str, str] = {}  # label → content
        self._conversation_summary: str | None = None  # compressed prior turns

    # ------------------------------------------------------------------
    # Context injection
    # ------------------------------------------------------------------

    def inject_file(self, path: str, label: str | None = None, max_tokens: int | None = None) -> bool:
        """Inject a file's contents into the context window.

        The file will be included in the system prompt on every generation
        until removed.  Returns True if the file was successfully read.
        """
        budget = max_tokens or self.profile.get("max_injected_file_tokens", 4000)
        content = read_file_for_injection(path, max_tokens=budget)
        if content is None:
            return False
        key = label or path
        self._injected_files[key] = f"# File: {path}\n\n```\n{content}\n```"
        return True

    def remove_file(self, path_or_label: str) -> bool:
        """Remove an injected file from context."""
        if path_or_label in self._injected_files:
            del self._injected_files[path_or_label]
            return True
        return False

    def inject_block(self, label: str, content: str) -> None:
        """Inject a named content block into the context.

        Blocks are included in the system prompt.  Use for instructions,
        project rules, coding standards, etc.  Call again with the same
        label to replace.
        """
        self._injected_blocks[label] = content

    def remove_block(self, label: str) -> bool:
        """Remove an injected content block."""
        if label in self._injected_blocks:
            del self._injected_blocks[label]
            return True
        return False

    def list_injected(self) -> dict[str, int]:
        """Return a dict of injected items and their estimated token counts."""
        items = {}
        for key, content in self._injected_files.items():
            items[f"[file] {key}"] = estimate_tokens(content, self.model)
        for key, content in self._injected_blocks.items():
            items[f"[block] {key}"] = estimate_tokens(content, self.model)
        if self._conversation_summary:
            items["[summary]"] = estimate_tokens(self._conversation_summary, self.model)
        return items

    def set_profile(self, profile: str) -> bool:
        """Switch to a different context profile."""
        if profile not in PROFILES:
            return False
        self.profile_name = profile
        self.profile = PROFILES[profile]
        return True

    # ------------------------------------------------------------------
    # System prompt builder
    # ------------------------------------------------------------------

    def build_system_context(self, base_system_prompt: str, tool_block: str | None = None) -> str:
        """Build the complete system prompt with all injected context.

        This is the core of the context engineering — assembles the system
        prompt from:
        1. Base system prompt
        2. Tool instructions (with examples based on profile)
        3. Conversation summary (if earlier turns were compressed)
        4. Injected files
        5. Injected content blocks
        """
        parts = [base_system_prompt]

        # Tool instructions — always inject when tools are available
        if tool_block:
            include_examples = self.profile.get("inject_tool_examples", True)
            parts.append(build_tool_instructions(tool_block, include_examples))

        # Conversation summary from earlier turns
        if self._conversation_summary:
            parts.append(
                "# Earlier Conversation Summary\n\n"
                "The following is a compressed summary of the earlier part of this conversation:\n\n"
                + self._conversation_summary
            )

        # Injected files
        if self._injected_files and self.profile.get("inject_project_files", True):
            parts.append("# Injected Project Files\n")
            for content in self._injected_files.values():
                parts.append(content)

        # Injected content blocks
        for label, content in self._injected_blocks.items():
            parts.append(f"# {label}\n\n{content}")

        return "\n\n---\n\n".join(parts)

    # ------------------------------------------------------------------
    # Summarization
    # ------------------------------------------------------------------

    def maybe_summarize(self, messages: list[dict]) -> list[dict]:
        """Summarize older messages if the conversation is getting long.

        Checks the profile's `summarize_after_turns` threshold.  When
        triggered, compresses the oldest messages into a summary that
        gets injected into the system prompt, and removes those messages.

        Returns the (possibly shortened) message list.
        """
        threshold = self.profile.get("summarize_after_turns", 0)
        if threshold <= 0:
            return messages  # summarization disabled

        # Count non-system user/assistant turns
        turns = [m for m in messages if m["role"] in ("user", "assistant")]
        if len(turns) < threshold:
            return messages

        # Summarize the first half of turns
        split_point = len(turns) // 2
        # Find the actual index in messages for the split point
        turn_count = 0
        split_idx = 0
        for i, msg in enumerate(messages):
            if msg["role"] in ("user", "assistant"):
                turn_count += 1
                if turn_count >= split_point:
                    split_idx = i + 1
                    break

        if split_idx <= 0:
            return messages

        # Build summary of older messages
        old_messages = messages[:split_idx]
        max_summary_tokens = self.profile.get("max_prior_summary_tokens", 2000)
        new_summary = summarize_messages(old_messages, max_tokens=max_summary_tokens)

        # Merge with existing summary if any
        if self._conversation_summary:
            combined = self._conversation_summary + "\n\n---\n\n" + new_summary
            # Truncate combined summary to budget
            max_chars = max_summary_tokens * 4
            if len(combined) > max_chars:
                combined = combined[-max_chars:]
            self._conversation_summary = combined
        else:
            self._conversation_summary = new_summary

        self._summarize_count += 1

        # Keep only the newer messages
        return messages[split_idx:]

    # ------------------------------------------------------------------
    # Auto-trimming (last resort)
    # ------------------------------------------------------------------

    @property
    def available_tokens(self) -> int:
        """Max tokens available for prompt (context minus response reserve)."""
        return int(self.context_limit * (1 - self.reserve_ratio))

    def count_tokens(self, messages: list[dict]) -> int:
        """Count tokens in a message list."""
        return estimate_messages_tokens(messages, self.model)

    def trim_if_needed(self, messages: list[dict]) -> list[dict]:
        """Trim the message list if it exceeds the available token budget.

        First tries summarization, then drops oldest messages as a last resort.
        """
        budget = self.available_tokens
        current = self.count_tokens(messages)

        if current <= budget:
            return messages

        # First: try summarization
        messages = self.maybe_summarize(messages)
        if self.count_tokens(messages) <= budget:
            return messages

        # Last resort: drop oldest non-system messages in pairs (user+assistant)
        system_msg = None
        if messages and messages[0]["role"] == "system":
            system_msg = messages[0]

        trimmed = False
        while len(messages) > _MIN_MESSAGES and self.count_tokens(messages) > budget:
            idx = 1 if system_msg else 0
            # Remove the message at idx
            removed_role = messages[idx]["role"]
            messages.pop(idx)
            trimmed = True
            # If we removed a user message and next is assistant (or vice versa),
            # drop the partner too to keep turns paired
            if (len(messages) > idx and
                    messages[idx]["role"] != removed_role and
                    messages[idx]["role"] in ("user", "assistant") and
                    len(messages) > _MIN_MESSAGES):
                messages.pop(idx)

        if trimmed:
            self._trim_count += 1
            marker_idx = 1 if system_msg else 0
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

    # ------------------------------------------------------------------
    # Fresh context reset
    # ------------------------------------------------------------------

    def fresh_context(self) -> None:
        """Reset to a clean context — preserve injected artifacts but clear
        conversation state.  Like gsd-2's fresh-session-per-task approach."""
        self._conversation_summary = None
        self._trim_count = 0
        self._summarize_count = 0

    # ------------------------------------------------------------------
    # Auto-inject project context
    # ------------------------------------------------------------------

    def auto_inject_project(self, project_root: str | None = None) -> int:
        """Scan a project root and inject common context files.

        Looks for README, CLAUDE.md, .cursorrules, pyproject.toml, etc.
        Returns the number of files injected.
        """
        if not self.profile.get("inject_project_files", True):
            return 0

        root = Path(project_root) if project_root else Path.cwd()
        if not root.is_dir():
            return 0

        count = 0
        candidates = [
            ("README.md", 2000),
            ("CLAUDE.md", 2000),
            (".cursorrules", 1000),
            ("pyproject.toml", 500),
            ("package.json", 500),
            (".froggy/context.md", 2000),
        ]

        max_total = self.profile.get("max_injected_file_tokens", 4000)
        used = 0

        for filename, default_budget in candidates:
            if used >= max_total:
                break
            filepath = root / filename
            if filepath.is_file():
                budget = min(default_budget, max_total - used)
                if self.inject_file(str(filepath), label=filename, max_tokens=budget):
                    used += estimate_tokens(self._injected_files[filename], self.model)
                    count += 1

        return count

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self, messages: list[dict]) -> dict:
        """Return a status dict with current context usage."""
        current = self.count_tokens(messages)
        budget = self.available_tokens
        injected = self.list_injected()
        injected_tokens = sum(injected.values())
        return {
            "current_tokens": current,
            "available_tokens": budget,
            "context_limit": self.context_limit,
            "utilization": current / budget if budget > 0 else 0,
            "messages": len(messages),
            "trims": self._trim_count,
            "summarizations": self._summarize_count,
            "injected_items": len(injected),
            "injected_tokens": injected_tokens,
            "profile": self.profile_name,
            "has_summary": self._conversation_summary is not None,
            "usage": self.usage.summary(),
        }
