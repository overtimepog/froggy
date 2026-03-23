"""Parse tool calls from LLM output (Hermes XML and bare JSON formats)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

# Hermes-style tool call: <tool_call>...</tool_call>
_HERMES_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    raw: str  # the original matched text


class ToolCallParser:
    """Stateful parser that detects tool calls in streaming LLM output.

    Supports:
    - Hermes XML format: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    - Bare JSON fallback: {"name": "...", "arguments": {...}}
    """

    def __init__(self):
        self._buffer = ""

    def feed(self, chunk: str) -> list[ToolCall]:
        """Feed a streaming chunk. Returns any complete tool calls found."""
        self._buffer += chunk
        calls, self._buffer = self._extract_calls(self._buffer)
        return calls

    def flush(self) -> list[ToolCall]:
        """Flush the buffer and attempt to parse any remaining content."""
        calls, self._buffer = self._extract_calls(self._buffer, final=True)
        return calls

    def reset(self) -> None:
        self._buffer = ""

    @property
    def buffer(self) -> str:
        return self._buffer

    def _extract_calls(self, text: str, final: bool = False) -> tuple[list[ToolCall], str]:
        calls: list[ToolCall] = []
        remaining = text

        # --- Hermes XML pass ---
        while True:
            m = _HERMES_RE.search(remaining)
            if not m:
                break
            inner = m.group(1).strip()
            call = _try_parse_json(inner, m.group(0))
            if call:
                calls.append(call)
            remaining = remaining[: m.start()] + remaining[m.end() :]

        # --- Bare JSON fallback (only when no partial XML tag is open) ---
        if "<tool_call>" not in remaining:
            parsed, remaining = _extract_bare_json_calls(remaining)
            calls.extend(parsed)

        return calls, remaining


def _try_parse_json(text: str, raw: str) -> ToolCall | None:
    # First try direct parse
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "name" in data:
            return ToolCall(
                name=data["name"],
                arguments=data.get("arguments", {}),
                raw=raw,
            )
    except (json.JSONDecodeError, TypeError):
        pass

    # Fallback: try to extract a JSON object from noisy text
    # Models sometimes add trailing brackets, whitespace, or other junk
    start = text.find("{")
    if start >= 0:
        obj_text, _ = _extract_json_object(text, start)
        if obj_text is not None:
            try:
                data = json.loads(obj_text)
                if isinstance(data, dict) and "name" in data:
                    return ToolCall(
                        name=data["name"],
                        arguments=data.get("arguments", {}),
                        raw=raw,
                    )
            except (json.JSONDecodeError, TypeError):
                pass

    return None


def _extract_bare_json_calls(text: str) -> tuple[list[ToolCall], str]:
    """Find and extract bare JSON tool calls from text."""
    calls: list[ToolCall] = []
    consumed_ranges: list[tuple[int, int]] = []

    pos = 0
    while pos < len(text):
        start = text.find("{", pos)
        if start == -1:
            break
        obj_text, end = _extract_json_object(text, start)
        if obj_text is None:
            pos = start + 1
            continue
        call = _try_parse_json(obj_text, obj_text)
        # Require both "name" and "arguments" keys to avoid false positives
        if call is not None and "arguments" in json.loads(obj_text):
            calls.append(call)
            consumed_ranges.append((start, end))
            pos = end
        else:
            pos = start + 1

    # Remove consumed ranges from text (reverse order preserves indices)
    remaining = text
    for s, e in reversed(consumed_ranges):
        remaining = remaining[:s] + remaining[e:]

    return calls, remaining


def _extract_json_object(text: str, start: int) -> tuple[str | None, int]:
    """Extract a complete JSON object starting at `start`.

    Returns (obj_str, end_pos) or (None, -1) if the object is incomplete.
    """
    depth = 0
    in_string = False
    escape_next = False
    i = start
    while i < len(text):
        ch = text[i]
        if escape_next:
            escape_next = False
        elif ch == "\\" and in_string:
            escape_next = True
        elif ch == '"':
            in_string = not in_string
        elif not in_string:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1], i + 1
        i += 1
    return None, -1


def parse_tool_calls(text: str) -> list[ToolCall]:
    """One-shot parse of a complete (non-streaming) text."""
    parser = ToolCallParser()
    calls = parser.feed(text)
    calls.extend(parser.flush())
    return calls
