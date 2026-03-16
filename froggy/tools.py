"""Tool registry for LLM function calling."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolParam:
    name: str
    type: str  # "string", "integer", "boolean", "array", "object"
    description: str
    required: bool = True
    enum: list[str] | None = None


@dataclass
class ToolDef:
    name: str
    description: str
    params: list[ToolParam] = field(default_factory=list)

    def to_json_schema(self) -> dict[str, Any]:
        """Export this tool as a JSON Schema object (OpenAI function format)."""
        properties: dict[str, Any] = {}
        required: list[str] = []
        for p in self.params:
            prop: dict[str, Any] = {"type": p.type, "description": p.description}
            if p.enum:
                prop["enum"] = p.enum
            properties[p.name] = prop
            if p.required:
                required.append(p.name)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


# 6 core tool definitions
CORE_TOOLS: list[ToolDef] = [
    ToolDef(
        name="read_file",
        description="Read the contents of a file at the given path.",
        params=[
            ToolParam("path", "string", "Absolute or relative path to the file to read."),
        ],
    ),
    ToolDef(
        name="write_file",
        description="Write text content to a file, creating it if necessary.",
        params=[
            ToolParam("path", "string", "Path where the file should be written."),
            ToolParam("content", "string", "Text content to write."),
        ],
    ),
    ToolDef(
        name="list_directory",
        description="List files and directories at the given path.",
        params=[
            ToolParam("path", "string", "Directory path to list. Defaults to current directory.", required=False),
        ],
    ),
    ToolDef(
        name="run_shell",
        description="Execute a shell command and return its stdout and stderr.",
        params=[
            ToolParam("command", "string", "Shell command to execute."),
        ],
    ),
    ToolDef(
        name="search_files",
        description="Search for a text pattern in files under a directory.",
        params=[
            ToolParam("pattern", "string", "Text or regex pattern to search for."),
            ToolParam("directory", "string", "Directory to search in. Defaults to current directory.", required=False),
        ],
    ),
    ToolDef(
        name="get_datetime",
        description="Return the current date and time in ISO 8601 format.",
        params=[],
    ),
]


class ToolRegistry:
    """Registry of available tools for LLM function calling."""

    def __init__(self, tools: list[ToolDef] | None = None):
        self._tools: dict[str, ToolDef] = {}
        for tool in (tools if tools is not None else CORE_TOOLS):
            self.register(tool)

    def register(self, tool: ToolDef) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDef | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def all(self) -> list[ToolDef]:
        return list(self._tools.values())

    def to_json_schema(self) -> list[dict[str, Any]]:
        """Export all tools as a list of JSON Schema function definitions."""
        return [t.to_json_schema() for t in self._tools.values()]

    def to_gbnf_grammar(self) -> str:
        """Generate a GBNF grammar that constrains output to valid tool calls.

        The grammar matches Hermes-style XML tool calls:
          <tool_call>{"name": "<tool_name>", "arguments": {...}}</tool_call>
        """
        tool_name_alts = " | ".join(f'"\\"{ name }\\""' for name in self._tools)
        if not tool_name_alts:
            tool_name_alts = '"\\"unknown\\""'

        lines = [
            'root ::= "<tool_call>" ws call-obj ws "</tool_call>"',
            f"tool-name-val ::= {tool_name_alts}",
            'call-obj ::= "{" ws "\\"name\\"" ws ":" ws tool-name-val ws "," ws "\\"arguments\\"" ws ":" ws json-obj ws "}"',
            'json-obj ::= "{" ws json-obj-inner? ws "}"',
            'json-obj-inner ::= json-kv (ws "," ws json-kv)*',
            'json-kv ::= json-string ws ":" ws json-value',
            'json-value ::= json-string | json-number | "true" | "false" | "null" | json-obj | json-array',
            'json-array ::= "[" ws (json-value (ws "," ws json-value)*)? ws "]"',
            r'json-string ::= "\"" ([^"\\] | "\\" .)* "\""',
            'json-number ::= "-"? [0-9]+ ("." [0-9]+)? ([eE] [-+]? [0-9]+)?',
            'ws ::= [ \t\n\r]*',
        ]
        return "\n".join(lines)

    def system_prompt_block(self) -> str:
        """Return a system prompt snippet that injects tool definitions."""
        schema_json = json.dumps(self.to_json_schema(), indent=2)
        return (
            "You have access to the following tools. "
            "To call a tool, respond with:\n"
            '<tool_call>{"name": "<tool_name>", "arguments": {<args>}}</tool_call>\n\n'
            f"Available tools:\n{schema_json}"
        )
