# froggy

A terminal-based chat tool for running local AI models. Supports HuggingFace Transformers (with LoRA adapters), GGUF models via llama.cpp, Apple MLX on Apple Silicon, Ollama, and cloud models via OpenRouter.

## Features

- **Auto-discovery** - Scans directories to find local models, LoRA adapters, and GGUF files
- **OpenRouter integration** - Access hundreds of cloud models (GPT-4o, Claude, Gemini, etc.) through a single API key
- **Context management** - Automatic token tracking, context window monitoring, and auto-trimming when conversations get long
- **Model management** - Download, list, inspect, and remove models from the command line
- **Hardware-matched recommendations** - Get model suggestions that fit your GPU/CPU/RAM via llmfit
- **Persistent configuration** - YAML-based settings for device, format, and host preferences
- **Ollama integration** - Auto-detects models from a running Ollama server and merges them into the selection menu
- **Streaming chat** - Real-time token streaming with rich markdown rendering, automatic thinking-block filtering, and end-of-turn detection
- **LoRA support** - Automatically detects and applies LoRA adapters, downloading base models as needed
- **Apple MLX** - Native acceleration on Apple Silicon Macs via mlx-lm, auto-detected when available
- **GPU acceleration** - Auto-detects CUDA and selects optimal dtype (bfloat16/float32)
- **In-session controls** - Switch models, adjust temperature, set system prompts, and more without restarting
- **Tool use** - LLM-driven function calling (read/write files, run shell commands, web search) with a 3-tier safety model and custom plugin support

## Installation

### One-line install (recommended)

```bash
curl -fsSL https://raw.githubusercontent.com/overtimepog/froggy/main/install.sh | bash
```

This auto-detects your platform (Apple Silicon → MLX, CUDA → GPU, otherwise CPU) and installs froggy with the right backends. Customize the install location with `FROGGY_INSTALL_DIR`:

```bash
FROGGY_INSTALL_DIR=~/my-froggy curl -fsSL https://raw.githubusercontent.com/overtimepog/froggy/main/install.sh | bash
```

### Manual install

```bash
# Clone the repo
git clone https://github.com/overtimepog/froggy.git
cd froggy

# Install (CPU only)
pip install .

# Install with GPU/Transformers support
pip install ".[gpu]"

# Install with Apple MLX support (Apple Silicon only)
pip install ".[mlx]"

# Install with tool-use support (includes duckduckgo_search)
pip install ".[tools]"
```

## Usage

```bash
# Start an interactive chat session
froggy chat

# Point to a specific models directory
froggy chat --models-dir /path/to/models

# Force CPU inference
froggy chat --device cpu

# Download a model from HuggingFace
froggy download mlx-community/Llama-3-8B-4bit

# List downloaded models
froggy list

# Show model details
froggy info Llama-3-8B-4bit

# Remove a model
froggy remove Llama-3-8B-4bit

# Get hardware-matched model recommendations
froggy recommend

# View or change configuration
froggy config
```

On launch, `froggy chat` scans for models and presents a selection menu. Pick a model and start chatting.

## Commands

### `froggy chat`

Start an interactive chat session with a local model.

```bash
froggy chat                              # Auto-discover models and pick one
froggy chat --models-dir /path/to/models # Scan a specific directory
froggy chat --device cpu                 # Force CPU inference
froggy chat --tools-dir ./my_tools       # Load custom tool plugins
```

### `froggy download`

Download a HuggingFace model to `~/.froggy/models/`.

```bash
# Download by repo ID
froggy download mlx-community/Llama-3-8B-4bit

# Download by full HuggingFace URL
froggy download https://huggingface.co/TheBloke/Mistral-7B-GGUF

# Download a specific format
froggy download TheBloke/Mistral-7B --format gguf

# Interactively pick from available variants
froggy download TheBloke/Mistral-7B --pick
```

| Option | Description |
|--------|-------------|
| `--format [auto\|mlx\|gguf\|safetensors]` | Model format to download (default: `auto`) |
| `--pick` | Interactively pick from available variants |

### `froggy list`

List downloaded models in `~/.froggy/models/`.

```bash
froggy list          # Pretty-printed table
froggy list --json   # Machine-readable JSON output
```

| Option | Description |
|--------|-------------|
| `--json` | Output as JSON |

### `froggy remove`

Remove a downloaded model.

```bash
froggy remove Llama-3-8B-4bit       # Prompts for confirmation
froggy remove Llama-3-8B-4bit -y    # Skip confirmation
```

| Option | Description |
|--------|-------------|
| `-y, --yes` | Skip confirmation prompt |

### `froggy info`

Show detailed information about a model (format, size, parameters, etc.).

```bash
froggy info Llama-3-8B-4bit
```

### `froggy recommend`

Recommend models that match your hardware (GPU, CPU, RAM) via [llmfit](https://github.com/jayminwest/llmfit).

```bash
froggy recommend                          # Default recommendations
froggy recommend --limit 5               # Show top 5
froggy recommend --use-case coding        # Filter by use case
froggy recommend --json                   # Machine-readable JSON
```

| Option | Description |
|--------|-------------|
| `--limit INTEGER` | Max number of recommendations |
| `--use-case TEXT` | Target use case (e.g. `coding`, `chat`) |
| `--json` | Output as JSON |

### `froggy config`

View or modify froggy configuration. Settings are stored in `~/.froggy/config.yaml`.

```bash
# Show all current settings
froggy config

# Get a specific setting
froggy config get device

# Set a value
froggy config set device mps
froggy config set format gguf
froggy config set ollama_host http://localhost:11434
```

**Available config keys:**

| Key | Description | Example values |
|-----|-------------|----------------|
| `device` | Inference device | `cpu`, `cuda`, `mps` |
| `format` | Preferred model format | `auto`, `mlx`, `gguf`, `safetensors` |
| `ollama_host` | Ollama server URL | `http://localhost:11434` |
| `openrouter_api_key` | OpenRouter API key for cloud models | `sk-or-v1-...` |

## Chat Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/model` | Switch to a different model |
| `/system [prompt]` | Set or show the system prompt |
| `/temp [value]` | Set or show temperature (0.0 - 2.0) |
| `/tokens [value]` | Set or show max output tokens |
| `/info` | Show current session settings |
| `/context` | Show context window usage, token stats, and auto-trim history |
| `/clear` | Clear conversation history |
| `/quit` | Exit froggy |
| `/tools` | List available tools and their active state |
| `/tools on\|off` | Enable or disable the tool system |
| `/tools add <name>` | Activate a specific tool for this session |
| `/tools remove <name>` | Deactivate a specific tool for this session |
| `/autorun` | Toggle auto-approve for tool calls (skip confirmation prompts) |

## Tool System

froggy includes a function-calling tool system that lets the model read files, run commands, and search the web. Tools use [Hermes XML format](https://huggingface.co/NousResearch/Hermes-2-Pro-Mistral-7B) for reliable detection across model families.

### Built-in Tools

| Tool | Description | Safety tier |
|------|-------------|-------------|
| `read_file` | Read a file's contents | Auto-approve |
| `write_file` | Write or create a file | Confirm |
| `edit_file` | Replace a string in a file | Confirm |
| `run_shell` | Execute a shell command | Confirm / Blocked |
| `web_search` | Search the web (requires `duckduckgo_search`) | Auto-approve |
| `python_eval` | Evaluate Python code | Confirm |

### Safety Model

Tools run under a 3-tier safety model:

- **Auto-approve** — low-risk operations run without prompting (read_file, web_search, safe shell commands)
- **Confirm** — medium/high-risk operations prompt for user approval before running
- **Blocked** — destructive commands (rm, sudo, curl, etc.) are never executed

On macOS, shell commands are wrapped in `sandbox-exec` to prevent filesystem writes outside `/tmp`.

### Custom Tool Plugins

Place `.py` files in a `tools/` directory (or pass `--tools-dir`) to add custom tools. Each file should export either:

- `TOOL`: a single `ToolDef` object
- `TOOLS`: a list of `ToolDef` objects

```python
# tools/my_tool.py
from froggy.tools import ToolDef, ToolParam

TOOL = ToolDef(
    name="list_todos",
    description="Return the current TODO list from todo.txt",
    params=[],
)
```

Files starting with `_` are ignored. Broken files are skipped with a warning rather than crashing froggy.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FROGGY_HOME` | `~/.froggy` | Base directory for models, config, and data |
| `FROGGY_TOOLS` | `0` | Set to `1` to enable tools at startup |
| `FROGGY_AUTORUN` | `0` | Set to `1` to auto-approve all tool calls |
| `FROGGY_PROJECT_ROOT` | `""` | Project root used to locate the `tools/` plugin directory |
| `FROGGY_MAX_TOOL_ROUNDS` | `5` | Maximum tool-call/response rounds per user message |
| `OPENROUTER_API_KEY` | `""` | OpenRouter API key (alternative to `froggy config set openrouter_api_key`) |

```bash
# Use a custom home directory
FROGGY_HOME=~/my-models froggy chat

# Enable tools and autorun in one command
FROGGY_TOOLS=1 FROGGY_AUTORUN=1 froggy chat
```

## Supported Backends

| Backend | Status | Formats |
|---------|--------|---------|
| HuggingFace Transformers | Working | SafeTensors, PyTorch bins, LoRA adapters |
| Apple MLX | Working | SafeTensors (Apple Silicon only, requires `mlx-lm`) |
| llama.cpp | Working | GGUF (requires `llama-cli` on PATH) |
| Ollama | Working | Any model available on your Ollama server |
| OpenRouter | Working | Any model on [OpenRouter](https://openrouter.ai/models) (cloud API, requires API key) |

### MLX Setup (Apple Silicon)

On Apple Silicon Macs, froggy automatically uses MLX when `mlx-lm` is installed — no configuration needed. MLX models from HuggingFace (e.g., from [mlx-community](https://huggingface.co/mlx-community)) work out of the box:

```bash
# Install MLX support
pip install ".[mlx]"

# Download an MLX model
froggy download mlx-community/Llama-3-8B-4bit

# Launch froggy — MLX backend is auto-selected
froggy chat
```

### Ollama Setup

If you have [Ollama](https://ollama.com) running, froggy will automatically discover its models:

```bash
# Start Ollama (if not already running)
ollama serve

# Pull a model
ollama pull llama3

# Launch froggy — Ollama models appear automatically
froggy chat
```

### OpenRouter Setup (Cloud Models)

[OpenRouter](https://openrouter.ai) gives you access to hundreds of cloud models (GPT-4o, Claude, Gemini, LLaMA, Mistral, and more) through a single API key — including free models.

1. Get an API key at [openrouter.ai/keys](https://openrouter.ai/keys)
2. Set it in froggy:

```bash
# Option A: save to froggy config (persistent)
froggy config set openrouter_api_key sk-or-v1-...

# Option B: set as environment variable
export OPENROUTER_API_KEY=sk-or-v1-...
```

3. Launch froggy — OpenRouter models appear in the selection menu:

```bash
froggy chat
```

OpenRouter models show up tagged with `[OpenRouter]` and their context window size. Free models have `:free` in their name (e.g. `openrouter/free`, `qwen/qwen3-next-80b-a3b-instruct:free`).

## Context Management

froggy automatically tracks token usage and manages the context window so long conversations don't silently overflow:

- **Token tracking** — every message is counted against the model's context limit
- **Auto-trimming** — when the conversation approaches the context limit, the oldest messages are dropped to make room, with a `[Earlier conversation trimmed]` marker so the model knows context was lost
- **Response reserve** — 15% of the context window is reserved for the model's response, so you won't hit the limit mid-generation
- **Model-aware limits** — context limits are detected automatically for known model families (GPT-4o: 128K, Claude: 200K, LLaMA: 8K, etc.) and from OpenRouter's API metadata

Use the `/context` command during a chat session to see current usage:

```
/context
╭──────────────── Context Window ─────────────────╮
│ Context limit           200,000 tokens          │
│ Available (after reserve) 170,000 tokens        │
│ Current usage           1,234 tokens            │
│ Utilization             1%                      │
│ Messages                12                      │
│ Auto-trims              0                       │
│ Session usage           prompt: 8,432  ...      │
╰─────────────────────────────────────────────────╯
```

The `/info` command also shows a compact context summary alongside other session settings.

## Project Structure

```
froggy/
  __init__.py         # Package init
  __main__.py         # Entry point (python -m froggy)
  cli.py              # CLI interface and model selection
  config.py           # YAML config persistence (load/save/get/set)
  paths.py            # Path helpers (~/.froggy resolution, FROGGY_HOME)
  download.py         # HuggingFace model downloader
  models.py           # Model listing, removal, and info
  llmfit.py           # Hardware-matched model recommendations
  backends.py         # Inference backends (Transformers, MLX, llama.cpp, Ollama)
  discovery.py        # Local + Ollama model discovery and validation
  session.py          # Chat session, tool loop, and command handling
  tools.py            # Tool registry and core tool definitions
  tool_parser.py      # Streaming-aware tool-call parser (Hermes XML + JSON)
  tool_executor.py    # Tool executor with 3-tier safety model
  tool_selector.py    # Tool filtering helper
tests/
  test_backends.py          # Backend selection and loading tests
  test_cli_config.py        # Config CLI command tests
  test_cli_group.py         # CLI group structure tests
  test_cli_management.py    # Model management CLI tests (download/list/remove/info/recommend)
  test_commands.py          # Chat command parsing tests
  test_config.py            # Config module unit tests
  test_custom_tools.py      # Custom tool plugin loader tests
  test_discovery.py         # Model discovery tests
  test_download.py          # Download module tests
  test_llmfit.py            # Recommendation engine tests
  test_mlx.py               # MLX backend and platform detection tests
  test_models.py            # Model list/remove/info tests
  test_ollama.py            # Ollama backend and discovery tests
  test_paths.py             # Path helper tests
  test_streaming.py         # Thinking filter and stop-string tests
  test_tool_executor.py     # Tool executor tests
  test_tool_integration.py  # Tool loop integration tests
  test_tool_parser.py       # Tool parser tests
  test_tool_safety.py       # Tool safety model tests
  test_tools.py             # Tool definition tests
tools/
  (place custom tool plugins here)
```

## Requirements

- Python 3.11+
- [click](https://click.palletsprojects.com/) and [rich](https://rich.readthedocs.io/) (installed automatically)
- [pyyaml](https://pyyaml.org/) (installed automatically — used for config persistence)
- [huggingface-hub](https://huggingface.co/docs/huggingface_hub/) (installed automatically — used for model downloads)
- For GPU inference: PyTorch, Transformers, Accelerate, PEFT
- For Apple Silicon: mlx, mlx-lm
- For tool use with web search: duckduckgo_search (`pip install ".[tools]"`)

## Running Tests

```bash
pip install pytest
pytest
```

## License

MIT
