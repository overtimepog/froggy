# froggy 🐸

A tiny agent harness that turns any LLM into a tool-using assistant. Point it at a local model, an Ollama server, or any of 300+ cloud models on OpenRouter — froggy handles context management, tool calling, and the agent loop.

```bash
# Install
curl -fsSL https://raw.githubusercontent.com/overtimepog/froggy/main/install.sh | bash

# Add a model and start chatting
froggy add openai/gpt-4o
froggy
```

## What It Does

You give froggy a model. Froggy gives it tools, context, and the ability to act.

- **Any model, any source** — local (Transformers, MLX, llama.cpp), Ollama, or 300+ cloud models via OpenRouter
- **Agent behavior** — the model decides when to use tools and when to just answer. No manual tool orchestration.
- **Context engineering** — inject files, manage token budgets, auto-summarize long conversations, switch between context profiles
- **13 tool call formats** — works with every major model family (Hermes, Claude, Gemini, Mistral, Llama, DeepSeek, Qwen, Cohere, Functionary, and more)
- **Tool safety** — 3-tier model (auto-approve / confirm / blocked) with OS-level sandboxing on macOS

## Quick Start

```bash
# One-line install (auto-detects your hardware)
curl -fsSL https://raw.githubusercontent.com/overtimepog/froggy/main/install.sh | bash

# Or manual install
pip install .                    # CPU only
pip install ".[gpu]"             # NVIDIA GPU
pip install ".[mlx]"             # Apple Silicon
pip install ".[tools]"           # Tool use (web search)
```

Set up an OpenRouter API key (optional — gives you access to 300+ cloud models):

```bash
froggy config set openrouter_api_key sk-or-v1-...
```

Build your model list:

```bash
froggy add openrouter/free           # Free auto-routing model
froggy add anthropic/claude-sonnet-4 # Claude on OpenRouter
froggy add ollama:llama3             # Local Ollama model
froggy add mlx-community/Llama-3-8B  # HuggingFace model
```

Start chatting:

```bash
froggy                               # Launch (bare command starts chat)
```

## Model Management

Froggy searches across platforms when you add a model — OpenRouter, HuggingFace, and Ollama — and lets you pick the exact variant you want.

```bash
# Add a model (searches all platforms, shows variants)
froggy add gpt-4o
froggy add llama
froggy add deepseek

# Any format works
froggy add openai/gpt-4o                                      # OpenRouter ID
froggy add https://openrouter.ai/models/anthropic/claude-3-opus # OpenRouter URL
froggy add mlx-community/Llama-3-8B-4bit                       # HuggingFace repo
froggy add https://huggingface.co/TheBloke/Mistral-7B-GGUF     # HuggingFace URL
froggy add ollama:codellama                                     # Ollama model

# Manage your saved list
froggy models                # List saved models
froggy models remove llama   # Remove by name (partial match)
froggy models clear          # Remove all

# Download HuggingFace models locally
froggy download mlx-community/Llama-3-8B-4bit
froggy download TheBloke/Mistral-7B --pick    # Interactive variant picker
```

Saved models appear first (with ⭐) when you launch froggy. When there are many models, froggy prompts you to search/filter instead of dumping a 300-item list.

## Chat Commands

Once you're in a chat session:

| Command | What it does |
|---------|-------------|
| `/help` | Show all commands |
| `/model` | Switch to a different model |
| `/info` | Session settings, context usage, tools |
| `/context` | Detailed context window status |
| `/inject <file>` | Load a file into context (model reads it without tools) |
| `/eject <file>` | Remove an injected file |
| `/profile <name>` | Switch context profile (`minimal` / `standard` / `full`) |
| `/fresh` | Reset history, keep injected files |
| `/system <prompt>` | Set or show the system prompt |
| `/temp <value>` | Set temperature (0.0–2.0) |
| `/tokens <value>` | Set max response tokens |
| `/tools` | List tools and their status |
| `/tools on\|off` | Enable or disable the tool system |
| `/autorun` | Toggle auto-approve for tool calls |
| `/clear` | Clear conversation history |
| `/quit` | Exit |

## Context Management

Froggy manages the context window so long conversations don't silently overflow.

**Token tracking** — every message is counted against the model's context limit. `/context` shows real-time usage.

**Context profiles** control how much gets injected into the system prompt:

| Profile | What's included | Best for |
|---------|----------------|----------|
| `minimal` | System prompt + tool defs only | Small models, fast responses |
| `standard` | + tool examples + injected files | Most use cases (default) |
| `full` | Everything, no compression | Complex tasks, large context models |

**Auto-summarization** — when conversations get long, older messages are compressed into a summary that's injected into context. The model keeps a condensed memory of the full conversation.

**Auto-trimming** — when even summarization isn't enough, the oldest messages are dropped with a `[Earlier conversation trimmed]` marker. The model knows context was lost.

**File injection** — `/inject README.md` loads a file directly into context. The model can answer questions about it without needing to call `read_file`. Useful for project context, coding standards, or reference docs.

## Tool System

Froggy includes 6 built-in tools. The model decides when to use them based on your request — you don't need to tell it which tool to call.

| Tool | What it does | Safety |
|------|-------------|--------|
| `read_file` | Read file contents | Auto-approve |
| `write_file` | Write or create a file | Confirm |
| `edit_file` | Replace text in a file | Confirm |
| `run_shell` | Execute a shell command | Confirm / Blocked |
| `web_search` | Search the web | Auto-approve |
| `python_eval` | Evaluate Python code | Confirm |

**Safety model:**
- **Auto-approve** — low-risk operations run without prompting
- **Confirm** — prompts for user approval before running
- **Blocked** — destructive commands (rm, sudo, etc.) are never executed

**Custom tools** — place `.py` files in a `tools/` directory. Export `TOOL` or `TOOLS`:

```python
from froggy.tools import ToolDef, ToolParam

TOOL = ToolDef(
    name="list_todos",
    description="Return the current TODO list from todo.txt",
    params=[],
)
```

### Tool Call Format Support

Froggy's parser handles 13 tool call formats, so it works regardless of which model family the OpenRouter free tier routes to:

| Format | Models |
|--------|--------|
| Hermes JSON | Nous Hermes, Qwen, most instruction-tuned |
| Claude XML | Claude (Bedrock/text mode) |
| Mistral `[TOOL_CALLS]` | Mistral Nemo, Mistral Large |
| Llama `<\|python_tag\|>` | Llama 3.x, Llama 4 |
| DeepSeek unicode tags | DeepSeek R1, V3 |
| Qwen/ChatML | Qwen 2.5+ |
| Gemini functionCall | Gemini variants |
| Functionary `>>>` | Functionary v3, MeetKai |
| Cohere `Action:` | Command R, Command R+ |
| Function XML | `<function=name>` style |
| Bare JSON | Any model outputting raw JSON |

## Supported Backends

| Backend | Models | Setup |
|---------|--------|-------|
| **OpenRouter** | 300+ cloud models (GPT, Claude, Gemini, etc.) | `froggy config set openrouter_api_key <key>` |
| **Ollama** | Any model on your Ollama server | `ollama serve` (auto-detected) |
| **Apple MLX** | MLX-format models on Apple Silicon | `pip install ".[mlx]"` |
| **llama.cpp** | GGUF models | `llama-cli` on PATH |
| **Transformers** | SafeTensors, PyTorch bins, LoRA | `pip install ".[gpu]"` |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FROGGY_HOME` | `~/.froggy` | Base directory for models, config, data |
| `FROGGY_TOOLS` | `0` | Enable tools at startup (`1`) |
| `FROGGY_AUTORUN` | `0` | Auto-approve tool calls (`1`) |
| `FROGGY_MAX_TOOL_ROUNDS` | `5` | Max tool-call rounds per message |
| `OPENROUTER_API_KEY` | — | OpenRouter API key (alt: config) |

## Project Structure

```
froggy/
  cli.py              # CLI commands and model selection
  session.py          # Chat session, agent loop, slash commands
  context.py          # Context engineering (injection, profiles, summarization, trimming)
  backends.py         # Inference backends (Transformers, MLX, llama.cpp, Ollama, OpenRouter)
  discovery.py        # Model discovery across local, Ollama, OpenRouter
  saved_models.py     # Saved model list, cross-platform variant search
  tools.py            # Tool registry and core tool definitions
  tool_parser.py      # Multi-format tool call parser (13 formats)
  tool_executor.py    # Tool executor with 3-tier safety model
  config.py           # YAML config persistence
  paths.py            # Path helpers (~/.froggy)
  download.py         # HuggingFace model downloader
  models.py           # Local model listing, removal, info
  llmfit.py           # Hardware-matched model recommendations
```

## Requirements

- Python 3.11+
- [click](https://click.palletsprojects.com/) and [rich](https://rich.readthedocs.io/) (installed automatically)

## Running Tests

```bash
pip install pytest
pytest
```

## License

MIT
