# froggy

A terminal-based chat tool for running local AI models. Supports HuggingFace Transformers (with LoRA adapters), GGUF models via llama.cpp, Apple MLX on Apple Silicon, and Ollama.

## Features

- **Auto-discovery** - Scans directories to find local models, LoRA adapters, and GGUF files
- **Ollama integration** - Auto-detects models from a running Ollama server and merges them into the selection menu
- **Streaming chat** - Real-time token streaming with rich markdown rendering, automatic thinking-block filtering, and end-of-turn detection
- **LoRA support** - Automatically detects and applies LoRA adapters, downloading base models as needed
- **Apple MLX** - Native acceleration on Apple Silicon Macs via mlx-lm, auto-detected when available
- **GPU acceleration** - Auto-detects CUDA and selects optimal dtype (bfloat16/float32)
- **In-session controls** - Switch models, adjust temperature, set system prompts, and more without restarting

## Installation

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
```

## Usage

```bash
# Start with auto-discovered models
froggy

# Point to a specific models directory
froggy --models-dir /path/to/models

# Force CPU inference
froggy --device cpu
```

On launch, froggy scans for models and presents a selection menu. Pick a model and start chatting.

## Chat Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/model` | Switch to a different model |
| `/system [prompt]` | Set or show the system prompt |
| `/temp [value]` | Set or show temperature (0.0 - 2.0) |
| `/tokens [value]` | Set or show max output tokens |
| `/info` | Show current session settings |
| `/clear` | Clear conversation history |
| `/quit` | Exit froggy |

## Supported Backends

| Backend | Status | Formats |
|---------|--------|---------|
| HuggingFace Transformers | Working | SafeTensors, PyTorch bins, LoRA adapters |
| Apple MLX | Working | SafeTensors (Apple Silicon only, requires `mlx-lm`) |
| llama.cpp | Working | GGUF (requires `llama-cli` on PATH) |
| Ollama | Working | Any model available on your Ollama server |

### MLX Setup (Apple Silicon)

On Apple Silicon Macs, froggy automatically uses MLX when `mlx-lm` is installed — no configuration needed. MLX models from HuggingFace (e.g., from [mlx-community](https://huggingface.co/mlx-community)) work out of the box:

```bash
# Install MLX support
pip install ".[mlx]"

# Download an MLX model (example)
huggingface-cli download mlx-community/Llama-3-8B-4bit --local-dir models/Llama-3-8B-4bit

# Launch froggy — MLX backend is auto-selected
froggy --models-dir models/
```

### Ollama Setup

If you have [Ollama](https://ollama.com) running, froggy will automatically discover its models:

```bash
# Start Ollama (if not already running)
ollama serve

# Pull a model
ollama pull llama3

# Launch froggy — Ollama models appear automatically
froggy
```

## Project Structure

```
froggy/
  __init__.py       # Package init
  __main__.py       # Entry point (python -m froggy)
  cli.py            # CLI interface and model selection
  backends.py       # Inference backends (Transformers, MLX, llama.cpp, Ollama)
  discovery.py      # Local + Ollama model discovery and validation
  session.py        # Chat session and command handling
tests/
  test_backends.py  # Backend selection and loading tests
  test_commands.py  # Chat command parsing tests
  test_discovery.py # Model discovery tests
  test_ollama.py    # Ollama backend and discovery tests
  test_mlx.py       # MLX backend and platform detection tests
  test_streaming.py # Thinking filter and stop-string tests
```

## Requirements

- Python 3.11+
- [click](https://click.palletsprojects.com/) and [rich](https://rich.readthedocs.io/) (installed automatically)
- For GPU inference: PyTorch, Transformers, Accelerate, PEFT, huggingface_hub
- For Apple Silicon: mlx, mlx-lm

## Running Tests

```bash
pip install pytest
pytest
```

## License

MIT
