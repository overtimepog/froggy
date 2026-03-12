# froggy

A terminal-based chat tool for running local AI models. Supports HuggingFace Transformers (with LoRA adapters) and GGUF models, with more backends planned.

## Features

- **Auto-discovery** - Scans directories to find local models, LoRA adapters, and GGUF files
- **Streaming chat** - Real-time token streaming with rich markdown rendering
- **LoRA support** - Automatically detects and applies LoRA adapters, downloading base models as needed
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
| llama.cpp | Planned | GGUF |

## Project Structure

```
froggy/
  __init__.py       # Package init
  __main__.py       # Entry point (python -m froggy)
  cli.py            # CLI interface and model selection
  backends.py       # Inference backends (Transformers, llama.cpp)
  discovery.py      # Local model discovery and validation
  session.py        # Chat session and command handling
tests/
  test_backends.py  # Backend selection and loading tests
  test_commands.py  # Chat command parsing tests
  test_discovery.py # Model discovery tests
```

## Requirements

- Python 3.11+
- [click](https://click.palletsprojects.com/) and [rich](https://rich.readthedocs.io/) (installed automatically)
- For GPU inference: PyTorch, Transformers, Accelerate, PEFT, huggingface_hub

## Running Tests

```bash
pip install pytest
pytest
```

## License

MIT
