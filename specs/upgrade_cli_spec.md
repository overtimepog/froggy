# Froggy CLI Upgrade Spec: `froggy download` + Managed Model Directory + llmfit Integration

## Overview

Transform froggy from a "point at a directory" chat tool into an **ollama-style managed CLI** with:

1. A **dedicated, well-known model directory** (`~/.froggy/models/`)
2. A `froggy download` command for pulling models from HuggingFace
3. **llmfit** integration for hardware-aware model recommendations
4. CLI subcommands for model lifecycle management (`list`, `remove`, `recommend`)

After this upgrade, froggy works out of the box — no `--models-dir` flag needed.

---

## 1. Managed Model Directory

### 1.1 Directory Structure

```
~/.froggy/
├── config.yaml              # Global froggy configuration
└── models/
    ├── mistral-7b-instruct/
    │   ├── config.json
    │   ├── model.safetensors
    │   ├── tokenizer.json
    │   └── ...
    ├── llama-3-8b-4bit/
    │   ├── config.json
    │   ├── model.safetensors      # MLX 4-bit quantized weights
    │   ├── tokenizer.json
    │   └── ...
    ├── llama-3-8b-Q4_K_M/
    │   └── llama-3-8b.Q4_K_M.gguf
    └── phi-3-mini/
        └── ...
```

### 1.2 Path Resolution

| Platform | Default Path | Override |
|----------|-------------|----------|
| macOS    | `~/.froggy/models/` | `FROGGY_HOME` env var |
| Linux    | `~/.froggy/models/` | `FROGGY_HOME` env var |
| Windows  | `%USERPROFILE%\.froggy\models\` | `FROGGY_HOME` env var |

Resolution order (first match wins):
1. `FROGGY_HOME` environment variable → `$FROGGY_HOME/models/`
2. `--models-dir` CLI flag (backwards-compatible, overrides default)
3. `~/.froggy/models/` (default)

### 1.3 First-Run Behavior

On first invocation of any froggy command:
1. Check if `~/.froggy/` exists
2. If not, create `~/.froggy/` and `~/.froggy/models/`
3. Print: `Created froggy home at ~/.froggy/`
4. Optionally create `~/.froggy/config.yaml` with defaults

Implementation: a `ensure_froggy_home()` utility called at CLI entry.

```python
# froggy/paths.py
import os
from pathlib import Path

def froggy_home() -> Path:
    """Return the froggy home directory, respecting FROGGY_HOME env var."""
    env = os.environ.get("FROGGY_HOME")
    if env:
        return Path(env)
    return Path.home() / ".froggy"

def models_dir() -> Path:
    """Return the default models directory."""
    return froggy_home() / "models"

def ensure_froggy_home() -> Path:
    """Create froggy home + models dir if they don't exist. Returns home path."""
    home = froggy_home()
    mdir = home / "models"
    mdir.mkdir(parents=True, exist_ok=True)
    return home
```

---

## 2. CLI Restructure: Click Groups

Currently froggy uses a single `@click.command()`. This upgrade converts to `@click.group()` with subcommands.

### 2.1 Command Tree

```
froggy                          # (no subcommand) → launch interactive chat (backwards-compatible)
froggy chat                     # Explicit chat mode (same as bare `froggy`)
froggy download <source>        # Download a model from HuggingFace
froggy list                     # List downloaded models
froggy remove <model-name>      # Delete a downloaded model
froggy recommend                # Hardware-aware model recommendations (via llmfit)
froggy info <model-name>        # Show details about a downloaded model
froggy config                   # View/edit global config
```

---

## 3. `froggy download` Command

### 3.1 Usage

```bash
# Smart download — auto-detects best format AND best quantization for your hardware
# On macOS Apple Silicon: downloads MLX version if available, falls back to GGUF
# On Linux/Windows with GPU: downloads GGUF with best quantization for your VRAM
# On Linux/Windows CPU-only: downloads GGUF with best quantization for your RAM
# If multiple variants exist, auto-picks the best one — no flags needed
froggy download meta-llama/Llama-3-8B-Instruct

# Download from a full HuggingFace URL
froggy download https://huggingface.co/meta-llama/Llama-3-8B-Instruct

# Download a specific file from a repo
froggy download TheBloke/Llama-2-7B-GGUF --file llama-2-7b.Q4_K_M.gguf

# Explicitly request a format (overrides auto-detection)
froggy download meta-llama/Llama-3-8B-Instruct --format mlx
froggy download meta-llama/Llama-3-8B-Instruct --format gguf
froggy download meta-llama/Llama-3-8B-Instruct --format safetensors

# Pick interactively instead of auto-selecting (browse all available variants)
froggy download meta-llama/Llama-3-8B-Instruct --pick

# Download to a custom name
froggy download meta-llama/Llama-3-8B-Instruct --name my-llama

# Force re-download
froggy download meta-llama/Llama-3-8B-Instruct --force
```

### 3.2 Source Parsing

The `<source>` argument accepts:
- **Repo ID**: `org/model-name` → passed directly to `huggingface_hub`
- **Full URL**: `https://huggingface.co/org/model-name` → extract `org/model-name`
- **URL with tree/blob path**: `https://huggingface.co/org/model/blob/main/file.gguf` → extract repo + filename

Parsing logic:

```python
import re
from dataclasses import dataclass

@dataclass
class ParsedSource:
    repo_id: str
    filename: str | None = None
    revision: str | None = None

def parse_source(source: str) -> ParsedSource:
    """Parse a HuggingFace repo ID or URL into components."""
    # Full URL pattern
    hf_url = re.match(
        r"https?://huggingface\.co/([^/]+/[^/]+)(?:/(?:blob|tree|resolve)/([^/]+)(?:/(.+))?)?",
        source,
    )
    if hf_url:
        repo_id = hf_url.group(1)
        revision = hf_url.group(2)  # e.g. "main"
        filename = hf_url.group(3)  # e.g. "model.Q4_K_M.gguf"
        return ParsedSource(repo_id, filename, revision if revision != "main" else None)

    # Bare repo ID: org/model
    if "/" in source and not source.startswith(("http://", "https://")):
        return ParsedSource(repo_id=source)

    raise click.BadParameter(f"Cannot parse source: {source}")
```

### 3.2.1 Platform Detection

```python
import platform
import sys

def detect_platform() -> str:
    """Detect the best model format for the current platform.

    Returns one of: 'mlx', 'gguf', 'safetensors'
    """
    if sys.platform == "darwin":
        # Check for Apple Silicon (arm64)
        if platform.machine() == "arm64":
            return "mlx"
        # Intel Mac — no MLX support, use GGUF
        return "gguf"
    # Linux / Windows → GGUF (works with llama.cpp on CPU or CUDA GPU)
    return "gguf"
```

### 3.2.2 MLX Repo Discovery

Many MLX-quantized models live under the `mlx-community` org on HuggingFace (e.g., `mlx-community/Llama-3-8B-Instruct-4bit`). When the user provides a base repo like `meta-llama/Llama-3-8B-Instruct` and the resolved format is `mlx`, froggy searches for an MLX variant:

```python
from huggingface_hub import HfApi

# Quantization preferences ordered best-to-worst for typical hardware
MLX_QUANT_PREFERENCES = ["4bit", "8bit", "bf16", "fp16"]

def find_mlx_repo(base_repo_id: str, preferred_quant: str | None = None) -> str | None:
    """Search for an MLX-quantized version of a model on HuggingFace.

    Checks mlx-community/<model-name>-<quant> variants.
    Returns the repo_id if found, or None.
    """
    api = HfApi()
    model_name = base_repo_id.split("/")[-1]

    # If user specified a quant, try just that
    if preferred_quant:
        candidates = [f"mlx-community/{model_name}-{preferred_quant}"]
    else:
        candidates = [f"mlx-community/{model_name}-{q}" for q in MLX_QUANT_PREFERENCES]

    for candidate in candidates:
        try:
            api.model_info(candidate)
            return candidate
        except Exception:
            continue

    return None
```

### 3.3 Download Flows

The default behavior (`froggy download <source>` with no flags) runs **Flow Auto**, which detects the platform and picks the best format + quantization automatically.

#### Flow Auto: Platform-Aware Smart Download (default)

This is the primary flow — triggered when the user runs `froggy download <source>` without `--format` or `--file`.

```
1. Detect platform via detect_platform() → "mlx" | "gguf" | "safetensors"
2. If format is "mlx":
   a. Search for MLX repo via find_mlx_repo(repo_id)
   b. If found → download the MLX repo (Flow MLX)
   c. If not found → print warning, fall back to GGUF (Flow GGUF Auto)
   d. If GGUF also unavailable → try full safetensors repo (Flow Safetensors)
   e. If nothing available → error with clear message
3. If format is "gguf":
   a. Search for GGUF files in repo or known GGUF re-quantization repos
   b. If found → pick best quantization for hardware (Flow GGUF Auto)
   c. If not found → try full safetensors repo (Flow Safetensors)
   d. If nothing available → error with clear message
4. If format is "safetensors":
   a. Download the full repo (Flow Safetensors)
```

User-facing output during auto-detection:

```
$ froggy download meta-llama/Llama-3-8B-Instruct
Detected: macOS Apple Silicon — looking for MLX version...
Found: mlx-community/Llama-3-8B-Instruct-4bit
Downloading... ████████████████████████████ 100%  4.3 GB
Saved to ~/.froggy/models/Llama-3-8B-Instruct-4bit/
```

```
$ froggy download some-org/Obscure-Model-7B
Detected: macOS Apple Silicon — looking for MLX version...
No MLX version found. Falling back to GGUF...
No GGUF files found either.
Error: Could not find a compatible download for some-org/Obscure-Model-7B.
Try: froggy download some-org/Obscure-Model-7B --format safetensors
```

#### Flow MLX: MLX Model Repository

Downloads an MLX-quantized model repo (typically from `mlx-community`).

```python
from huggingface_hub import snapshot_download

dest = models_dir() / model_name
snapshot_download(
    repo_id=mlx_repo_id,  # e.g. "mlx-community/Llama-3-8B-Instruct-4bit"
    local_dir=str(dest),
    ignore_patterns=["*.md", ".gitattributes", "*.msgpack", "*.h5", "*.gguf"],
    revision=revision,
)
```

#### Flow Safetensors: Full Model Repository (safetensors / transformers)

```python
from huggingface_hub import snapshot_download

dest = models_dir() / model_name
snapshot_download(
    repo_id=repo_id,
    local_dir=str(dest),
    ignore_patterns=["*.md", ".gitattributes", "*.msgpack", "*.h5", "*.gguf"],
    revision=revision,
)
```

#### Flow GGUF Single: Specific GGUF File (via `--file`)

```python
from huggingface_hub import hf_hub_download

dest_dir = models_dir() / model_name
dest_dir.mkdir(parents=True, exist_ok=True)
hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    local_dir=str(dest_dir),
)
```

#### Flow GGUF Auto: Hardware-Aware GGUF Quantization Selection

Used when format resolves to `gguf` (either explicitly or as fallback from MLX).

1. List all `.gguf` files in the repo via `huggingface_hub.list_repo_files()`
2. Parse quantization levels from filenames (Q2_K, Q3_K_S, Q4_K_M, Q5_K_M, Q6_K, Q8_0, F16)
3. Query system RAM/VRAM (or use llmfit data)
4. Select the best quantization that fits in available memory
5. Download that single file

#### Flow Pick: Interactive Selection (via `--pick`)

Works for any format — shows all available variants and lets the user choose.

1. Detect available formats for the model (MLX repos, GGUF files, safetensors)
2. Display a Rich table with: variant name, format, quantization, and size
3. User picks from the list
4. Download the selected variant

### 3.4 Download Features

| Feature | Implementation |
|---------|---------------|
| **Progress bar** | Built-in from `huggingface_hub` (tqdm-based) |
| **Resume** | Built-in from `huggingface_hub` (automatic) |
| **Duplicate check** | Before download, check if `models_dir() / name` exists; skip unless `--force` |
| **Incomplete download cleanup** | On KeyboardInterrupt, delete partial directory |
| **Auth** | Respects `HF_TOKEN` env var or `huggingface-cli login` |
| **Private repos** | Supported via HF token authentication |

### 3.5 Click Command Definition

```python
@cli.command()
@click.argument("source")
@click.option("--file", "filename", default=None, help="Download a specific file from the repo.")
@click.option("--format", "fmt", type=click.Choice(["auto", "mlx", "gguf", "safetensors"]), default="auto",
              help="Model format to download. 'auto' detects platform (MLX on Apple Silicon, GGUF otherwise).")
@click.option("--pick", is_flag=True, help="Interactively pick from available variants.")
@click.option("--name", default=None, help="Custom local name for the downloaded model.")
@click.option("--force", is_flag=True, help="Re-download even if model already exists.")
def download(source: str, filename: str | None, fmt: str, pick: bool, name: str | None, force: bool):
    """Download a model from HuggingFace Hub."""
    ...
```

---

## 4. `froggy list` Command

### 4.1 Usage

```bash
froggy list            # Table of all downloaded models
froggy list --json     # JSON output for scripting
```

### 4.2 Output

```
┌──────────────────────────────┬───────────────┬──────────┬────────────┐
│ Name                         │ Format        │ Size     │ Downloaded │
├──────────────────────────────┼───────────────┼──────────┼────────────┤
│ Llama-3-8B-Instruct-4bit     │ MLX (4-bit)   │ 4.3 GB   │ 2026-03-15 │
│ Llama-3-8B-Instruct          │ safetensors   │ 15.2 GB  │ 2026-03-14 │
│ llama-2-7b-Q4_K_M            │ GGUF          │ 3.8 GB   │ 2026-03-16 │
│ phi-3-mini                   │ safetensors   │ 7.6 GB   │ 2026-03-10 │
└──────────────────────────────┴───────────────┴──────────┴────────────┘
```

### 4.3 Implementation

Reuse existing `discover_models()` from `discovery.py`, pointed at `~/.froggy/models/`. Add size calculation (sum of files in each model dir) and download date (directory mtime).

```python
@cli.command("list")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def list_models(as_json: bool):
    """List all downloaded models."""
    ...
```

---

## 5. `froggy remove` Command

### 5.1 Usage

```bash
froggy remove llama-2-7b-Q4_K_M
froggy remove llama-2-7b-Q4_K_M --yes  # Skip confirmation
```

### 5.2 Behavior

1. Resolve model name to path in `~/.froggy/models/`
2. Confirm with user (unless `--yes`)
3. `shutil.rmtree()` the model directory
4. Print confirmation with freed disk space

```python
@cli.command()
@click.argument("model_name")
@click.option("--yes", is_flag=True, help="Skip confirmation prompt.")
def remove(model_name: str, yes: bool):
    """Remove a downloaded model."""
    ...
```

---

## 6. `froggy recommend` — llmfit Integration

### 6.1 Overview

Integrate [llmfit](https://github.com/AlexsJones/llmfit) to provide hardware-aware model recommendations. llmfit is a Rust CLI tool (9.1k stars) that detects RAM/CPU/GPU and ranks 206+ models by a composite fit score.

### 6.2 Integration Strategy

llmfit is a Rust binary, not a Python library. Integration options:

**Option A: Bundled binary (recommended for UX)**
- Ship pre-built llmfit binaries for macOS (arm64, x86_64) and Linux (x86_64) in the froggy package or download on first use
- Store at `~/.froggy/bin/llmfit`
- Call via subprocess with `--json` flag for structured output

**Option B: Require user install (simpler, less UX)**
- Document: "Install llmfit from https://github.com/AlexsJones/llmfit/releases"
- Detect if `llmfit` is on PATH; gracefully degrade if missing

**Recommended: Option A with fallback to Option B.**

### 6.3 llmfit Binary Management (Install + Auto-Update)

On first use, froggy downloads llmfit automatically. On subsequent runs, it checks for newer versions (at most once per day) and updates if available.

```
# First run — installs llmfit
$ froggy recommend
llmfit not found. Downloading llmfit v0.5.5...
Downloading llmfit-v0.5.5-aarch64-apple-darwin.tar.gz... done
Installed to ~/.froggy/bin/llmfit
```

```
# Later run — auto-update check
$ froggy recommend
Updating llmfit v0.5.5 → v0.6.0...
Downloaded llmfit-v0.6.0-aarch64-apple-darwin.tar.gz... done
Updated ~/.froggy/bin/llmfit
```

**Auto-update logic:**

1. Store the installed version and last-check timestamp in `~/.froggy/config.yaml` under `llmfit.version` and `llmfit.last_update_check`
2. On each invocation of `froggy recommend` or `froggy download` (which uses llmfit for hardware info):
   - If `last_update_check` is older than 24 hours, query the GitHub Releases API for the latest llmfit release
   - If a newer version is available, download and replace the binary
   - Update `llmfit.version` and `llmfit.last_update_check`
3. Auto-update can be disabled via `froggy config set llmfit.auto_update false`
4. If the update fails (network error, etc.), log a warning and continue with the existing binary

```python
import time
from datetime import datetime, timedelta

UPDATE_CHECK_INTERVAL = timedelta(hours=24)
LLMFIT_RELEASES_URL = "https://api.github.com/repos/AlexsJones/llmfit/releases/latest"

def check_llmfit_update(config: dict) -> str | None:
    """Check if a newer llmfit version is available. Returns new version tag or None."""
    last_check = config.get("llmfit", {}).get("last_update_check")
    if last_check:
        last_check_dt = datetime.fromisoformat(last_check)
        if datetime.now() - last_check_dt < UPDATE_CHECK_INTERVAL:
            return None  # Too soon to check again

    import urllib.request, json
    try:
        resp = urllib.request.urlopen(LLMFIT_RELEASES_URL, timeout=5)
        data = json.loads(resp.read())
        latest = data["tag_name"]
        current = config.get("llmfit", {}).get("version", "")
        if latest != current:
            return latest
    except Exception:
        pass  # Network error — skip update
    return None
```

```
# First run — installs llmfit
$ froggy recommend
llmfit not found. Downloading llmfit v0.5.5...
Downloading llmfit-v0.5.5-aarch64-apple-darwin.tar.gz... done
Installed to ~/.froggy/bin/llmfit

Detecting hardware...
  RAM: 32 GB
  GPU: Apple M2 Pro (16 GB unified)
  CPU: 12 cores

Top 5 recommended models:
┌────┬─────────────────────────┬───────┬──────────┬────────────┬───────────┐
│ #  │ Model                   │ Score │ Speed    │ Quant      │ Run Mode  │
├────┼─────────────────────────┼───────┼──────────┼────────────┼───────────┤
│ 1  │ Llama 3 8B Instruct     │ 95    │ 42 t/s   │ Q4_K_M     │ GPU       │
│ 2  │ Mistral 7B Instruct     │ 93    │ 45 t/s   │ Q4_K_M     │ GPU       │
│ 3  │ Phi-3 Mini              │ 91    │ 55 t/s   │ Q5_K_M     │ GPU       │
│ 4  │ Qwen2 7B                │ 88    │ 40 t/s   │ Q4_K_M     │ GPU       │
│ 5  │ Gemma 2 9B              │ 85    │ 35 t/s   │ Q4_K_M     │ GPU       │
└────┴─────────────────────────┴───────┴──────────┴────────────┴───────────┘

Download a model? Enter # or 'n' to skip:
```

### 6.4 Subprocess Integration

```python
import json
import subprocess
from pathlib import Path

def get_llmfit_path() -> Path | None:
    """Find llmfit binary: bundled first, then PATH."""
    bundled = froggy_home() / "bin" / "llmfit"
    if bundled.is_file() and os.access(bundled, os.X_OK):
        return bundled
    # Check PATH
    from shutil import which
    found = which("llmfit")
    return Path(found) if found else None

def llmfit_recommend(limit: int = 10) -> list[dict]:
    """Get model recommendations from llmfit."""
    llmfit = get_llmfit_path()
    if not llmfit:
        raise FileNotFoundError("llmfit not installed")
    result = subprocess.run(
        [str(llmfit), "recommend", "--json", "--limit", str(limit)],
        capture_output=True, text=True, timeout=30,
    )
    result.check_returncode()
    return json.loads(result.stdout)

def llmfit_system_info() -> dict:
    """Get hardware info from llmfit."""
    llmfit = get_llmfit_path()
    if not llmfit:
        raise FileNotFoundError("llmfit not installed")
    result = subprocess.run(
        [str(llmfit), "system", "--json"],
        capture_output=True, text=True, timeout=10,
    )
    result.check_returncode()
    return json.loads(result.stdout)
```

### 6.5 `froggy recommend` Click Command

```python
@cli.command()
@click.option("--limit", "-n", default=10, help="Number of recommendations.")
@click.option("--use-case", type=str, default=None, help="Filter by use case (e.g., 'coding', 'chat').")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
@click.option("--download", is_flag=True, help="Prompt to download a recommended model.")
def recommend(limit: int, use_case: str | None, as_json: bool, download: bool):
    """Get hardware-aware model recommendations via llmfit."""
    ...
```

### 6.6 `--format auto` Integration with llmfit

When `froggy download repo/model` (or `--format auto`) runs, llmfit data enhances format and quantization selection:

**For MLX (macOS Apple Silicon):**
1. Call `llmfit_system_info()` to get unified memory
2. Use memory budget to pick MLX quantization level (4bit vs 8bit)
3. If llmfit unavailable, fall back to `os.sysconf` to read system RAM and use 70% as budget

**For GGUF (Linux/Windows/Intel Mac):**
1. Call `llmfit_system_info()` to get RAM and VRAM
2. List available GGUF files from the repo
3. Estimate model memory requirements per quantization level
4. Pick the highest quality quantization that fits
5. If llmfit unavailable, fall back to simple heuristic:
   - Get system RAM via `psutil.virtual_memory()` or `os.sysconf`
   - Use 70% of available RAM as the budget
   - Pick accordingly

---

## 7. `froggy info` Command

```bash
froggy info llama-3-8b-instruct
```

Output:
```
Model: Llama-3-8B-Instruct-4bit
Path:  ~/.froggy/models/Llama-3-8B-Instruct-4bit
Format: MLX (4-bit)
Size:   4.3 GB
Type:   LlamaForCausalLM
Architectures: ['LlamaForCausalLM']
Source: mlx-community/Llama-3-8B-Instruct-4bit
Files:  8
```

---

## 8. `froggy config` Command

### 8.1 Config File: `~/.froggy/config.yaml`

```yaml
# Default device for inference
device: auto

# Default models directory (override with FROGGY_HOME)
# models_dir: ~/.froggy/models

# Ollama integration
ollama:
  enabled: true
  base_url: http://localhost:11434

# HuggingFace settings
huggingface:
  default_format: auto  # auto | mlx | gguf | safetensors
  auth_token: null      # Or set HF_TOKEN env var

# llmfit settings
llmfit:
  auto_install: true    # Download llmfit binary on first `recommend`
  auto_update: true     # Check for newer versions (at most once per 24h)
  version: v0.5.5
  last_update_check: null  # ISO timestamp of last update check
```

### 8.2 Commands

```bash
froggy config              # Print current config
froggy config set device cuda:0
froggy config set ollama.enabled false
```

---

## 9. Updated CLI Entry Point

### 9.1 New `cli.py` Structure

```python
import click
from .paths import ensure_froggy_home, models_dir

@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """Froggy — chat with your local models."""
    ensure_froggy_home()
    if ctx.invoked_subcommand is None:
        ctx.invoke(chat)

@cli.command()
@click.option("--models-dir", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--device", type=str, default="auto")
@click.option("--tools-dir", type=click.Path(path_type=Path), default=None)
def chat(models_dir, device, tools_dir):
    """Interactive chat session (default command)."""
    # Current main() logic goes here
    # models_dir defaults to ~/.froggy/models/ if not specified
    ...

@cli.command()
@click.argument("source")
@click.option("--file", "filename", default=None)
@click.option("--format", "fmt", type=click.Choice(["auto", "mlx", "gguf", "safetensors"]), default="auto")
@click.option("--pick", is_flag=True)
@click.option("--name", default=None)
@click.option("--force", is_flag=True)
def download(source, filename, fmt, pick, name, force):
    """Download a model from HuggingFace Hub."""
    ...

@cli.command("list")
@click.option("--json", "as_json", is_flag=True)
def list_models(as_json):
    """List downloaded models."""
    ...

@cli.command()
@click.argument("model_name")
@click.option("--yes", is_flag=True)
def remove(model_name, yes):
    """Remove a downloaded model."""
    ...

@cli.command()
@click.option("--limit", "-n", default=10)
@click.option("--use-case", default=None)
@click.option("--json", "as_json", is_flag=True)
@click.option("--download", is_flag=True)
def recommend(limit, use_case, as_json, download):
    """Get hardware-aware model recommendations."""
    ...

@cli.command()
@click.argument("model_name")
def info(model_name):
    """Show details about a downloaded model."""
    ...
```

### 9.2 Entry Point Update

In `pyproject.toml`:
```toml
[project.scripts]
froggy = "froggy.cli:cli"   # was: froggy.cli:main
```

---

## 10. New Dependencies

### Required (add to `[project.dependencies]`)

```toml
dependencies = [
    "click>=8.0",
    "rich>=13.0",
    "huggingface-hub>=0.25.0",  # NEW — for download command
]
```

### Optional

```toml
[project.optional-dependencies]
gpu = ["torch", "transformers", "accelerate", "peft", "huggingface-hub"]
mlx = ["mlx>=0.16.0", "mlx-lm>=0.16.0"]
tools = ["duckduckgo_search>=6.0"]
# huggingface-hub is now in core deps, no longer only in [gpu]
```

---

## 11. New Files

| File | Purpose |
|------|---------|
| `froggy/paths.py` | `froggy_home()`, `models_dir()`, `ensure_froggy_home()` |
| `froggy/download.py` | Download logic: source parsing, platform detection, MLX repo discovery, HF API calls, GGUF selection |
| `froggy/llmfit.py` | llmfit binary management: install, invoke, parse JSON output |
| `froggy/config.py` | Config file read/write (`~/.froggy/config.yaml`) |

Modified files:
| File | Changes |
|------|---------|
| `froggy/cli.py` | Convert from `@click.command` to `@click.group`, add subcommands |
| `froggy/discovery.py` | Default search path → `~/.froggy/models/` |
| `pyproject.toml` | Add `huggingface-hub` to core deps, update entry point |
| `README.md` | Document new commands, installation, usage |

---

## 12. Development Methodology: Test-Driven Development (TDD)

All implementation work in this upgrade **must follow TDD** (Red → Green → Refactor):

### 12.1 TDD Workflow

For every new function, command, or behavior:

1. **Red** — Write a failing test first that defines the expected behavior
2. **Green** — Write the minimum implementation code to make the test pass
3. **Refactor** — Clean up the implementation while keeping tests green

### 12.2 TDD Rules

- **No production code without a failing test.** Every new function, CLI command, and code path must have a corresponding test written *before* the implementation.
- **One behavior per test.** Each test should assert a single, clearly named behavior (e.g., `test_parse_source_bare_repo_id`, not `test_parse_source`).
- **Tests run continuously.** Run `pytest` after every Red → Green → Refactor cycle. Never move to the next feature with failing tests.
- **Mock external dependencies, not internal logic.** Mock `huggingface_hub` API calls, `subprocess.run` (for llmfit), and filesystem I/O where needed. Use `tmp_path` fixtures for directory operations — never touch real `~/.froggy/`.
- **Test the CLI surface.** Use Click's `CliRunner` to test command invocation, argument parsing, help text, and error messages.
- **Commit rhythm:** Each passing test + its implementation is a valid commit point. Keep commits small and atomic.

### 12.3 TDD by Phase

Each phase below follows the pattern: **write tests → confirm they fail → implement → confirm they pass**.

---

## 13. Implementation Order

### Phase 1: Foundation
1. **Test:** Write `test_paths.py` — `froggy_home()` returns default, respects `FROGGY_HOME`, `models_dir()` appends `/models`, `ensure_froggy_home()` creates directories (use `tmp_path` + monkeypatch)
2. **Confirm tests fail** (module doesn't exist yet)
3. Create `froggy/paths.py` — implement until tests pass
4. **Test:** Write `test_cli_group.py` — bare `froggy` invokes chat, `froggy --help` lists subcommands, `froggy chat` works explicitly (use `CliRunner`)
5. **Confirm tests fail**
6. Refactor `cli.py` to use `click.group()` with backwards-compatible bare invocation
7. Update `pyproject.toml` entry point
8. Update `discovery.py` to default to `~/.froggy/models/`
9. **Confirm all tests pass**, then refactor

### Phase 2: Download
10. **Test:** Write `test_source_parsing.py` — bare repo ID, full URL, URL with blob path, URL with revision, invalid source raises `BadParameter`
11. **Confirm tests fail**
12. Implement `parse_source()` in `froggy/download.py` — until tests pass
13. **Test:** Write `test_platform_detection.py` — `detect_platform()` returns `mlx` on Apple Silicon, `gguf` on Intel Mac, `gguf` on Linux/Windows
14. **Confirm tests fail**
15. Implement `detect_platform()` in `froggy/download.py` — until tests pass
16. **Test:** Write `test_mlx_discovery.py` — `find_mlx_repo()` finds `mlx-community` variants, returns None when no MLX version exists, respects quantization preferences
17. **Confirm tests fail**
18. Implement `find_mlx_repo()` in `froggy/download.py` — until tests pass
19. **Test:** Write `test_download.py` — mock `snapshot_download` for MLX and safetensors flows, mock `hf_hub_download` for single GGUF file, mock `list_repo_files` for GGUF listing, verify auto-detection picks MLX on Apple Silicon, verify fallback from MLX to GGUF when no MLX repo exists, verify `--force` re-downloads, verify duplicate check skips
20. **Confirm tests fail**
21. Implement download flows (Flow Auto, Flow MLX, Flow Safetensors, Flow GGUF Auto, Flow GGUF Single, Flow Pick) — until tests pass
22. **Test:** Write CLI-level tests for `froggy download` — argument parsing, `--file`, `--format mlx`, `--format gguf`, `--format auto`, `--pick`, `--name`, `--force`, error cases (no compatible format found)
23. **Confirm tests fail**, implement CLI wiring, **confirm pass**

### Phase 3: Model Management
18. **Test:** Write `test_list_remove.py` — `froggy list` shows models in table format, `--json` outputs valid JSON, `froggy remove` deletes directory, `--yes` skips confirmation, removing nonexistent model shows error
19. **Confirm tests fail**
20. Implement `froggy list` with Rich table output — until tests pass
21. Implement `froggy remove` with confirmation — until tests pass
22. **Test:** Write `test_info.py` — `froggy info` displays model metadata, nonexistent model shows error
23. **Confirm tests fail**
24. Implement `froggy info` — until tests pass

### Phase 4: llmfit Integration
25. **Test:** Write `test_llmfit.py` — `get_llmfit_path()` finds bundled binary, falls back to PATH, returns None when missing; `llmfit_recommend()` parses JSON output; `llmfit_system_info()` parses JSON output; subprocess timeout handled; `check_llmfit_update()` skips check if <24h since last check, returns new version when available, returns None on network error; `froggy recommend` CLI renders table
26. **Confirm tests fail**
27. Create `froggy/llmfit.py` with binary detection, auto-update, and subprocess calls — until tests pass
28. Implement `froggy recommend` command — until tests pass
29. **Test:** Wire `--format auto` to llmfit data (MLX quant selection on Mac, GGUF quant selection elsewhere), write test for fallback heuristic when llmfit unavailable
30. **Confirm tests fail**, implement, **confirm pass**

### Phase 5: Config & Polish
31. **Test:** Write `test_config.py` — config read/write/set, default values, `froggy config` CLI output
32. **Confirm tests fail**
33. Create `froggy/config.py` and `froggy config` command — until tests pass
34. Update README.md with full documentation
35. **Final CI verification:** full test suite green across Python 3.11 + 3.12

---

## 14. Testing Plan

All tests are written **before** their corresponding implementation (see Section 12). The table below summarizes the full test suite:

| Test | What it verifies | Phase |
|------|-----------------|-------|
| `test_paths.py` | `froggy_home()` respects `FROGGY_HOME`, `models_dir()` path, `ensure_froggy_home()` creates dirs | 1 |
| `test_cli_group.py` | Bare `froggy` invokes chat, `froggy chat` works, `--help` lists subcommands | 1 |
| `test_source_parsing.py` | URL parsing: bare repo IDs, full URLs, URLs with blob/tree paths, revision extraction, invalid input errors | 2 |
| `test_platform_detection.py` | `detect_platform()` returns correct format per OS/arch (mlx on Apple Silicon, gguf elsewhere) | 2 |
| `test_mlx_discovery.py` | `find_mlx_repo()` finds mlx-community variants, returns None when missing, respects quant preferences | 2 |
| `test_download.py` | Mock `huggingface_hub` calls, auto-detection picks MLX on Apple Silicon, MLX→GGUF fallback, correct args for snapshot/single-file, GGUF listing/selection, `--force` re-download, duplicate skip | 2 |
| `test_list_remove.py` | `list` shows Rich table, `--json` outputs valid JSON, `remove` deletes directory, `--yes` skips prompt, error on missing model | 3 |
| `test_info.py` | `info` displays model metadata (format, size, type), error on missing model | 3 |
| `test_llmfit.py` | Binary detection (bundled → PATH → None), JSON parsing of recommend/system output, subprocess timeout, auto-update check (skips if <24h, downloads if newer version, handles network errors gracefully), `recommend` CLI table | 4 |
| `test_config.py` | Config read/write/set, default values, CLI output | 5 |

### Testing conventions

- Use `tmp_path` fixtures to avoid touching real `~/.froggy/`
- Use `monkeypatch` to override `FROGGY_HOME` and other env vars
- Use Click's `CliRunner` for all CLI-level tests
- Mock `huggingface_hub` and `subprocess.run` — never make real network calls in tests
- Each test function asserts one behavior and is named descriptively (e.g., `test_parse_source_full_url_with_revision`)

---

## 15. UX Examples

### First-time user experience (macOS Apple Silicon)

```
$ pip install froggy[mlx]
$ froggy recommend
Created froggy home at ~/.froggy/
Downloading llmfit... done

Your hardware:
  RAM: 32 GB (unified) | GPU: Apple M2 Pro | CPU: 12 cores

Top recommendations:
  1. Llama 3 8B Instruct  [Score: 95]  MLX 4-bit  ~4.3 GB
  2. Mistral 7B Instruct  [Score: 93]  MLX 4-bit  ~4.1 GB
  3. Phi-3 Mini 3.8B      [Score: 91]  MLX 4-bit  ~2.8 GB

Download a model? [1/2/3/n]: 1

$ froggy download meta-llama/Llama-3-8B-Instruct
Detected: macOS Apple Silicon — looking for MLX version...
Found: mlx-community/Llama-3-8B-Instruct-4bit
Downloading... ████████████████████████████ 100%  4.3 GB
Saved to ~/.froggy/models/Llama-3-8B-Instruct-4bit/

$ froggy
   __
  / _|_ __ ___   __ _  __ _ _   _
 | |_| '__/ _ \ / _` |/ _` | | | |
 |  _| | | (_) | (_| | (_| | |_| |
 |_| |_|  \___/ \__, |\__, |\__, |
                 |___/ |___/ |___/
Chat with your local models

┌───────────────────────────────────┬───────────────┬────────┐
│ # │ Name                          │ Type          │ Format │
├───┼───────────────────────────────┼───────────────┼────────┤
│ 1 │ Llama-3-8B-Instruct-4bit     │ LlamaFor...   │ MLX    │
└───┴───────────────────────────────┴───────────────┴────────┘

Select a model: 1
Loading... done (MLX backend)

You: Hello!
```

### First-time user experience (Linux with NVIDIA GPU)

```
$ pip install froggy[gpu]
$ froggy download mistralai/Mistral-7B-Instruct-v0.3
Detected: Linux (NVIDIA RTX 4060, 8 GB VRAM) — using GGUF format
Fetching repo info... found 6 GGUF variants
Selected: mistral-7b-instruct-v0.3.Q4_K_M.gguf (4.1 GB)
Downloading... ████████████████████████████ 100%  4.1 GB
Saved to ~/.froggy/models/mistral-7b-instruct-v0.3-Q4_K_M/
```

### Power user

```
$ froggy download https://huggingface.co/meta-llama/Llama-3-8B-Instruct --name llama3
$ froggy download meta-llama/Llama-3-8B-Instruct --format gguf    # Force GGUF even on Mac
$ froggy download meta-llama/Llama-3-8B-Instruct --pick            # See all variants (MLX, GGUF, safetensors)
$ froggy list --json | jq '.[].name'
$ froggy remove llama-2-7b --yes
$ FROGGY_HOME=/mnt/fast-ssd/.froggy froggy chat
```

---

## 16. Out of Scope (Future Work)

- Model quantization/conversion within froggy (use external tools)
- Model registry / search HuggingFace from CLI (keep it simple — user provides repo ID)
- Ollama-style Modelfile / custom model creation
- Model sharing / upload
- Windows binary bundling for llmfit (Linux + macOS first)
