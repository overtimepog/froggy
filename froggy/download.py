"""Download building blocks and orchestration for HuggingFace model pulling.

Pure functions for source parsing, platform detection, MLX repo discovery,
GGUF file listing/selection, and system memory detection.  Orchestrator
functions compose these into the fallback download chain used by the CLI.
"""

from __future__ import annotations

import logging
import os
import platform
import re
from dataclasses import dataclass
from pathlib import Path

import click
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError
from rich.console import Console

from .llmfit import get_memory_budget
from .paths import ensure_froggy_home, models_dir

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Source parsing
# ---------------------------------------------------------------------------

_HF_URL_RE = re.compile(
    r"https?://huggingface\.co/"
    r"(?P<repo>[^/]+/[^/?#]+)"        # org/model
    r"(?:/(?P<action>blob|tree)"       # optional /blob or /tree
    r"/(?P<revision>[^/]+)"           # revision (branch/tag)
    r"(?:/(?P<path>.+))?"             # optional file path
    r")?"
    r"[/?#]?.*$"                       # trailing slash / query / fragment
)


@dataclass(frozen=True)
class ParsedSource:
    """Result of parsing a HuggingFace source string."""

    repo_id: str
    filename: str | None = None
    revision: str | None = None


def parse_source(source: str) -> ParsedSource:
    """Parse a HuggingFace source string into its components.

    Accepts:
    - Bare repo IDs: ``org/model``
    - Full URLs: ``https://huggingface.co/org/model``
    - URLs with paths: ``.../blob/main/file.gguf``, ``.../tree/v2``

    Raises ``ValueError`` on unparseable input.
    """
    source = source.strip()
    if not source:
        raise ValueError("Cannot parse empty source string")

    # URL form
    if source.startswith(("http://", "https://")):
        m = _HF_URL_RE.match(source)
        if not m:
            raise ValueError(f"Cannot parse HuggingFace URL: {source}")
        repo_id = m.group("repo").rstrip("/")
        action = m.group("action")
        revision = m.group("revision")
        path = m.group("path")
        filename = path if action == "blob" and path else None
        revision_out = revision if action else None
        return ParsedSource(
            repo_id=repo_id, filename=filename, revision=revision_out
        )

    # Bare repo ID: must contain exactly one slash
    if "/" in source and not source.startswith("/"):
        parts = source.strip("/").split("/")
        if len(parts) == 2 and parts[0] and parts[1]:
            return ParsedSource(repo_id=f"{parts[0]}/{parts[1]}")

    raise ValueError(f"Cannot parse source: {source}")


# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------


def detect_platform() -> str:
    """Return ``'mlx'`` on macOS ARM64, ``'gguf'`` elsewhere."""
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "mlx"
    return "gguf"


# ---------------------------------------------------------------------------
# MLX repo discovery
# ---------------------------------------------------------------------------

_MLX_PATTERNS = [
    "mlx-community/{name}-4bit",
    "mlx-community/{name}-8bit",
    "mlx-community/{name}-bf16",
    "mlx-community/{name}-fp16",
    "mlx-community/{name}-4bit-mlx",
    "mlx-community/{name}-mlx-4bit",
]


def find_mlx_repo(repo_id: str, api: HfApi) -> str | None:
    """Search ``mlx-community`` for a converted version of *repo_id*.

    Tries several naming patterns and returns the first that exists,
    or ``None`` if nothing is found.
    """
    model_name = repo_id.split("/")[-1]
    for pattern in _MLX_PATTERNS:
        candidate = pattern.format(name=model_name)
        try:
            api.model_info(candidate)
            return candidate
        except (RepositoryNotFoundError, HfHubHTTPError):
            continue
    return None


# ---------------------------------------------------------------------------
# GGUF file listing & selection
# ---------------------------------------------------------------------------

_QUANT_RE = re.compile(
    r"[_.-]"
    r"(Q[0-9]+_[A-Z0-9_]+|q[0-9]+_[a-z0-9_]+|[Ff][0-9]+|IQ[0-9]+_[A-Z]+)"
    r"(?:[_.]|$)"
)

# Higher score = better quality.  Covers the most common llama.cpp quants.
_QUANT_RANK: dict[str, int] = {
    "Q2_K": 10,
    "Q3_K_S": 20,
    "Q3_K_M": 25,
    "Q3_K_L": 30,
    "Q4_0": 35,
    "Q4_K_S": 40,
    "Q4_K_M": 45,
    "Q4_K": 45,
    "Q5_0": 50,
    "Q5_K_S": 55,
    "Q5_K_M": 60,
    "Q6_K": 70,
    "Q8_0": 80,
    "f16": 90,
    "F16": 90,
    "f32": 95,
    "F32": 95,
    "IQ2_XXS": 5,
    "IQ2_XS": 6,
    "IQ3_XXS": 15,
    "IQ3_XS": 18,
    "IQ4_XS": 38,
}


def _parse_quant(filename: str) -> str:
    """Extract quantization tag from a GGUF filename, or ``'unknown'``."""
    m = _QUANT_RE.search(filename)
    return m.group(1) if m else "unknown"


def list_gguf_files(repo_id: str, api: HfApi) -> list[dict]:
    """Return a list of GGUF files in *repo_id* with parsed metadata.

    Each dict has keys ``filename``, ``size``, and ``quant``.
    """
    info = api.model_info(repo_id, files_metadata=True)
    results: list[dict] = []
    for sibling in info.siblings:
        if sibling.rfilename.endswith(".gguf"):
            results.append(
                {
                    "filename": sibling.rfilename,
                    "size": sibling.size,
                    "quant": _parse_quant(sibling.rfilename),
                }
            )
    return results


def select_gguf_file(
    files: list[dict], max_memory: int | None = None
) -> dict | None:
    """Pick the highest-quality GGUF file that fits in memory.

    If *max_memory* is provided the file must be smaller than 70% of it.
    Returns ``None`` when *files* is empty or nothing fits.
    """
    if not files:
        return None

    candidates = files
    if max_memory is not None:
        budget = int(max_memory * 0.7)
        candidates = [f for f in files if f["size"] <= budget]

    if not candidates:
        return None

    return max(candidates, key=lambda f: _QUANT_RANK.get(f["quant"], 0))


# ---------------------------------------------------------------------------
# System memory
# ---------------------------------------------------------------------------


def get_system_memory() -> int | None:
    """Return total physical memory in bytes, or ``None`` on failure."""
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
        return page_size * page_count
    except (ValueError, OSError, AttributeError):
        return None


# ---------------------------------------------------------------------------
# Variant listing (for --pick mode)
# ---------------------------------------------------------------------------


def list_variants(
    repo_id: str, api: HfApi | None = None
) -> list[dict]:
    """Return all downloadable variants for *repo_id*.

    Each variant dict has keys: ``type`` (``"mlx"``/``"gguf"``/``"safetensors"``),
    ``repo``, ``filename`` (or ``None``), ``size`` (or ``None``), ``quant`` (or ``None``).
    """
    if api is None:
        api = HfApi()

    variants: list[dict] = []

    # MLX variant
    mlx_repo = find_mlx_repo(repo_id, api)
    if mlx_repo:
        variants.append({
            "type": "mlx",
            "repo": mlx_repo,
            "filename": None,
            "size": None,
            "quant": None,
        })

    # GGUF variants
    try:
        gguf_files = list_gguf_files(repo_id, api)
        for gf in gguf_files:
            variants.append({
                "type": "gguf",
                "repo": repo_id,
                "filename": gf["filename"],
                "size": gf["size"],
                "quant": gf["quant"],
            })
    except (RepositoryNotFoundError, HfHubHTTPError):
        pass

    # Safetensors variant (the original repo itself)
    variants.append({
        "type": "safetensors",
        "repo": repo_id,
        "filename": None,
        "size": None,
        "quant": None,
    })

    return variants


# ---------------------------------------------------------------------------
# Download orchestrator
# ---------------------------------------------------------------------------

_console = Console(stderr=True)


def _dest_dir(repo_id: str) -> Path:
    """Compute download destination: ``models_dir() / repo_id.replace('/', '--')``."""
    ensure_froggy_home()
    return models_dir() / repo_id.replace("/", "--")


def download_model(
    source: str,
    fmt: str = "auto",
    api: HfApi | None = None,
) -> Path:
    """Download a model using the MLX → GGUF → safetensors fallback chain.

    Parameters
    ----------
    source:
        A HuggingFace repo ID or URL.
    fmt:
        ``"auto"`` detects platform; ``"mlx"``/``"gguf"``/``"safetensors"``
        forces that format.
    api:
        Optional HfApi instance (pass a mock in tests).

    Returns the local ``Path`` where files were saved.

    Raises ``click.ClickException`` if no format succeeds.
    """
    if api is None:
        api = HfApi()

    parsed = parse_source(source)
    dest = _dest_dir(parsed.repo_id)

    # Direct filename download — user gave a specific file URL
    if parsed.filename:
        _console.print(
            f"[cyan]Downloading[/] {parsed.filename} from {parsed.repo_id}…"
        )
        try:
            hf_hub_download(
                parsed.repo_id,
                parsed.filename,
                revision=parsed.revision,
                local_dir=str(dest),
                token=None,
            )
            _console.print(f"[green]✓[/] Saved to {dest}")
            return dest
        except Exception as exc:
            raise click.ClickException(
                f"Failed to download {parsed.filename}: {exc}"
            ) from exc

    # Determine target format
    if fmt == "auto":
        target = detect_platform()
    else:
        target = fmt

    # --- MLX path ---
    if target in ("mlx", "auto"):
        mlx_repo = find_mlx_repo(parsed.repo_id, api)
        if mlx_repo:
            _console.print(
                f"[cyan]Downloading MLX model[/] {mlx_repo}…"
            )
            mlx_dest = _dest_dir(mlx_repo)
            try:
                snapshot_download(
                    mlx_repo,
                    allow_patterns=["*.safetensors", "*.json", "*.jinja", "tokenizer*"],
                    local_dir=str(mlx_dest),
                    token=None,
                )
                _console.print(f"[green]✓[/] MLX model saved to {mlx_dest}")
                return mlx_dest
            except Exception as exc:
                logger.debug("MLX download failed: %s", exc)
                _console.print(
                    "[yellow]MLX download failed, trying GGUF…[/]"
                )

    # --- GGUF path ---
    if target in ("gguf", "mlx", "auto"):
        try:
            gguf_files = list_gguf_files(parsed.repo_id, api)
        except (RepositoryNotFoundError, HfHubHTTPError):
            gguf_files = []

        if gguf_files:
            mem = get_memory_budget()
            selected = select_gguf_file(gguf_files, max_memory=mem)
            if selected:
                _console.print(
                    f"[cyan]Downloading GGUF[/] {selected['filename']} "
                    f"({selected['quant']})…"
                )
                try:
                    hf_hub_download(
                        parsed.repo_id,
                        selected["filename"],
                        local_dir=str(dest),
                        token=None,
                    )
                    _console.print(f"[green]✓[/] GGUF saved to {dest}")
                    return dest
                except Exception as exc:
                    logger.debug("GGUF download failed: %s", exc)
                    _console.print(
                        "[yellow]GGUF download failed, trying safetensors…[/]"
                    )

    # --- Safetensors path ---
    if target in ("safetensors", "gguf", "mlx", "auto"):
        _console.print(
            f"[cyan]Downloading safetensors[/] from {parsed.repo_id}…"
        )
        try:
            previous_disable_xet = os.environ.get("HF_HUB_DISABLE_XET")
            os.environ["HF_HUB_DISABLE_XET"] = "1"
            try:
                snapshot_download(
                    parsed.repo_id,
                    allow_patterns=["*.safetensors", "*.json", "*.jinja", "tokenizer*"],
                    local_dir=str(dest),
                    token=None,
                )
            finally:
                if previous_disable_xet is None:
                    os.environ.pop("HF_HUB_DISABLE_XET", None)
                else:
                    os.environ["HF_HUB_DISABLE_XET"] = previous_disable_xet
            _console.print(f"[green]✓[/] Safetensors saved to {dest}")
            return dest
        except Exception as exc:
            logger.debug("Safetensors download failed: %s", exc)

    # Nothing worked
    raise click.ClickException(
        f"No compatible model format found for '{source}'. "
        f"Tried formats for target '{target}'. "
        "Check the repo exists and contains downloadable files."
    )
