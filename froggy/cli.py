"""CLI entry point and main loop."""

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import IntPrompt, Prompt
from rich.table import Table

from .backends import pick_backend
from .config import get_config, load_config, save_config, set_config
from .discovery import ModelInfo, discover_models, discover_ollama_models, discover_openrouter_models
from .download import download_model, list_variants
from .llmfit import ensure_llmfit, llmfit_recommend
from .models import list_models, model_info, remove_model
from .paths import models_dir
from .saved_models import add_saved_model, get_saved_models, parse_model_source, remove_saved_model
from .session import (
    _TOOLS_AVAILABLE,
    FROGGY_PROJECT_ROOT,
    ChatSession,
    handle_command,
    load_custom_tools,
)

console = Console()

BANNER = r"""[bold green]
   __
  / _|_ __ ___   __ _  __ _ _   _
 | |_| '__/ _ \ / _` |/ _` | | | |
 |  _| | | (_) | (_| | (_| | |_| |
 |_| |_|  \___/ \__, |\__, |\__, |
                 |___/ |___/ |___/[/]
[dim]A tiny agent harness for any model[/]"""


def select_model(models: list[ModelInfo]) -> ModelInfo | None:
    """Interactive model selection with search when the list is large."""
    # This is only called when there are no saved models, or from the
    # "Browse all models" path. Simple search + pick.
    if len(models) > 20:
        console.print(f"\n[dim]{len(models)} models available.[/]")
        console.print("[dim]Type a search term to filter, or press Enter to see all.[/]\n")
        try:
            query = Prompt.ask("[bold]Search models[/]", default="")
        except (KeyboardInterrupt, EOFError):
            return None

        if query.strip():
            filtered = [m for m in models if query.lower() in m.name.lower() or query.lower() in m.model_type.lower()]
            if not filtered:
                console.print(f"[yellow]No models matching '{query}'.[/] Showing all.")
                filtered = models
            else:
                console.print(f"[dim]Found {len(filtered)} match(es).[/]")
        else:
            filtered = models
    else:
        filtered = models

    tbl = Table(title="Available Models", border_style="cyan", title_style="bold cyan")
    tbl.add_column("#", style="bold", width=4)
    tbl.add_column("Name")
    tbl.add_column("Type", style="dim")

    for i, m in enumerate(filtered, 1):
        tbl.add_row(str(i), m.label, m.model_type)

    console.print()
    console.print(tbl)
    console.print()

    try:
        choice = IntPrompt.ask(
            "[bold]Select a model[/]",
            choices=[str(i) for i in range(1, len(filtered) + 1)],
        )
        return filtered[choice - 1]
    except (KeyboardInterrupt, EOFError):
        return None


def select_from_saved_or_browse(
    saved: list[ModelInfo],
    all_models: list[ModelInfo],
) -> ModelInfo | None:
    """Show saved models as a short menu with a 'Browse all' option at the end."""
    tbl = Table(title="Your Models", border_style="cyan", title_style="bold cyan")
    tbl.add_column("#", style="bold", width=4)
    tbl.add_column("Name")
    tbl.add_column("Type", style="dim")

    for i, m in enumerate(saved, 1):
        tbl.add_row(str(i), f"⭐ {m.label}", m.model_type)

    browse_idx = len(saved) + 1
    other_count = len(all_models)
    tbl.add_row(
        str(browse_idx),
        f"[dim]Browse all models ({other_count} available)...[/]",
        "",
    )

    console.print()
    console.print(tbl)
    console.print()

    try:
        choice = IntPrompt.ask(
            "[bold]Select a model[/]",
            choices=[str(i) for i in range(1, browse_idx + 1)],
        )
    except (KeyboardInterrupt, EOFError):
        return None

    if choice == browse_idx:
        # User wants to browse the full catalog
        return select_model(all_models)

    return saved[choice - 1]


def _build_tool_system(tools_dir: Path | None):
    """Instantiate ToolRegistry and ToolExecutor if the tool modules are available.

    Returns ``(registry, executor)`` or ``(None, None)`` when unavailable.
    """
    if not _TOOLS_AVAILABLE:
        return None, None

    try:
        from .tool_executor import ToolExecutor
        from .tools import CORE_TOOLS, ToolRegistry

        # Start with core tools
        all_tools = list(CORE_TOOLS)

        # Append custom tools from the tools/ directory
        if tools_dir is not None and tools_dir.is_dir():
            custom = load_custom_tools(tools_dir)
            if custom:
                console.print(f"[dim]Loaded {len(custom)} custom tool(s) from {tools_dir}[/]")
            all_tools.extend(custom)

        registry = ToolRegistry(tools=all_tools)
        executor = ToolExecutor()
        return registry, executor
    except Exception as exc:
        console.print(f"[dim]Tool system unavailable: {exc}[/]")
        return None, None


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """A tiny agent harness for any model."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(chat)


@cli.command()
@click.argument("source")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["auto", "mlx", "gguf", "safetensors"]),
    default="auto",
    help="Model format to download.",
)
@click.option("--pick", is_flag=True, help="Interactively pick from available variants.")
def download(source, fmt, pick):
    """Download a HuggingFace model to ~/.froggy/models/."""
    if pick:
        from huggingface_hub import HfApi
        from rich.prompt import IntPrompt

        api = HfApi()
        console.print(f"[cyan]Scanning variants for[/] {source}…")
        try:
            from .download import parse_source as _ps

            _ps(source)  # validate input early
            variants = list_variants(source, api=api)
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc

        if not variants:
            raise click.ClickException(f"No variants found for '{source}'.")

        tbl = Table(title="Available Variants", border_style="cyan", title_style="bold cyan")
        tbl.add_column("#", style="bold", width=3)
        tbl.add_column("Type")
        tbl.add_column("Repo / File")
        tbl.add_column("Quant", style="dim")
        tbl.add_column("Size", style="dim")

        for i, v in enumerate(variants, 1):
            label = v["filename"] if v["filename"] else v["repo"]
            quant = v["quant"] or "—"
            size = f"{v['size'] / 1e9:.1f} GB" if v["size"] else "—"
            tbl.add_row(str(i), v["type"], label, quant, size)

        console.print()
        console.print(tbl)
        console.print()

        try:
            choice = IntPrompt.ask(
                "[bold]Select a variant[/]",
                choices=[str(i) for i in range(1, len(variants) + 1)],
            )
        except (KeyboardInterrupt, EOFError):
            return

        selected = variants[choice - 1]
        # Download the chosen variant
        if selected["type"] == "gguf" and selected["filename"]:
            download_model(
                f"https://huggingface.co/{selected['repo']}/blob/main/{selected['filename']}",
                fmt="gguf",
                api=api,
            )
        else:
            download_model(source, fmt=selected["type"], api=api)
        return

    # Non-pick mode: run fallback chain
    try:
        download_model(source, fmt=fmt)
    except click.ClickException:
        raise
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


@cli.command()
@click.pass_context
@click.option(
    "--models-dir",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Directory to scan for models. Defaults to ../AI relative to this package.",
)
@click.option(
    "--device",
    type=str,
    default="auto",
    help="Device: auto, cpu, cuda, cuda:0, etc.",
)
@click.option(
    "--tools-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory to scan for custom tool plugins. Defaults to tools/ in project root.",
)
def chat(ctx, models_dir: Path | None, device: str, tools_dir: Path | None):
    """Start an interactive chat session with a local model."""

    if models_dir is None:
        # Default: look in ../AI relative to the froggy package
        models_dir = Path(__file__).resolve().parent.parent.parent / "AI"
        if not models_dir.is_dir():
            models_dir = Path.cwd()

    # Resolve tools directory
    if tools_dir is None:
        project_root = Path(FROGGY_PROJECT_ROOT) if FROGGY_PROJECT_ROOT else Path.cwd()
        candidate = project_root / "tools"
        tools_dir = candidate if candidate.is_dir() else None

    console.print(Panel(BANNER, border_style="green", padding=(0, 2)))

    # Start with saved/favorited models
    from .saved_models import saved_models_as_model_info
    saved_infos = saved_models_as_model_info()
    if saved_infos:
        console.print(f"[dim]Loaded {len(saved_infos)} saved model(s)[/]")

    # Discover all other models in the background
    all_models: list[ModelInfo] = []

    local_models = discover_models(models_dir)
    if local_models:
        all_models.extend(local_models)

    ollama_models = discover_ollama_models()
    if ollama_models:
        console.print(f"[dim]Found {len(ollama_models)} model(s) on Ollama server[/]")
        all_models.extend(ollama_models)

    openrouter_models = discover_openrouter_models()
    if openrouter_models:
        console.print(f"[dim]Found {len(openrouter_models)} model(s) on OpenRouter[/]")
        all_models.extend(openrouter_models)

    # Dedup: remove from all_models anything already in saved_infos
    saved_names = {m.name for m in saved_infos}
    all_models = [m for m in all_models if m.name not in saved_names]

    if not saved_infos and not all_models:
        console.print("[red]No models found.[/]")
        console.print("[dim]Use [cyan]froggy add <model>[/dim][cyan][/] to add a model, or set up a backend:[/]")
        console.print("[dim]  • OpenRouter: froggy config set openrouter_api_key <key>[/]")
        console.print("[dim]  • Ollama: ollama serve[/]")
        console.print("[dim]  • Local: froggy download <huggingface-repo>[/]")
        sys.exit(1)

    # Build tool system once (shared across model switches within a session)
    tool_registry, tool_executor = _build_tool_system(tools_dir)

    while True:
        if saved_infos:
            selected_model = select_from_saved_or_browse(saved_infos, all_models)
        elif all_models:
            selected_model = select_model(all_models)
        else:
            break
        if selected_model is None:
            break

        backend = pick_backend(selected_model)
        session = ChatSession(
            backend,
            selected_model,
            device,
            tool_registry=tool_registry,
            tool_executor=tool_executor,
        )

        try:
            session.load()
        except Exception as e:
            console.print(f"[red]Failed to load model:[/] {e}")
            continue

        # Show a compact welcome with useful context
        ctx_limit = session.context_mgr.context_limit
        profile = session.context_mgr.profile_name
        tools_state = "on" if session.tools_enabled else "off"
        console.print(f"\n[dim]Context: {ctx_limit:,} tokens · Profile: {profile} · Tools: {tools_state}[/]")
        console.print("[dim]Type [cyan]/help[/dim][cyan][/] for commands, [cyan]/quit[/] to exit.[/]\n")

        # Set up readline for arrow keys, history, and line editing
        _has_readline = False
        histfile = ""
        try:
            import readline
            histfile = str(Path.home() / ".froggy" / "history")
            Path(histfile).parent.mkdir(parents=True, exist_ok=True)
            try:
                readline.read_history_file(histfile)
            except (FileNotFoundError, PermissionError, OSError):
                pass
            readline.set_history_length(500)
            _has_readline = True
        except (ImportError, OSError):
            pass

        def _input_prompt() -> str | None:
            """Read user input with readline support."""
            try:
                return input("\033[1;32mYou\033[0m: ")
            except (KeyboardInterrupt, EOFError):
                return None

        try:
            while True:
                user_input = _input_prompt()
                if user_input is None:
                    break

                if not user_input.strip():
                    continue

                if user_input.strip().startswith("/"):
                    result = handle_command(user_input.strip(), session)
                    if result == "quit":
                        backend.unload()
                        console.print("[dim]Goodbye! \U0001f438[/]")
                        return
                    elif result == "switch":
                        backend.unload()
                        break
                    continue

                try:
                    session.chat(user_input)
                except KeyboardInterrupt:
                    console.print("\n[dim]Generation interrupted.[/]")
                except Exception as e:
                    console.print(f"[red]Error during generation:[/] {e}")

        except KeyboardInterrupt:
            backend.unload()
            break
        finally:
            # Save readline history
            if _has_readline and histfile:
                try:
                    readline.write_history_file(histfile)
                except (PermissionError, OSError):
                    pass

    console.print("[dim]Goodbye! \U0001f438[/]")


def _format_size(size_bytes: int) -> str:
    """Format a byte count as a human-readable string."""
    if size_bytes >= 1_000_000_000:
        return f"{size_bytes / 1_000_000_000:.1f} GB"
    if size_bytes >= 1_000_000:
        return f"{size_bytes / 1_000_000:.1f} MB"
    if size_bytes >= 1_000:
        return f"{size_bytes / 1_000:.1f} KB"
    return f"{size_bytes} B"


@cli.command("list")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def list_cmd(as_json):
    """List downloaded models."""
    models = list_models(models_dir())

    if not models:
        click.echo("No models found.")
        return

    if as_json:
        click.echo(json.dumps(models, indent=2, default=str))
        return

    tbl = Table(title="Downloaded Models", border_style="cyan", title_style="bold cyan")
    tbl.add_column("Name", style="bold")
    tbl.add_column("Format")
    tbl.add_column("Size", justify="right")
    tbl.add_column("Modified", style="dim")

    for m in models:
        modified = datetime.fromtimestamp(m["modified"], tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M"
        )
        tbl.add_row(m["name"], m["format"], _format_size(m["size"]), modified)

    console.print(tbl)


@cli.command()
@click.argument("name")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt.")
def remove(name, yes):
    """Remove a saved model or a downloaded model.

    Checks saved models first, then downloaded models.
    """
    # Try saved models first
    saved = get_saved_models()
    matches = [m for m in saved if name.lower() in m.get("name", "").lower()]
    if matches:
        if not yes:
            names = ", ".join(m["name"] for m in matches)
            click.confirm(f"Remove saved model(s): {names}?", abort=True)
        ok = remove_saved_model(name)
        if ok:
            console.print(f"[green]✓[/] Removed saved model(s) matching '{name}'.")
        return

    # Fall back to downloaded models
    mdir = models_dir()
    try:
        info = model_info(name, mdir)
    except ValueError:
        # Not found anywhere
        raise click.ClickException(
            f"No model matching '{name}' in saved models or {mdir}.\n"
            f"  Use [cyan]froggy models[/] to see saved models.\n"
            f"  Use [cyan]froggy list[/] to see downloaded models."
        )

    size_str = _format_size(info["size"])

    if not yes:
        click.confirm(
            f"Remove downloaded '{info['name']}' ({size_str})?",
            abort=True,
        )

    freed = remove_model(name, mdir)
    click.echo(f"Removed '{info['name']}' — freed {_format_size(freed)}")


@cli.command()
@click.argument("name")
def info(name):
    """Show detailed information about a model."""
    try:
        meta = model_info(name, models_dir())
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    modified = datetime.fromtimestamp(meta["modified"], tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )

    lines = [
        f"[bold]Name:[/]          {meta['name']}",
        f"[bold]Path:[/]          {meta['path']}",
        f"[bold]Format:[/]        {meta['format']}",
        f"[bold]Size:[/]          {_format_size(meta['size'])}",
        f"[bold]Model type:[/]    {meta['model_type']}",
        f"[bold]Architectures:[/] {', '.join(meta['architectures']) or 'N/A'}",
        f"[bold]Files:[/]         {meta['file_count']}",
        f"[bold]Has GGUF:[/]      {'Yes' if meta['has_gguf'] else 'No'}",
        f"[bold]Has LoRA:[/]      {'Yes' if meta['has_lora'] else 'No'}",
        f"[bold]Modified:[/]      {modified}",
    ]

    console.print(Panel("\n".join(lines), title=meta["name"], border_style="cyan"))


@cli.command()
@click.option("--limit", default=5, type=int, help="Max number of recommendations.")
@click.option("--use-case", default=None, type=str, help="Target use case (e.g. coding, chat).")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def recommend(limit, use_case, as_json):
    """Recommend hardware-matched models via llmfit."""
    binary_path = ensure_llmfit()
    if binary_path is None:
        console.print("[red]Error:[/] Could not install or find llmfit binary.")
        raise SystemExit(1)

    models = llmfit_recommend(binary_path, limit=limit, use_case=use_case)

    if not models:
        console.print("[yellow]No recommendations available.[/]")
        return

    if as_json:
        click.echo(json.dumps(models, indent=2))
        return

    tbl = Table(title="Recommended Models", border_style="cyan", title_style="bold cyan")
    tbl.add_column("Model", style="bold")
    tbl.add_column("Score", justify="right")
    tbl.add_column("Speed (TPS)", justify="right")
    tbl.add_column("Quant", style="dim")
    tbl.add_column("Fit")
    tbl.add_column("Run Mode")
    tbl.add_column("Memory", justify="right")

    for m in models:
        tbl.add_row(
            m["name"],
            str(m["score"]),
            f"{m['estimated_tps']:.1f}",
            m["best_quant"],
            m["fit_level"],
            m["run_mode"],
            f"{m['memory_required_gb']:.1f} GB",
        )

    console.print(tbl)


# ---------------------------------------------------------------------------
# Config command group
# ---------------------------------------------------------------------------


@cli.group(invoke_without_command=True)
@click.pass_context
def config(ctx):
    """View or modify froggy configuration."""
    if ctx.invoked_subcommand is not None:
        return
    # Bare `froggy config` — show all settings
    data = load_config()
    if not data:
        click.echo("No configuration set. Use 'froggy config set <key> <value>' to add settings.")
        return
    for key, value in data.items():
        click.echo(f"{key}: {value}")


@config.command("get")
@click.argument("key")
def config_get(key):
    """Get a config value by key."""
    value = get_config(key)
    if value is None:
        click.echo(f"{key} is not set.")
    else:
        click.echo(f"{key}: {value}")


@config.command("set")
@click.argument("key")
@click.argument("value")
def config_set(key, value):
    """Set a config value."""
    set_config(key, value)
    click.echo(f"Set {key} = {value}")


# ---------------------------------------------------------------------------
# Model management: froggy add / froggy models
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("source")
def add(source):
    """Add a model to your saved list.

    SOURCE can be any of:
      - OpenRouter model ID: openai/gpt-4o
      - OpenRouter URL: https://openrouter.ai/models/openai/gpt-4o
      - HuggingFace repo: mlx-community/Llama-3-8B-4bit
      - HuggingFace URL: https://huggingface.co/TheBloke/Mistral-7B-GGUF
      - Ollama model: ollama:llama3
      - Bare name: llama3 (searches all platforms)
    """
    from .saved_models import discover_variants

    try:
        record = parse_model_source(source)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    # If the source was explicit (URL or org/model), check for variants
    # If it's a bare name or "auto", search across platforms
    if record["source"] == "auto" or record.get("_search"):
        # Bare name — search everywhere
        console.print(f"[cyan]Searching for[/] '{source}' [cyan]across platforms...[/]")
        variants = discover_variants(source)
    else:
        # Explicit source — still search for variants on that platform + others
        console.print(f"[cyan]Searching for variants of[/] '{record['name']}'...")
        # Extract a search term: use the model name part (after the org/)
        search_term = record["name"].split("/")[-1] if "/" in record["name"] else record["name"]
        # Remove common suffixes for broader search
        search_term = re.sub(r"[-:](free|instruct|chat|thinking|preview)$", "", search_term, flags=re.IGNORECASE)
        variants = discover_variants(search_term)

    if not variants:
        # No variants found — just add the parsed record directly
        console.print("[dim]No variants found. Adding as-is.[/]")
        ok = add_saved_model(record)
        if ok:
            console.print(f"[green]✓[/] Added [bold]{record['name']}[/] ({record['source']})")
        else:
            console.print(f"[yellow]Already saved:[/] {record['name']}")
        return

    # Show variant picker
    tbl = Table(title=f"Variants for '{source}'", border_style="cyan", title_style="bold cyan")
    tbl.add_column("#", style="bold", width=4)
    tbl.add_column("Model", min_width=30)
    tbl.add_column("Platform", style="dim", width=12)
    tbl.add_column("Details", style="dim")
    tbl.add_column("Pricing", style="dim", width=15)

    for i, v in enumerate(variants, 1):
        tbl.add_row(
            str(i),
            v["name"],
            v["source"],
            v.get("description", ""),
            v.get("pricing", ""),
        )

    console.print()
    console.print(tbl)
    console.print()
    console.print("[dim]Enter number(s) to add (e.g. 1,3,5), 'all' for all, or 'q' to cancel.[/]")

    try:
        choice = Prompt.ask("[bold]Add which[/]", default="1")
    except (KeyboardInterrupt, EOFError):
        return

    if choice.strip().lower() == "q":
        return

    if choice.strip().lower() == "all":
        indices = list(range(len(variants)))
    else:
        indices = []
        for part in choice.split(","):
            part = part.strip()
            try:
                idx = int(part) - 1
                if 0 <= idx < len(variants):
                    indices.append(idx)
            except ValueError:
                pass

    if not indices:
        console.print("[red]No valid selections.[/]")
        return

    added = 0
    for idx in indices:
        v = variants[idx]
        rec = {
            "name": v["name"],
            "source": v["source"],
            "original": source,
        }
        if v.get("context_length"):
            rec["context_length"] = v["context_length"]
        if add_saved_model(rec):
            console.print(f"  [green]✓[/] {v['name']} ({v['source']})")
            added += 1
        else:
            console.print(f"  [dim]Already saved:[/] {v['name']}")

    console.print(f"\n[bold]{added}[/] model(s) added.")


@cli.group(name="models", invoke_without_command=True)
@click.pass_context
def models_cmd(ctx):
    """Manage your saved model list."""
    if ctx.invoked_subcommand is not None:
        return
    # Bare `froggy models` — show saved models
    saved = get_saved_models()
    if not saved:
        console.print("[dim]No saved models. Use[/] [cyan]froggy add <model>[/] [dim]to add one.[/]")
        return

    tbl = Table(title="Saved Models", border_style="cyan", title_style="bold cyan")
    tbl.add_column("#", style="bold", width=4)
    tbl.add_column("Name", min_width=30)
    tbl.add_column("Platform", style="dim", width=12)

    for i, m in enumerate(saved, 1):
        tbl.add_row(str(i), m.get("name", "?"), m.get("source", "?"))

    console.print(tbl)
    console.print(f"\n[dim]{len(saved)} saved model(s). These appear first when you run[/] [cyan]froggy chat[/].")


@models_cmd.command("remove")
@click.argument("name")
def models_remove(name):
    """Remove a model from your saved list."""
    ok = remove_saved_model(name)
    if ok:
        console.print(f"[green]✓[/] Removed models matching '{name}'.")
    else:
        console.print(f"[red]No saved model matching:[/] {name}")


@models_cmd.command("clear")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation.")
def models_clear(yes):
    """Remove all saved models."""
    saved = get_saved_models()
    if not saved:
        console.print("[dim]No saved models to clear.[/]")
        return
    if not yes:
        click.confirm(f"Remove all {len(saved)} saved models?", abort=True)
    cfg = load_config()
    cfg["models"] = []
    save_config(cfg)
    console.print(f"[green]✓[/] Cleared {len(saved)} saved model(s).")
