"""CLI entry point and main loop."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import IntPrompt, Prompt
from rich.table import Table

from .backends import pick_backend
from .config import get_config, load_config, set_config
from .discovery import ModelInfo, discover_models, discover_ollama_models
from .download import download_model, list_variants
from .llmfit import ensure_llmfit, llmfit_recommend
from .models import list_models, model_info, remove_model
from .paths import models_dir
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
[dim]Chat with your local models[/]"""


def select_model(models: list[ModelInfo]) -> ModelInfo | None:
    tbl = Table(title="Available Models", border_style="cyan", title_style="bold cyan")
    tbl.add_column("#", style="bold", width=3)
    tbl.add_column("Name")
    tbl.add_column("Type", style="dim")

    for i, m in enumerate(models, 1):
        tbl.add_row(str(i), m.label, m.model_type)

    console.print()
    console.print(tbl)
    console.print()

    try:
        choice = IntPrompt.ask(
            "[bold]Select a model[/]",
            choices=[str(i) for i in range(1, len(models) + 1)],
        )
        return models[choice - 1]
    except (KeyboardInterrupt, EOFError):
        return None


def _build_tool_system(tools_dir: Path | None):
    """Instantiate ToolRegistry and ToolExecutor if the tool modules are available.

    Returns ``(registry, executor, mcp_manager)`` or ``(None, None, None)`` when unavailable.
    """
    if not _TOOLS_AVAILABLE:
        return None, None, None

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

        # Connect to MCP servers
        mcp_manager = None
        try:
            from .mcp_client import MCPManager
            mcp_mgr = MCPManager()
            mcp_tools = mcp_mgr.connect_all()
            if mcp_tools:
                all_tools.extend(mcp_tools)
                server_names = mcp_mgr.server_names
                console.print(
                    f"[dim]MCP: {len(mcp_tools)} tool(s) from "
                    f"{len(server_names)} server(s) ({', '.join(server_names)})[/]"
                )
                mcp_manager = mcp_mgr
        except ImportError:
            pass  # mcp package not installed
        except Exception as exc:
            console.print(f"[dim]MCP unavailable: {exc}[/]")

        registry = ToolRegistry(tools=all_tools)
        executor = ToolExecutor(mcp_manager=mcp_manager)
        return registry, executor, mcp_manager
    except Exception as exc:
        console.print(f"[dim]Tool system unavailable: {exc}[/]")
        return None, None, None


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """Chat with local HuggingFace models in your terminal."""
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
        # Default: use the managed ~/.froggy/models/ directory
        from .paths import models_dir as _models_dir
        managed = _models_dir()
        if managed.is_dir() and any(managed.iterdir()):
            models_dir = managed
        else:
            # Fallback: look in ../AI relative to the froggy package (dev layout)
            dev_dir = Path(__file__).resolve().parent.parent.parent / "AI"
            if dev_dir.is_dir():
                models_dir = dev_dir
            else:
                models_dir = managed  # point at ~/.froggy/models even if empty (for error msg)

    # Resolve tools directory
    if tools_dir is None:
        project_root = Path(FROGGY_PROJECT_ROOT) if FROGGY_PROJECT_ROOT else Path.cwd()
        candidate = project_root / "tools"
        tools_dir = candidate if candidate.is_dir() else None

    console.print(Panel(BANNER, border_style="green", padding=(0, 2)))

    models = discover_models(models_dir)

    # Also discover Ollama models if server is running
    ollama_models = discover_ollama_models()
    if ollama_models:
        console.print(f"[dim]Found {len(ollama_models)} model(s) on Ollama server[/]")
        models.extend(ollama_models)

    if not models:
        console.print(f"[red]No models found in {models_dir}[/]")
        console.print("[dim]Models need a config.json and weight files in their directory,[/]")
        console.print("[dim]or start an Ollama server (ollama serve).[/]")
        sys.exit(1)

    # Build tool system once (shared across model switches within a session)
    tool_registry, tool_executor, mcp_manager = _build_tool_system(tools_dir)

    while True:
        selected_model = select_model(models)
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

        console.print("\n[dim]Type [cyan]/help[/dim][cyan][/] for commands, [cyan]/quit[/] to exit.[/]\n")

        try:
            while True:
                try:
                    user_input = Prompt.ask("[bold green]You[/]")
                except (KeyboardInterrupt, EOFError):
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

    # Clean up MCP connections
    if mcp_manager is not None:
        mcp_manager.disconnect_all()

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
    """Remove a downloaded model."""
    mdir = models_dir()
    try:
        info = model_info(name, mdir)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    size_str = _format_size(info["size"])

    if not yes:
        click.confirm(
            f"Remove '{info['name']}' ({size_str})?",
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
        f"[bold]Has JANG:[/]      {'Yes' if meta.get('has_jang') else 'No'}",
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
# MCP command group
# ---------------------------------------------------------------------------


@cli.command()
def mcp():
    """List MCP servers and their tools."""
    try:
        from .mcp_client import MCPManager, load_mcp_config, mcp_config_path
    except ImportError:
        console.print("[red]MCP support requires the 'mcp' package:[/] pip install mcp")
        return

    configs = load_mcp_config()
    if not configs:
        console.print("[yellow]No MCP servers configured.[/]")
        console.print(f"[dim]Create {mcp_config_path()} with:[/]")
        console.print()
        console.print("[dim]servers:[/]")
        console.print("[dim]  fetch:[/]")
        console.print("[dim]    command: uvx[/]")
        console.print("[dim]    args: [mcp-server-fetch][/]")
        return

    console.print(f"[cyan]Connecting to {len(configs)} MCP server(s)...[/]")
    mgr = MCPManager()
    try:
        tools = mgr.connect_all()
    except Exception as exc:
        console.print(f"[red]Error:[/] {exc}")
        return

    tbl = Table(title="MCP Servers & Tools", border_style="cyan", title_style="bold cyan")
    tbl.add_column("Server", style="bold")
    tbl.add_column("Tool", style="green")
    tbl.add_column("Description", style="dim")

    for server_name in mgr.server_names:
        first = True
        for tool_name in mgr.server_tools(server_name):
            # Find the ToolDef
            td = next((t for t in tools if t.name == tool_name), None)
            desc = td.description if td else ""
            tbl.add_row(
                server_name if first else "",
                tool_name,
                desc,
            )
            first = False

    console.print(tbl)
    console.print(f"\n[dim]{len(tools)} total tool(s) available during chat[/]")

    mgr.disconnect_all()
