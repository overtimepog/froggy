"""CLI entry point and main loop."""

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import IntPrompt, Prompt
from rich.table import Table

from .backends import pick_backend
from .discovery import ModelInfo, discover_models, discover_ollama_models
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


@click.command()
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
def main(models_dir: Path | None, device: str, tools_dir: Path | None):
    """Chat with local HuggingFace models in your terminal."""

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
    tool_registry, tool_executor = _build_tool_system(tools_dir)

    while True:
        model_info = select_model(models)
        if model_info is None:
            break

        backend = pick_backend(model_info)
        session = ChatSession(
            backend,
            model_info,
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

    console.print("[dim]Goodbye! \U0001f438[/]")
