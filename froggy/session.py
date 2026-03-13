"""Chat session and slash command handling."""

import re

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .backends import Backend
from .discovery import ModelInfo

# Tokens that signal the model's turn is over (safety net for when EOS
# token IDs alone don't catch it)
_STOP_STRINGS = ["<|im_end|>", "<|im_start|>", "<|endoftext|>",
                 "<|eot_id|>", "<end_of_turn>"]

# Regex to strip <think>...</think> reasoning blocks (including partial ones)
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_THINK_OPEN_RE = re.compile(r"<think>.*", re.DOTALL)


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from model output."""
    text = _THINK_RE.sub("", text)
    # Remove unclosed <think> block at the end (still being generated)
    text = _THINK_OPEN_RE.sub("", text)
    return text

console = Console()

HELP_TEXT = """[bold]Commands:[/]
  [cyan]/help[/]              Show this help
  [cyan]/quit[/]              Exit
  [cyan]/clear[/]             Clear chat history
  [cyan]/system <prompt>[/]   Set system prompt
  [cyan]/temp <value>[/]      Set temperature (0.0 - 2.0)
  [cyan]/tokens <value>[/]    Set max response tokens
  [cyan]/model[/]             Switch to a different model
  [cyan]/info[/]              Show current settings"""


class ChatSession:
    def __init__(self, backend: Backend, model_info: ModelInfo, device: str):
        self.backend = backend
        self.model_info = model_info
        self.device = device
        self.messages: list[dict] = []
        self.system_prompt: str = "You are a helpful assistant."
        self.temperature: float = 0.7
        self.max_tokens: int = 1024

    def load(self):
        console.print(Panel(
            f"[bold]{self.model_info.name}[/]",
            title="Loading Model",
            border_style="cyan",
            padding=(0, 2),
        ))
        self.backend.load(self.model_info, self.device)
        console.print("  [bold green]\u2714 Ready![/]")

    def chat(self, user_input: str):
        self.messages.append({"role": "user", "content": user_input})

        full_messages = [{"role": "system", "content": self.system_prompt}] + self.messages

        full_response = []
        stopped = False
        console.print()

        with Live(Text("\u258d", style="bold cyan"), console=console, refresh_per_second=15) as live:
            for chunk in self.backend.generate_stream(
                full_messages, self.temperature, self.max_tokens
            ):
                full_response.append(chunk)
                raw = "".join(full_response)

                # Safety-net: stop if the model emits a turn boundary token
                # that wasn't caught by the EOS token ID list.
                # Find the earliest stop marker to avoid order-dependent bugs.
                stop_positions = [raw.index(s) for s in _STOP_STRINGS if s in raw]
                if stop_positions:
                    raw = raw[:min(stop_positions)]
                    full_response = [raw]
                    stopped = True

                # Strip thinking blocks before display
                display = strip_thinking(raw)

                try:
                    live.update(Markdown(display + " \u258d"))
                except Exception:
                    live.update(Text(display + " \u258d"))

                if stopped:
                    break

        response_text = strip_thinking("".join(full_response)).strip()
        self.messages.append({"role": "assistant", "content": response_text})

    def clear(self):
        self.messages.clear()
        console.print("[dim]Chat history cleared.[/]")


def handle_command(cmd: str, session: ChatSession) -> str | None:
    """Handle a slash command. Returns 'quit', 'switch', or None."""
    parts = cmd.strip().split(maxsplit=1)
    command = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if command in ("/quit", "/exit", "/q"):
        return "quit"
    elif command == "/help":
        console.print(Panel(HELP_TEXT, title="Help", border_style="cyan"))
    elif command == "/clear":
        session.clear()
    elif command == "/system":
        if arg:
            session.system_prompt = arg
            console.print(f"[dim]System prompt set to:[/] {arg}")
        else:
            console.print(f"[dim]Current system prompt:[/] {session.system_prompt}")
    elif command == "/temp":
        try:
            val = float(arg)
            if 0.0 <= val <= 2.0:
                session.temperature = val
                console.print(f"[dim]Temperature set to:[/] {val}")
            else:
                console.print("[red]Temperature must be between 0.0 and 2.0[/]")
        except ValueError:
            console.print(f"[dim]Current temperature:[/] {session.temperature}")
    elif command == "/tokens":
        try:
            val = int(arg)
            if val > 0:
                session.max_tokens = val
                console.print(f"[dim]Max tokens set to:[/] {val}")
            else:
                console.print("[red]Must be a positive integer[/]")
        except ValueError:
            console.print(f"[dim]Current max tokens:[/] {session.max_tokens}")
    elif command == "/model":
        return "switch"
    elif command == "/info":
        tbl = Table(show_header=False, box=None, padding=(0, 2))
        tbl.add_row("[dim]Model[/]", session.model_info.name)
        tbl.add_row("[dim]Backend[/]", session.backend.name)
        tbl.add_row("[dim]Device[/]", session.device)
        tbl.add_row("[dim]System prompt[/]", session.system_prompt[:60] + ("..." if len(session.system_prompt) > 60 else ""))
        tbl.add_row("[dim]Temperature[/]", str(session.temperature))
        tbl.add_row("[dim]Max tokens[/]", str(session.max_tokens))
        tbl.add_row("[dim]History[/]", f"{len(session.messages)} messages")
        console.print(Panel(tbl, title="Session Info", border_style="cyan"))
    else:
        console.print(f"[red]Unknown command:[/] {command}  [dim](type /help)[/]")

    return None
