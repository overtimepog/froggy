"""Chat session and slash command handling."""

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .backends import Backend
from .discovery import ModelInfo

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
        console.print()

        with Live(Text("\u258d", style="bold cyan"), console=console, refresh_per_second=15) as live:
            for chunk in self.backend.generate_stream(
                full_messages, self.temperature, self.max_tokens
            ):
                full_response.append(chunk)
                text = "".join(full_response)
                try:
                    live.update(Markdown(text + " \u258d"))
                except Exception:
                    live.update(Text(text + " \u258d"))

        response_text = "".join(full_response).strip()
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
