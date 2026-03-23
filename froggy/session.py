"""Chat session and slash command handling."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import time
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from .backends import Backend
from .context import (
    AGENT_SYSTEM_PROMPT,
    ContextManager,
    context_limit_for_model,
    estimate_messages_tokens,
)
from .discovery import ModelInfo

# ---------------------------------------------------------------------------
# Optional tool system imports — gracefully degrade when not installed.
# ---------------------------------------------------------------------------

try:
    from .tool_executor import ToolExecutor
    from .tool_parser import ToolCallParser
    from .tools import CORE_TOOLS, ToolDef, ToolRegistry  # noqa: F401
    _TOOLS_AVAILABLE = True
except ImportError:
    _TOOLS_AVAILABLE = False
    ToolRegistry = None  # type: ignore[assignment,misc]
    ToolDef = None  # type: ignore[assignment,misc]
    CORE_TOOLS = []  # type: ignore[assignment]
    ToolCallParser = None  # type: ignore[assignment,misc]
    ToolExecutor = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Environment variable configuration
# ---------------------------------------------------------------------------

def _env_bool(key: str) -> bool:
    return os.environ.get(key, "").lower() in ("1", "true", "yes")


def _env_int(key: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(key, str(default))))
    except ValueError:
        return default


FROGGY_TOOLS = _env_bool("FROGGY_TOOLS")
FROGGY_AUTORUN = _env_bool("FROGGY_AUTORUN")
FROGGY_PROJECT_ROOT = os.environ.get("FROGGY_PROJECT_ROOT", "")
FROGGY_MAX_TOOL_ROUNDS = _env_int("FROGGY_MAX_TOOL_ROUNDS", 5)

# ---------------------------------------------------------------------------
# Thinking-block and stop-string helpers
# ---------------------------------------------------------------------------

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
  [cyan]/help[/]                  Show this help
  [cyan]/quit[/]                  Exit
  [cyan]/clear[/]                 Clear chat history
  [cyan]/system <prompt>[/]       Set system prompt
  [cyan]/temp <value>[/]          Set temperature (0.0 - 2.0)
  [cyan]/tokens <value>[/]        Set max response tokens
  [cyan]/model[/]                 Switch to a different model
  [cyan]/info[/]                  Show current settings
  [cyan]/context[/]               Show context window usage and token stats
  [cyan]/inject <path>[/]         Inject a file into the context window
  [cyan]/eject <path>[/]          Remove an injected file from context
  [cyan]/profile <name>[/]        Set context profile (minimal/standard/full)
  [cyan]/fresh[/]                 Reset context (keep injected files, clear history)
  [cyan]/tools[/]                 List available tools
  [cyan]/tools on|off[/]          Enable or disable the tool system
  [cyan]/tools add <name>[/]      Activate a specific tool
  [cyan]/tools remove <name>[/]   Deactivate a specific tool
  [cyan]/autorun[/]               Toggle auto-approve for tool calls"""

# ---------------------------------------------------------------------------
# Custom tool loader
# ---------------------------------------------------------------------------


def load_custom_tools(tools_dir: Path) -> list:
    """Scan *tools_dir* for custom ToolDef definitions.

    Each ``*.py`` file (not starting with ``_``) may export:
    - ``TOOLS``: a list of ``ToolDef`` objects, or
    - ``TOOL``: a single ``ToolDef`` object.

    Returns a flat list of ``ToolDef`` objects.  Silently skips files that
    fail to import or that export neither attribute.
    """
    if not _TOOLS_AVAILABLE or not tools_dir.is_dir():
        return []

    custom: list = []
    for py_file in sorted(tools_dir.glob("*.py")):
        if py_file.name.startswith("_"):
            continue
        try:
            spec = importlib.util.spec_from_file_location(
                f"_froggy_custom_{py_file.stem}", py_file
            )
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
            if hasattr(mod, "TOOLS"):
                for t in mod.TOOLS:
                    if ToolDef is not None and isinstance(t, ToolDef):
                        custom.append(t)
            elif hasattr(mod, "TOOL"):
                if ToolDef is not None and isinstance(mod.TOOL, ToolDef):
                    custom.append(mod.TOOL)
        except Exception:
            # Silently skip broken custom tool files so a bad plugin
            # doesn't crash the whole session.
            pass

    return custom


# ---------------------------------------------------------------------------
# ChatSession
# ---------------------------------------------------------------------------


class ChatSession:
    def __init__(
        self,
        backend: Backend,
        model_info: ModelInfo,
        device: str,
        *,
        tool_registry=None,
        tool_executor=None,
    ):
        self.backend = backend
        self.model_info = model_info
        self.device = device
        self.messages: list[dict] = []
        self.system_prompt: str = AGENT_SYSTEM_PROMPT
        self.temperature: float = 0.7
        self.max_tokens: int = 1024

        # Context management
        ctx_limit = context_limit_for_model(
            model_info.name,
            getattr(model_info, "context_length", None),
        )
        self.context_mgr = ContextManager(
            context_limit=ctx_limit,
            model=model_info.name,
        )

        # Tool system state
        self.tools_enabled: bool = FROGGY_TOOLS and _TOOLS_AVAILABLE
        self.autorun: bool = FROGGY_AUTORUN
        self.max_tool_rounds: int = FROGGY_MAX_TOOL_ROUNDS
        self._registry = tool_registry      # ToolRegistry or None
        self._executor = tool_executor      # ToolExecutor or None
        self._active_tool_names: list[str] | None = None  # None = all tools

    # ------------------------------------------------------------------
    # Tool helpers
    # ------------------------------------------------------------------

    def _get_active_registry(self):
        """Return a ToolRegistry restricted to currently active tools."""
        if self._registry is None:
            return None
        if self._active_tool_names is None:
            return self._registry
        if _TOOLS_AVAILABLE and ToolRegistry is not None:
            active = [t for t in self._registry.all() if t.name in self._active_tool_names]
            return ToolRegistry(tools=active)
        return self._registry

    def _build_system_prompt(self) -> str:
        """Build the effective system prompt via the context manager.

        Assembles: base prompt + tool instructions + injected files/blocks +
        conversation summary.  The context manager controls what gets inlined
        based on the active profile.
        """
        tool_block = None
        if self.tools_enabled and self._registry is not None:
            reg = self._get_active_registry()
            if reg is not None:
                tool_block = reg.system_prompt_block()

        return self.context_mgr.build_system_context(
            self.system_prompt, tool_block=tool_block
        )

    def _sync_autorun(self) -> None:
        """Propagate autorun state to the executor's confirm_fn."""
        if self._executor is None or not _TOOLS_AVAILABLE:
            return
        if self.autorun:
            self._executor._confirm_fn = lambda desc, risk: True
        else:
            self._executor._confirm_fn = ToolExecutor._default_confirm  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Core session operations
    # ------------------------------------------------------------------

    def load(self):
        console.print(Panel(
            f"[bold]{self.model_info.name}[/]",
            title="Loading Model",
            border_style="cyan",
            padding=(0, 2),
        ))
        self.backend.load(self.model_info, self.device)
        console.print("  [bold green]\u2714 Ready![/]")

    def chat(self, user_input: str) -> None:
        """Send *user_input*, run the tool loop, append final assistant message."""
        self.messages.append({"role": "user", "content": user_input})

        for round_num in range(self.max_tool_rounds):
            response_text, tool_calls = self._generate_one_round()

            if tool_calls and self.tools_enabled and self._executor is not None:
                # Append assistant turn (may contain partial text before tool call)
                self.messages.append({"role": "assistant", "content": response_text})

                # Execute each tool call and inject result as a user message
                for call in tool_calls:
                    result = self._executor.execute(call.name, **call.arguments)
                    result_json = json.dumps(result)
                    status_icon = "[green]\u2713[/]" if result.get("ok") else "[red]\u2717[/]"
                    # Show tool name and a preview of the result
                    preview = ""
                    if result.get("ok") and result.get("output"):
                        out = str(result["output"])
                        preview = f" [dim]({len(out)} chars)[/]" if len(out) > 100 else ""
                    console.print(f"  [dim]\u2192 Tool:[/] [cyan]{call.name}[/] {status_icon}{preview}")
                    self.messages.append({
                        "role": "user",
                        "content": f"<tool_response>\n{result_json}\n</tool_response>",
                    })

                if round_num == self.max_tool_rounds - 1:
                    console.print("[dim]Max tool rounds reached.[/]")
                # Loop: let the model consume tool results and optionally call more tools.
            else:
                # No tool calls (or tools disabled) — this is the final response.
                self.messages.append({"role": "assistant", "content": response_text})

                # Track token usage for this exchange
                from .context import estimate_tokens
                prompt_tokens = estimate_messages_tokens(self.messages, self.model_info.name)
                completion_tokens = estimate_tokens(response_text, self.model_info.name)
                self.context_mgr.usage.record(prompt_tokens, completion_tokens)

                # Warn if context is getting full
                full_msgs = [{"role": "system", "content": self._build_system_prompt()}] + self.messages
                util = self.context_mgr.count_tokens(full_msgs) / self.context_mgr.available_tokens if self.context_mgr.available_tokens > 0 else 0
                if util > 0.85:
                    console.print(f"[yellow]  ⚠ Context {util:.0%} full — older messages may be trimmed soon[/]")
                elif util > 0.6:
                    console.print(f"[dim]  Context: {util:.0%}[/]")
                break

    def _generate_one_round(self) -> tuple[str, list]:
        """Stream one generation round.

        Returns ``(response_text, tool_calls)`` where ``tool_calls`` is a
        (possibly empty) list of ``ToolCall`` objects extracted from the stream.
        """
        # Auto-summarize old turns, then trim if still over budget
        self.messages = self.context_mgr.maybe_summarize(self.messages)
        full_messages = [
            {"role": "system", "content": self._build_system_prompt()}
        ] + self.messages
        full_messages = self.context_mgr.trim_if_needed(full_messages)
        # Sync back: messages is everything after the system prompt
        self.messages = full_messages[1:]

        full_response: list[str] = []
        stopped = False
        console.print()

        # Streaming tool-call parser — buffers chunks and detects tool calls.
        parser = None
        streaming_calls: list = []
        if self.tools_enabled and _TOOLS_AVAILABLE and ToolCallParser is not None:
            parser = ToolCallParser()

        first_token = True
        token_count = 0
        t_start = time.time()
        t_first_token = t_start

        # Phase 1: Show thinking spinner until first token arrives
        with Live(
            Spinner("dots", text="[dim]Thinking...[/]", style="cyan"),
            console=console,
            refresh_per_second=15,
        ) as live:
            for chunk in self.backend.generate_stream(
                full_messages, self.temperature, self.max_tokens
            ):
                full_response.append(chunk)
                token_count += 1
                raw = "".join(full_response)

                if first_token:
                    t_first_token = time.time()
                    first_token = False

                # Safety-net: stop if the model emits a turn boundary token
                stop_positions = [raw.index(s) for s in _STOP_STRINGS if s in raw]
                if stop_positions:
                    raw = raw[:min(stop_positions)]
                    full_response = [raw]
                    stopped = True

                # Feed chunk to the streaming tool-call buffer.
                if parser is not None:
                    streaming_calls.extend(parser.feed(chunk))

                # Strip thinking blocks before display
                display = strip_thinking(raw)

                # Switch from spinner to streaming text after first token
                try:
                    live.update(Markdown(display + " \u258d"))
                except Exception:
                    live.update(Text(display + " \u258d"))

                if stopped:
                    break

        # Show generation stats
        t_end = time.time()
        elapsed = t_end - t_start
        ttft = t_first_token - t_start
        tps = token_count / elapsed if elapsed > 0 else 0
        if token_count > 0:
            stats_parts = []
            if ttft < elapsed:
                stats_parts.append(f"first token: {ttft:.1f}s")
            stats_parts.append(f"{token_count} tokens")
            stats_parts.append(f"{tps:.1f} tok/s")
            stats_parts.append(f"{elapsed:.1f}s")
            console.print(f"[dim]  {' · '.join(stats_parts)}[/]")

        # Collect any remaining tool calls that were buffered during streaming.
        tool_calls: list = list(streaming_calls)
        if parser is not None:
            tool_calls.extend(parser.flush())

        response_text = strip_thinking("".join(full_response)).strip()
        return response_text, tool_calls

    def clear(self) -> None:
        self.messages.clear()
        console.print("[dim]Chat history cleared.[/]")


# ---------------------------------------------------------------------------
# Slash command helpers
# ---------------------------------------------------------------------------


def _list_tools(session: ChatSession) -> None:
    """Print the tools table for the current session."""
    if not _TOOLS_AVAILABLE or session._registry is None:
        console.print("[dim]Tool system not available.[/]")
        return

    tbl = Table(border_style="cyan", title_style="bold cyan")
    tbl.add_column("Name", style="cyan", no_wrap=True)
    tbl.add_column("Description")
    tbl.add_column("Active", style="bold", justify="center", width=6)

    active = session._active_tool_names
    for tool in session._registry.all():
        is_active = active is None or tool.name in active
        tbl.add_row(
            tool.name,
            tool.description,
            "[green]yes[/]" if is_active else "[dim]no[/]",
        )

    enabled_label = "[green]enabled[/]" if session.tools_enabled else "[dim]disabled[/]"
    console.print(Panel(tbl, title=f"Tools ({enabled_label})", border_style="cyan"))


def _handle_tools_command(arg: str, session: ChatSession) -> None:
    """Dispatch /tools subcommands."""
    if session._registry is None:
        console.print("[dim]Tool system not available (froggy tool modules not installed).[/]")
        return

    parts = arg.strip().split(maxsplit=1) if arg.strip() else []
    subcmd = parts[0].lower() if parts else ""
    subarg = parts[1].strip() if len(parts) > 1 else ""

    if subcmd == "on":
        session.tools_enabled = True
        console.print("[dim]Tools enabled.[/]")
    elif subcmd == "off":
        session.tools_enabled = False
        console.print("[dim]Tools disabled.[/]")
    elif subcmd == "add":
        if not subarg:
            console.print("[red]Usage: /tools add <name>[/]")
            return
        if session._registry.get(subarg) is None:
            console.print(f"[red]Unknown tool:[/] {subarg}")
            return
        if session._active_tool_names is None:
            session._active_tool_names = list(session._registry.names())
        if subarg not in session._active_tool_names:
            session._active_tool_names.append(subarg)
        console.print(f"[dim]Tool activated:[/] {subarg}")
    elif subcmd == "remove":
        if not subarg:
            console.print("[red]Usage: /tools remove <name>[/]")
            return
        if session._active_tool_names is None:
            session._active_tool_names = list(session._registry.names())
        if subarg in session._active_tool_names:
            session._active_tool_names.remove(subarg)
            console.print(f"[dim]Tool deactivated:[/] {subarg}")
        else:
            console.print(f"[dim]Tool not active:[/] {subarg}")
    else:
        _list_tools(session)


# ---------------------------------------------------------------------------
# Top-level command handler
# ---------------------------------------------------------------------------


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
        ctx = session.context_mgr.status(
            [{"role": "system", "content": session._build_system_prompt()}] + session.messages
        )
        pct = ctx['utilization']
        color = "green" if pct < 0.6 else "yellow" if pct < 0.85 else "red"
        tbl.add_row("[dim]Context[/]", f"[{color}]{ctx['current_tokens']:,} / {ctx['available_tokens']:,} tokens ({pct:.0%})[/]")
        tbl.add_row("[dim]Profile[/]", ctx['profile'])
        if ctx["injected_items"] > 0:
            tbl.add_row("[dim]Injected[/]", f"{ctx['injected_items']} items ({ctx['injected_tokens']:,} tokens)")
        if ctx["trims"] > 0 or ctx["summarizations"] > 0:
            tbl.add_row("[dim]Compressions[/]", f"{ctx['summarizations']} summaries, {ctx['trims']} trims")
        if _TOOLS_AVAILABLE:
            tools_state = "enabled" if session.tools_enabled else "disabled"
            tbl.add_row("[dim]Tools[/]", tools_state)
            tbl.add_row("[dim]Autorun[/]", "on" if session.autorun else "off")
        console.print(Panel(tbl, title="Session Info", border_style="cyan"))
    elif command == "/context":
        full_msgs = [{"role": "system", "content": session._build_system_prompt()}] + session.messages
        ctx = session.context_mgr.status(full_msgs)
        tbl = Table(show_header=False, box=None, padding=(0, 2))
        tbl.add_row("[dim]Profile[/]", ctx['profile'])
        tbl.add_row("[dim]Context limit[/]", f"{ctx['context_limit']:,} tokens")
        tbl.add_row("[dim]Available (after reserve)[/]", f"{ctx['available_tokens']:,} tokens")
        tbl.add_row("[dim]Current usage[/]", f"{ctx['current_tokens']:,} tokens")
        pct = ctx['utilization']
        color = "green" if pct < 0.6 else "yellow" if pct < 0.85 else "red"
        tbl.add_row("[dim]Utilization[/]", f"[{color}]{pct:.0%}[/]")
        tbl.add_row("[dim]Messages[/]", str(ctx['messages']))
        tbl.add_row("[dim]Injected items[/]", f"{ctx['injected_items']} ({ctx['injected_tokens']:,} tokens)")
        tbl.add_row("[dim]Summarizations[/]", str(ctx['summarizations']))
        tbl.add_row("[dim]Auto-trims[/]", str(ctx['trims']))
        tbl.add_row("[dim]Has summary[/]", "yes" if ctx['has_summary'] else "no")
        tbl.add_row("[dim]Session usage[/]", ctx['usage'])
        # List injected items
        injected = session.context_mgr.list_injected()
        if injected:
            tbl.add_row("", "")
            for name, tokens in injected.items():
                tbl.add_row(f"  [dim]{name}[/]", f"{tokens:,} tokens")
        console.print(Panel(tbl, title="Context Window", border_style="cyan"))
    elif command == "/inject":
        if not arg:
            console.print("[red]Usage: /inject <file_path>[/]")
        else:
            ok = session.context_mgr.inject_file(arg.strip())
            if ok:
                tokens = session.context_mgr.list_injected().get(f"[file] {arg.strip()}", 0)
                console.print(f"[dim]Injected:[/] {arg.strip()} [dim]({tokens:,} tokens)[/]")
            else:
                console.print(f"[red]Could not read:[/] {arg.strip()}")
    elif command == "/eject":
        if not arg:
            console.print("[red]Usage: /eject <file_path>[/]")
        else:
            ok = session.context_mgr.remove_file(arg.strip())
            if ok:
                console.print(f"[dim]Ejected:[/] {arg.strip()}")
            else:
                console.print(f"[dim]Not found in context:[/] {arg.strip()}")
    elif command == "/profile":
        if not arg:
            console.print(f"[dim]Current profile:[/] {session.context_mgr.profile_name}")
            from .context import PROFILES
            for name, p in PROFILES.items():
                marker = " [bold cyan]← active[/]" if name == session.context_mgr.profile_name else ""
                console.print(f"  [cyan]{name}[/] — {p['description']}{marker}")
        else:
            ok = session.context_mgr.set_profile(arg.strip())
            if ok:
                console.print(f"[dim]Profile set to:[/] {arg.strip()}")
            else:
                console.print(f"[red]Unknown profile:[/] {arg.strip()} [dim](minimal/standard/full)[/]")
    elif command == "/fresh":
        session.context_mgr.fresh_context()
        session.messages.clear()
        console.print("[dim]Context reset — injected files preserved, history cleared.[/]")
    elif command == "/tools":
        _handle_tools_command(arg, session)
    elif command == "/autorun":
        session.autorun = not session.autorun
        session._sync_autorun()
        state = "on" if session.autorun else "off"
        console.print(f"[dim]Autorun is now:[/] {state}")
    else:
        console.print(f"[red]Unknown command:[/] {command}  [dim](type /help)[/]")

    return None
