"""Tool executor with safety model, path validation, and result formatting.

Three-layer safety model (spec §8):
  Layer 1 — Path & command validation (always on)
  Layer 2 — Confirmation prompts (interactive)
  Layer 3 — OS sandbox (optional, macOS sandbox-exec)
"""

from __future__ import annotations

import io
import json
import math
import os
import platform
import re
import shlex
import subprocess
import sys
import tempfile
import textwrap
from enum import Enum
from pathlib import Path, PurePath
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Safety tiers & risk levels
# ---------------------------------------------------------------------------


class SafetyTier(Enum):
    AUTO_APPROVE = "auto_approve"
    CONFIRM = "confirm"
    BLOCKED = "blocked"


class RiskLevel(Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    BLOCKED = "blocked"


# ---------------------------------------------------------------------------
# Path validation (Layer 1)
# ---------------------------------------------------------------------------

# Glob-style sensitive path patterns checked against the resolved path string.
_SENSITIVE_PATTERNS: list[re.Pattern] = [
    re.compile(r"(^|[\\/])\.ssh[\\/]"),
    re.compile(r"(^|[\\/])\.aws[\\/]"),
    re.compile(r"(^|[\\/])\.gnupg[\\/]"),
    re.compile(r"(^|[\\/])\.env($|[\\/])"),
    re.compile(r"\.env\b", re.IGNORECASE),
    re.compile(r"credentials", re.IGNORECASE),
    re.compile(r"secret", re.IGNORECASE),
    re.compile(r"\.pem$", re.IGNORECASE),
    re.compile(r"\.key$", re.IGNORECASE),
    re.compile(r"id_rsa"),
    re.compile(r"id_ed25519"),
    re.compile(r"/etc/shadow"),
    re.compile(r"/etc/passwd"),
    re.compile(r"/etc/sudoers"),
]


def _is_sensitive(path: Path) -> bool:
    path_str = str(path)
    return any(pat.search(path_str) for pat in _SENSITIVE_PATTERNS)


def validate_path(
    raw: str,
    project_root: str | Path,
    *,
    allow_write: bool = False,
) -> tuple[bool, str, Path | None]:
    """
    Validate a file path relative to *project_root*.

    Returns (ok, reason, resolved_path).
    If ok is False, resolved_path is None and reason explains the rejection.
    """
    root = Path(project_root).resolve()

    # Resolve: relative paths are anchored at project_root; absolute allowed.
    try:
        candidate = Path(raw)
        if candidate.is_absolute():
            resolved = candidate.resolve()
        else:
            resolved = (root / candidate).resolve()
    except Exception as exc:
        return False, f"Cannot resolve path: {exc}", None

    # Block path traversal (resolved path must stay inside project_root)
    try:
        resolved.relative_to(root)
    except ValueError:
        return False, f"Path traversal blocked: '{raw}' escapes project root", None

    # Block sensitive path patterns
    if _is_sensitive(resolved):
        return False, f"Sensitive path blocked: {resolved}", None

    # Block system directories for writes
    if allow_write:
        system_dirs = ["/etc", "/usr", "/bin", "/sbin", "/lib", "/boot",
                       "/private/etc", "/private/var/db"]
        for sd in system_dirs:
            if str(resolved).startswith(sd + os.sep) or str(resolved) == sd:
                return False, f"Writing to system directory blocked: {sd}", None

    return True, "", resolved


# ---------------------------------------------------------------------------
# Command classification (Layer 1)
# ---------------------------------------------------------------------------

# Regex patterns that are always blocked regardless of executable.
_BLOCKED_CMD_RE = re.compile(
    r"(sudo\b|rm\s+-rf\s+/|chmod\s+777|curl[^|]*\|\s*(ba)?sh"
    r"|wget[^|]*\|\s*(ba)?sh"
    r"|\beval\s"
    r"|mkfs\b|dd\s+if="
    r"|>\s*/dev/"
    r"|\bfork\s*bomb\b"
    r"|\bnc\b|\bnetcat\b|\bncat\b"
    r"|\bssh\b|\bscp\b|\bsftp\b)",
    re.IGNORECASE,
)

# Executables that are auto-approved when they don't match a danger pattern.
_AUTO_APPROVE_EXES: frozenset[str] = frozenset({
    "git", "python", "python3", "pytest", "ruff",
    "ls", "cat", "head", "tail", "wc", "grep", "find", "diff",
    "echo", "pwd", "which", "sed", "awk", "sort", "uniq", "cut",
    "pip", "pip3", "env", "printenv", "date", "whoami", "true", "false",
    "test",
})

# Patterns inside otherwise-safe commands that bump tier to CONFIRM.
_DANGER_INSIDE_RE = re.compile(
    r"(>\s*/|>\s*~/|\|\s*(ba)?sh|\|\s*zsh|;\s*rm|&&\s*rm)",
)


def classify_command(cmd: str) -> tuple[SafetyTier, RiskLevel]:
    """Classify a shell command string into (SafetyTier, RiskLevel)."""
    # Immediately blocked patterns
    if _BLOCKED_CMD_RE.search(cmd):
        return SafetyTier.BLOCKED, RiskLevel.BLOCKED

    try:
        tokens = shlex.split(cmd)
    except ValueError:
        return SafetyTier.BLOCKED, RiskLevel.BLOCKED

    if not tokens:
        return SafetyTier.BLOCKED, RiskLevel.BLOCKED

    exe = Path(tokens[0]).name.lower()

    if exe in _AUTO_APPROVE_EXES:
        if _DANGER_INSIDE_RE.search(cmd):
            return SafetyTier.CONFIRM, RiskLevel.MEDIUM
        return SafetyTier.AUTO_APPROVE, RiskLevel.LOW

    # Unknown executable — ask user
    return SafetyTier.CONFIRM, RiskLevel.HIGH


# ---------------------------------------------------------------------------
# python_eval safety classification
# ---------------------------------------------------------------------------

_PYTHON_DANGEROUS_RE = re.compile(
    r"(\bimport\b|\bopen\b\s*\(|\bsubprocess\b|\bos\.system\b"
    r"|\beval\b\s*\(|\bexec\b\s*\(|__)",
)


def classify_python(code: str) -> tuple[SafetyTier, RiskLevel]:
    """Determine the safety tier for a python_eval code snippet."""
    if _PYTHON_DANGEROUS_RE.search(code):
        return SafetyTier.CONFIRM, RiskLevel.HIGH
    return SafetyTier.CONFIRM, RiskLevel.MEDIUM


# ---------------------------------------------------------------------------
# Token-budget truncation
# ---------------------------------------------------------------------------

_CHARS_PER_TOKEN = 4  # rough approximation


def truncate_to_budget(text: str, token_budget: int) -> str:
    """Symmetrically truncate *text* to stay within the token budget."""
    if token_budget <= 0:
        return ""
    char_limit = token_budget * _CHARS_PER_TOKEN
    if len(text) <= char_limit:
        return text
    half = char_limit // 2
    omitted = len(text) - char_limit
    return (
        text[:half]
        + f"\n[...truncated, {omitted} chars omitted]\n"
        + text[-half:]
    )


def format_result(
    tool_name: str,
    output: str,
    *,
    token_budget: int = 2000,
    error: bool = False,
) -> dict[str, Any]:
    """Return a structured result dict, output truncated to the token budget."""
    truncated = truncate_to_budget(output, token_budget)
    return {
        "tool": tool_name,
        "ok": not error,
        "output": truncated,
        "truncated": len(truncated) < len(output),
    }


def format_tool_result(tool_name: str, result: str, max_tokens: int = 800) -> dict:
    """Format a tool result as a chat message for re-injection (spec §7)."""
    truncated = truncate_to_budget(result, max_tokens)
    if len(result) > len(truncated):
        truncated += f"\n[...truncated, {len(result) - len(truncated)} chars omitted]"
    return {
        "role": "tool",
        "content": (
            f"<tool_response>\n"
            f'{{"name": "{tool_name}", "content": {json.dumps(truncated)}}}\n'
            f"</tool_response>"
        ),
    }


# ---------------------------------------------------------------------------
# OS sandbox helper (Layer 3)
# ---------------------------------------------------------------------------

_SANDBOX_PROFILE_TMPL = textwrap.dedent("""\
    (version 1)
    (deny default)
    (allow file-read* (subpath "{root}"))
    (allow file-write* (subpath "{root}"))
    (allow file-read* (subpath "/usr"))
    (allow file-read* (subpath "/bin"))
    (allow file-read* (subpath "/lib"))
    (allow process-exec (subpath "/usr/bin") (subpath "/usr/local/bin"))
    (allow sysctl-read)
    (allow mach-lookup)
""")


def _run_sandboxed(cmd: str, project_root: str, timeout: int) -> subprocess.CompletedProcess:
    """Run a shell command inside a macOS sandbox-exec profile."""
    profile = _SANDBOX_PROFILE_TMPL.format(root=project_root)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sb", delete=False) as f:
        f.write(profile)
        sb_file = f.name
    try:
        return subprocess.run(
            ["sandbox-exec", "-f", sb_file, "sh", "-c", cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=project_root,
        )
    finally:
        try:
            os.unlink(sb_file)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# ToolExecutor
# ---------------------------------------------------------------------------

# Restricted builtins available to python_eval.
_SAFE_MODULES = {
    "math": math,
    "json": json,
    "re": re,
    "os": type(os)("os"),  # placeholder; only os.path exposed below
    "datetime": __import__("datetime"),
    "collections": __import__("collections"),
    "itertools": __import__("itertools"),
    "functools": __import__("functools"),
    "textwrap": textwrap,
    "PurePath": PurePath,
}

# Expose only os.path, not full os module.
_EVAL_GLOBALS: dict[str, Any] = {
    "__builtins__": {
        # safe built-ins only
        "abs": abs, "all": all, "any": any, "bin": bin, "bool": bool,
        "chr": chr, "dict": dict, "dir": dir, "divmod": divmod,
        "enumerate": enumerate, "filter": filter, "float": float,
        "format": format, "frozenset": frozenset, "getattr": getattr,
        "hasattr": hasattr, "hash": hash, "hex": hex, "int": int,
        "isinstance": isinstance, "issubclass": issubclass,
        "iter": iter, "len": len, "list": list, "map": map,
        "max": max, "min": min, "next": next, "oct": oct, "ord": ord,
        "pow": pow, "print": print, "range": range, "repr": repr,
        "reversed": reversed, "round": round, "set": set, "setattr": setattr,
        "slice": slice, "sorted": sorted, "str": str, "sum": sum,
        "tuple": tuple, "type": type, "vars": vars, "zip": zip,
        "True": True, "False": False, "None": None,
    },
    **_SAFE_MODULES,
    "os_path": os.path,
}


class ToolExecutor:
    """
    Dispatches tool calls by name with a three-layer safety model.

    Parameters
    ----------
    project_root:
        Root directory for all file operations. Paths outside this root are
        rejected. Defaults to the current working directory.
    token_budget:
        Approximate token budget for result output (default 2000).
    use_sandbox:
        Wrap shell commands in macOS sandbox-exec when available.
    confirm_fn:
        ``(description: str, risk: RiskLevel) -> bool`` — called for
        CONFIRM-tier operations.  Defaults to an interactive Rich/readline
        prompt.
    shell_timeout:
        Seconds before a shell command is killed (default 30).
    python_timeout:
        Seconds before python_eval is killed (default 10).
    """

    def __init__(
        self,
        *,
        project_root: str | Path | None = None,
        token_budget: int = 2000,
        use_sandbox: bool = False,
        confirm_fn: Callable[[str, RiskLevel], bool] | None = None,
        shell_timeout: int = 30,
        python_timeout: int = 10,
        mcp_manager: Any | None = None,
    ):
        self.project_root = Path(project_root or os.getcwd()).resolve()
        self.token_budget = token_budget
        self.use_sandbox = use_sandbox
        self.shell_timeout = shell_timeout
        self.python_timeout = python_timeout
        self._confirm_fn = confirm_fn or self._default_confirm
        self._mcp_manager = mcp_manager

        self._tools: dict[str, Any] = {
            "read_file": self._read_file,
            "write_file": self._write_file,
            "edit_file": self._edit_file,
            "run_shell": self._run_shell,
            "web_search": self._web_search,
            "python_eval": self._python_eval,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self, tool_name: str, **kwargs) -> dict[str, Any]:
        """Dispatch a tool by name. Returns a structured result dict."""
        # Check MCP tools first
        if self._mcp_manager is not None and self._mcp_manager.is_mcp_tool(tool_name):
            return self._execute_mcp(tool_name, kwargs)

        fn = self._tools.get(tool_name)
        if fn is None:
            return format_result(tool_name, f"Unknown tool: '{tool_name}'", error=True)
        try:
            return fn(**kwargs)
        except Exception as exc:
            return format_result(tool_name, f"Unexpected error: {exc}", error=True)

    def _execute_mcp(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute an MCP tool via the MCP manager."""
        # MCP tools require user confirmation
        allowed, reason = self._gate(
            SafetyTier.CONFIRM, RiskLevel.MEDIUM, f"MCP tool: {tool_name}"
        )
        if not allowed:
            return format_result(tool_name, reason, error=True)
        try:
            result = self._mcp_manager.call_tool(tool_name, arguments)
            return format_result(tool_name, result, token_budget=self.token_budget)
        except Exception as exc:
            return format_result(tool_name, f"MCP error: {exc}", error=True)

    # Alias used by session integration (spec §9)
    def run(self, tool_call: Any) -> str:
        """Execute a ToolCall object; return the raw output string."""
        result = self.execute(tool_call.name, **tool_call.arguments)
        return result["output"]

    @property
    def available_tools(self) -> list[str]:
        return list(self._tools.keys())

    # ------------------------------------------------------------------
    # Confirmation gate
    # ------------------------------------------------------------------

    @staticmethod
    def _default_confirm(description: str, risk: RiskLevel) -> bool:
        try:
            from rich.prompt import Confirm
            return Confirm.ask(f"[yellow]Allow {risk.value}-risk:[/] {description}")
        except ImportError:
            answer = input(f"Allow {risk.value}-risk: {description}? [y/N] ")
            return answer.strip().lower() in ("y", "yes")

    def _gate(self, tier: SafetyTier, risk: RiskLevel, description: str) -> tuple[bool, str]:
        """Apply the safety gate. Returns (allowed, denial_reason)."""
        if tier == SafetyTier.BLOCKED:
            return False, f"Blocked by safety policy: {description}"
        if tier == SafetyTier.AUTO_APPROVE:
            return True, ""
        approved = self._confirm_fn(description, risk)
        return (True, "") if approved else (False, "User declined the operation")

    # ------------------------------------------------------------------
    # Tool: read_file
    # ------------------------------------------------------------------

    def _read_file(
        self, path: str, *, offset: int = 0, limit: int = 200, encoding: str = "utf-8"
    ) -> dict[str, Any]:
        ok, reason, resolved = validate_path(path, self.project_root)
        if not ok:
            return format_result("read_file", reason, error=True)

        # Safety tier: auto-approve (read-only)
        allowed, reason = self._gate(SafetyTier.AUTO_APPROVE, RiskLevel.NONE, f"read_file: {path}")
        if not allowed:
            return format_result("read_file", reason, error=True)

        try:
            lines = resolved.read_text(encoding=encoding).splitlines()
        except FileNotFoundError:
            return format_result("read_file", f"File not found: {path}", error=True)
        except PermissionError:
            return format_result("read_file", f"Permission denied: {path}", error=True)
        except Exception as exc:
            return format_result("read_file", str(exc), error=True)

        sliced = lines[offset: offset + limit]
        numbered = "\n".join(f"{offset + i + 1:>4}  {line}" for i, line in enumerate(sliced))
        return format_result("read_file", numbered, token_budget=self.token_budget)

    # ------------------------------------------------------------------
    # Tool: write_file
    # ------------------------------------------------------------------

    def _write_file(self, path: str, content: str, *, encoding: str = "utf-8") -> dict[str, Any]:
        ok, reason, resolved = validate_path(path, self.project_root, allow_write=True)
        if not ok:
            return format_result("write_file", reason, error=True)

        overwriting = resolved.exists()
        tier = SafetyTier.CONFIRM if overwriting else SafetyTier.CONFIRM
        risk = RiskLevel.MEDIUM
        desc = f"write_file: {path}" + (" (overwrite)" if overwriting else " (new file)")

        allowed, reason = self._gate(tier, risk, desc)
        if not allowed:
            return format_result("write_file", reason, error=True)

        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content, encoding=encoding)
        except PermissionError:
            return format_result("write_file", f"Permission denied: {path}", error=True)
        except Exception as exc:
            return format_result("write_file", str(exc), error=True)

        return format_result("write_file", f"Written {len(content)} bytes to {path}")

    # ------------------------------------------------------------------
    # Tool: edit_file
    # ------------------------------------------------------------------

    def _edit_file(self, path: str, old_string: str, new_string: str) -> dict[str, Any]:
        ok, reason, resolved = validate_path(path, self.project_root, allow_write=True)
        if not ok:
            return format_result("edit_file", reason, error=True)

        allowed, reason = self._gate(
            SafetyTier.CONFIRM, RiskLevel.MEDIUM, f"edit_file: {path}"
        )
        if not allowed:
            return format_result("edit_file", reason, error=True)

        try:
            original = resolved.read_text()
        except FileNotFoundError:
            return format_result("edit_file", f"File not found: {path}", error=True)
        except Exception as exc:
            return format_result("edit_file", str(exc), error=True)

        count = original.count(old_string)
        if count == 0:
            return format_result("edit_file", "old_string not found in file", error=True)
        if count > 1:
            return format_result(
                "edit_file",
                f"old_string appears {count} times — provide more context to make it unique",
                error=True,
            )

        updated = original.replace(old_string, new_string, 1)
        try:
            resolved.write_text(updated)
        except PermissionError:
            return format_result("edit_file", f"Permission denied writing: {path}", error=True)
        except Exception as exc:
            return format_result("edit_file", str(exc), error=True)

        return format_result("edit_file", f"Edit applied to {path}")

    # ------------------------------------------------------------------
    # Tool: run_shell
    # ------------------------------------------------------------------

    def _run_shell(self, cmd: str) -> dict[str, Any]:
        tier, risk = classify_command(cmd)
        allowed, reason = self._gate(tier, risk, f"run_shell: {cmd}")
        if not allowed:
            return format_result("run_shell", reason, error=True)

        try:
            if self.use_sandbox and platform.system() == "Darwin":
                proc = _run_sandboxed(cmd, str(self.project_root), self.shell_timeout)
            else:
                proc = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=self.shell_timeout,
                    cwd=str(self.project_root),
                )
        except subprocess.TimeoutExpired:
            return format_result(
                "run_shell", f"Command timed out after {self.shell_timeout}s", error=True
            )
        except Exception as exc:
            return format_result("run_shell", str(exc), error=True)

        parts = []
        if proc.stdout:
            parts.append(proc.stdout)
        if proc.stderr:
            parts.append(f"[stderr]\n{proc.stderr}")
        combined = "\n".join(parts)

        if proc.returncode != 0:
            combined += f"\n[exit code: {proc.returncode}]"
            return format_result("run_shell", combined, error=True, token_budget=self.token_budget)

        return format_result("run_shell", combined, token_budget=self.token_budget)

    # ------------------------------------------------------------------
    # Tool: web_search
    # ------------------------------------------------------------------

    def _web_search(self, query: str) -> dict[str, Any]:
        allowed, reason = self._gate(
            SafetyTier.AUTO_APPROVE, RiskLevel.NONE, f"web_search: {query}"
        )
        if not allowed:
            return format_result("web_search", reason, error=True)

        results = self._do_web_search(query)
        if results is None:
            return format_result(
                "web_search",
                "No search backend available (install duckduckgo_search or set BRAVE_API_KEY)",
                error=True,
            )
        return format_result("web_search", results, token_budget=self.token_budget)

    def _do_web_search(self, query: str) -> str | None:
        """
        Try search backends in priority order:
          1. SearXNG local instance (http://localhost:8888)
          2. DuckDuckGo (duckduckgo_search package)
          3. Brave Search API (BRAVE_API_KEY env var)
        """
        # 1. SearXNG
        try:
            import urllib.parse
            import urllib.request

            url = "http://localhost:8888/search?" + urllib.parse.urlencode({
                "q": query, "format": "json", "results": 5
            })
            with urllib.request.urlopen(url, timeout=5) as resp:
                data = json.loads(resp.read())
            items = data.get("results", [])[:5]
            return self._format_search_results(items, title_key="title", url_key="url", snippet_key="content")
        except Exception:
            pass

        # 2. DuckDuckGo
        try:
            from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                items = list(ddgs.text(query, max_results=5))
            return self._format_search_results(items, title_key="title", url_key="href", snippet_key="body")
        except Exception:
            pass

        # 3. Brave Search
        brave_key = os.environ.get("BRAVE_API_KEY")
        if brave_key:
            try:
                import urllib.request

                req = urllib.request.Request(
                    f"https://api.search.brave.com/res/v1/web/search?q={urllib.parse.quote(query)}&count=5",
                    headers={"Accept": "application/json", "X-Subscription-Token": brave_key},
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read())
                items = data.get("web", {}).get("results", [])[:5]
                return self._format_search_results(items, title_key="title", url_key="url", snippet_key="description")
            except Exception:
                pass

        return None

    @staticmethod
    def _format_search_results(
        items: list[dict],
        *,
        title_key: str,
        url_key: str,
        snippet_key: str,
    ) -> str:
        lines = []
        for i, item in enumerate(items, 1):
            title = str(item.get(title_key, ""))[:150]
            url = str(item.get(url_key, ""))
            snippet = str(item.get(snippet_key, ""))[:150]
            lines.append(f"{i}. {title}\n   {url}\n   {snippet}")
        return "\n\n".join(lines)

    # ------------------------------------------------------------------
    # Tool: python_eval
    # ------------------------------------------------------------------

    def _python_eval(self, code: str) -> dict[str, Any]:
        tier, risk = classify_python(code)
        allowed, reason = self._gate(tier, risk, f"python_eval: {code[:120]!r}")
        if not allowed:
            return format_result("python_eval", reason, error=True)

        captured = io.StringIO()
        local_ns: dict = {}

        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = captured
        sys.stderr = captured
        try:
            exec(code, dict(_EVAL_GLOBALS), local_ns)  # noqa: S102
        except Exception as exc:
            return format_result(
                "python_eval",
                f"{captured.getvalue()}{type(exc).__name__}: {exc}",
                error=True,
                token_budget=self.token_budget,
            )
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        return format_result("python_eval", captured.getvalue(), token_budget=self.token_budget)
