"""Tests for ToolExecutor dispatch, file tools, shell tool, and token truncation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from froggy.tool_executor import ToolExecutor, format_result, truncate_to_budget

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def auto_confirm(description: str, risk) -> bool:
    """confirm_fn that always approves — used so tests don't block on stdin."""
    return True


def auto_deny(description: str, risk) -> bool:
    return False


def make_executor(tmp_path: Path, *, confirm: bool = True) -> ToolExecutor:
    confirm_fn = auto_confirm if confirm else auto_deny
    return ToolExecutor(project_root=tmp_path, confirm_fn=confirm_fn, use_sandbox=False)


# ---------------------------------------------------------------------------
# format_result / truncate_to_budget
# ---------------------------------------------------------------------------


class TestFormatResult:
    def test_ok_result(self):
        r = format_result("read_file", "hello")
        assert r["ok"] is True
        assert r["tool"] == "read_file"
        assert r["output"] == "hello"
        assert r["truncated"] is False

    def test_error_result(self):
        r = format_result("read_file", "oops", error=True)
        assert r["ok"] is False

    def test_truncation_flag(self):
        long_text = "x" * 10_000
        r = format_result("read_file", long_text, token_budget=10)
        assert r["truncated"] is True
        assert len(r["output"]) < len(long_text)

    def test_no_truncation_within_budget(self):
        text = "short"
        r = format_result("read_file", text, token_budget=100)
        assert r["output"] == text
        assert r["truncated"] is False


class TestTruncateToBudget:
    def test_fits_in_budget(self):
        assert truncate_to_budget("hello", 100) == "hello"

    def test_exceeds_budget(self):
        text = "a" * 400
        result = truncate_to_budget(text, 10)  # 10 tokens = 40 chars
        assert len(result) < len(text)
        assert "truncated" in result

    def test_zero_budget(self):
        assert truncate_to_budget("hello", 0) == ""

    def test_symmetric_truncation(self):
        text = "start" + ("m" * 2000) + "end"
        result = truncate_to_budget(text, 10)
        assert "start" in result
        assert "end" in result


# ---------------------------------------------------------------------------
# ToolExecutor.execute — unknown tool
# ---------------------------------------------------------------------------


class TestDispatch:
    def test_unknown_tool(self, tmp_path):
        ex = make_executor(tmp_path)
        result = ex.execute("nonexistent_tool")
        assert result["ok"] is False
        assert "Unknown tool" in result["output"]

    def test_available_tools(self, tmp_path):
        ex = make_executor(tmp_path)
        tools = ex.available_tools
        for name in ["read_file", "write_file", "edit_file", "run_shell", "web_search", "python_eval"]:
            assert name in tools


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------


class TestReadFile:
    def test_reads_existing_file(self, tmp_path):
        (tmp_path / "hello.txt").write_text("line1\nline2\nline3")
        ex = make_executor(tmp_path)
        result = ex.execute("read_file", path="hello.txt")
        assert result["ok"] is True
        assert "line1" in result["output"]
        assert "line2" in result["output"]

    def test_file_not_found(self, tmp_path):
        ex = make_executor(tmp_path)
        result = ex.execute("read_file", path="missing.txt")
        assert result["ok"] is False
        assert "not found" in result["output"].lower()

    def test_line_numbers_in_output(self, tmp_path):
        (tmp_path / "f.txt").write_text("a\nb\nc")
        ex = make_executor(tmp_path)
        result = ex.execute("read_file", path="f.txt")
        assert result["ok"] is True
        assert "1" in result["output"]

    def test_offset_and_limit(self, tmp_path):
        lines = "\n".join(str(i) for i in range(20))
        (tmp_path / "nums.txt").write_text(lines)
        ex = make_executor(tmp_path)
        result = ex.execute("read_file", path="nums.txt", offset=5, limit=3)
        assert result["ok"] is True
        output = result["output"]
        # Should contain lines 5,6,7 (0-indexed) — values "5","6","7"
        assert "5" in output
        assert "6" in output
        assert "7" in output

    def test_path_traversal_blocked(self, tmp_path):
        ex = make_executor(tmp_path)
        result = ex.execute("read_file", path="../../etc/passwd")
        assert result["ok"] is False
        assert "traversal" in result["output"].lower() or "blocked" in result["output"].lower()

    def test_sensitive_path_blocked(self, tmp_path):
        ex = make_executor(tmp_path)
        # Absolute path to a .pem file
        result = ex.execute("read_file", path="/some/path/cert.pem")
        assert result["ok"] is False

    def test_nested_path_ok(self, tmp_path):
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "file.txt").write_text("nested")
        ex = make_executor(tmp_path)
        result = ex.execute("read_file", path="subdir/file.txt")
        assert result["ok"] is True
        assert "nested" in result["output"]


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------


class TestWriteFile:
    def test_writes_new_file(self, tmp_path):
        ex = make_executor(tmp_path)
        result = ex.execute("write_file", path="new.txt", content="hello world")
        assert result["ok"] is True
        assert (tmp_path / "new.txt").read_text() == "hello world"

    def test_creates_parent_dirs(self, tmp_path):
        ex = make_executor(tmp_path)
        result = ex.execute("write_file", path="a/b/c.txt", content="deep")
        assert result["ok"] is True
        assert (tmp_path / "a" / "b" / "c.txt").read_text() == "deep"

    def test_overwrite_existing_file(self, tmp_path):
        (tmp_path / "existing.txt").write_text("old")
        ex = make_executor(tmp_path)
        result = ex.execute("write_file", path="existing.txt", content="new")
        assert result["ok"] is True
        assert (tmp_path / "existing.txt").read_text() == "new"

    def test_denied_when_user_declines(self, tmp_path):
        ex = make_executor(tmp_path, confirm=False)
        result = ex.execute("write_file", path="denied.txt", content="x")
        assert result["ok"] is False
        assert "declined" in result["output"].lower()

    def test_path_traversal_blocked(self, tmp_path):
        ex = make_executor(tmp_path)
        result = ex.execute("write_file", path="../outside.txt", content="x")
        assert result["ok"] is False

    def test_sensitive_path_blocked(self, tmp_path):
        ex = make_executor(tmp_path)
        result = ex.execute("write_file", path="/etc/hosts", content="x")
        assert result["ok"] is False


# ---------------------------------------------------------------------------
# edit_file
# ---------------------------------------------------------------------------


class TestEditFile:
    def test_basic_edit(self, tmp_path):
        (tmp_path / "src.py").write_text("def foo():\n    return 1\n")
        ex = make_executor(tmp_path)
        result = ex.execute("edit_file", path="src.py", old_string="return 1", new_string="return 2")
        assert result["ok"] is True
        assert (tmp_path / "src.py").read_text() == "def foo():\n    return 2\n"

    def test_old_string_not_found(self, tmp_path):
        (tmp_path / "f.txt").write_text("hello")
        ex = make_executor(tmp_path)
        result = ex.execute("edit_file", path="f.txt", old_string="missing", new_string="x")
        assert result["ok"] is False
        assert "not found" in result["output"]

    def test_ambiguous_match(self, tmp_path):
        (tmp_path / "f.txt").write_text("foo foo foo")
        ex = make_executor(tmp_path)
        result = ex.execute("edit_file", path="f.txt", old_string="foo", new_string="bar")
        assert result["ok"] is False
        assert "3" in result["output"]  # "appears 3 times"

    def test_file_not_found(self, tmp_path):
        ex = make_executor(tmp_path)
        result = ex.execute("edit_file", path="nope.py", old_string="x", new_string="y")
        assert result["ok"] is False

    def test_denied_when_user_declines(self, tmp_path):
        (tmp_path / "f.txt").write_text("old")
        ex = make_executor(tmp_path, confirm=False)
        result = ex.execute("edit_file", path="f.txt", old_string="old", new_string="new")
        assert result["ok"] is False

    def test_path_traversal_blocked(self, tmp_path):
        ex = make_executor(tmp_path)
        result = ex.execute("edit_file", path="../../etc/passwd", old_string="x", new_string="y")
        assert result["ok"] is False


# ---------------------------------------------------------------------------
# run_shell
# ---------------------------------------------------------------------------


class TestRunShell:
    def test_simple_echo(self, tmp_path):
        ex = make_executor(tmp_path)
        result = ex.execute("run_shell", cmd="echo hello")
        assert result["ok"] is True
        assert "hello" in result["output"]

    def test_exit_nonzero_is_error(self, tmp_path):
        ex = make_executor(tmp_path)
        result = ex.execute("run_shell", cmd="false")
        assert result["ok"] is False

    def test_blocked_command(self, tmp_path):
        ex = make_executor(tmp_path)
        result = ex.execute("run_shell", cmd="sudo ls")
        assert result["ok"] is False
        assert "blocked" in result["output"].lower()

    def test_rm_rf_root_blocked(self, tmp_path):
        ex = make_executor(tmp_path)
        result = ex.execute("run_shell", cmd="rm -rf /")
        assert result["ok"] is False

    def test_pipe_to_shell_blocked(self, tmp_path):
        ex = make_executor(tmp_path)
        result = ex.execute("run_shell", cmd="echo bad | bash")
        assert result["ok"] is False

    def test_unknown_command_requires_confirm(self, tmp_path):
        """Unknown executable → CONFIRM tier; auto_deny returns False → blocked."""
        ex = make_executor(tmp_path, confirm=False)
        result = ex.execute("run_shell", cmd="some_unknown_binary --flag")
        assert result["ok"] is False

    def test_timeout(self, tmp_path):
        ex = ToolExecutor(
            project_root=tmp_path, confirm_fn=auto_confirm, use_sandbox=False, shell_timeout=1
        )
        result = ex.execute("run_shell", cmd="sleep 10")
        assert result["ok"] is False
        assert "timed out" in result["output"].lower()

    def test_stderr_included(self, tmp_path):
        ex = make_executor(tmp_path)
        result = ex.execute("run_shell", cmd="python3 -c \"import sys; sys.stderr.write('err\\n'); sys.exit(1)\"")
        assert "err" in result["output"]


# ---------------------------------------------------------------------------
# web_search (stub behaviour)
# ---------------------------------------------------------------------------


class TestWebSearch:
    def test_returns_error_when_no_backend(self, tmp_path):
        """With no search backend configured the tool should report an error gracefully."""
        ex = make_executor(tmp_path)
        # Patch _do_web_search to simulate no backend available
        with patch.object(ex, "_do_web_search", return_value=None):
            result = ex.execute("web_search", query="python pathlib")
        assert result["ok"] is False
        assert "backend" in result["output"].lower() or "available" in result["output"].lower()

    def test_returns_results_when_backend_works(self, tmp_path):
        ex = make_executor(tmp_path)
        fake = "1. Python docs\n   https://docs.python.org\n   Official Python docs."
        with patch.object(ex, "_do_web_search", return_value=fake):
            result = ex.execute("web_search", query="python")
        assert result["ok"] is True
        assert "Python" in result["output"]


# ---------------------------------------------------------------------------
# python_eval
# ---------------------------------------------------------------------------


class TestPythonEval:
    def test_basic_arithmetic(self, tmp_path):
        ex = make_executor(tmp_path)
        result = ex.execute("python_eval", code="print(2 + 2)")
        assert result["ok"] is True
        assert "4" in result["output"]

    def test_capture_stdout(self, tmp_path):
        ex = make_executor(tmp_path)
        result = ex.execute("python_eval", code="print('hello from eval')")
        assert result["ok"] is True
        assert "hello from eval" in result["output"]

    def test_syntax_error(self, tmp_path):
        ex = make_executor(tmp_path)
        result = ex.execute("python_eval", code="def (broken")
        assert result["ok"] is False

    def test_runtime_exception(self, tmp_path):
        ex = make_executor(tmp_path)
        result = ex.execute("python_eval", code="1/0")
        assert result["ok"] is False
        assert "ZeroDivisionError" in result["output"]

    def test_import_requires_confirm(self, tmp_path):
        """Code with 'import' hits CONFIRM tier; auto_deny = False."""
        ex = make_executor(tmp_path, confirm=False)
        result = ex.execute("python_eval", code="import os; print(os.getcwd())")
        assert result["ok"] is False

    def test_open_requires_confirm(self, tmp_path):
        ex = make_executor(tmp_path, confirm=False)
        result = ex.execute("python_eval", code="open('/etc/passwd')")
        assert result["ok"] is False

    def test_dunder_requires_confirm(self, tmp_path):
        ex = make_executor(tmp_path, confirm=False)
        result = ex.execute("python_eval", code="print(__builtins__)")
        assert result["ok"] is False

    def test_math_module_available(self, tmp_path):
        ex = make_executor(tmp_path)
        result = ex.execute("python_eval", code="print(math.sqrt(16))")
        assert result["ok"] is True
        assert "4.0" in result["output"]

    def test_json_module_available(self, tmp_path):
        ex = make_executor(tmp_path)
        result = ex.execute("python_eval", code='print(json.dumps({"a": 1}))')
        assert result["ok"] is True
        assert '"a"' in result["output"]

    def test_denied_returns_error(self, tmp_path):
        ex = make_executor(tmp_path, confirm=False)
        result = ex.execute("python_eval", code="print('hi')")
        # pure compute but confirm_fn always denies
        assert result["ok"] is False
        assert "declined" in result["output"].lower()
