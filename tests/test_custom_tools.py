"""Tests for the custom tool loader (load_custom_tools)."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from froggy.session import load_custom_tools, _TOOLS_AVAILABLE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_tool_file(tmp_path: Path, name: str, content: str) -> Path:
    """Write a Python file to tmp_path and return its path."""
    p = tmp_path / name
    p.write_text(textwrap.dedent(content))
    return p


# ---------------------------------------------------------------------------
# Behaviour when tool modules are not available
# ---------------------------------------------------------------------------


def test_load_custom_tools_returns_empty_when_no_dir(tmp_path):
    """Non-existent directory returns an empty list."""
    result = load_custom_tools(tmp_path / "nonexistent")
    assert result == []


def test_load_custom_tools_returns_empty_when_unavailable(tmp_path):
    """Returns empty list when _TOOLS_AVAILABLE is False."""
    with patch("froggy.session._TOOLS_AVAILABLE", False):
        result = load_custom_tools(tmp_path)
    assert result == []


# ---------------------------------------------------------------------------
# Tests that run only when the tool modules are importable
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _TOOLS_AVAILABLE, reason="tool modules not installed")
class TestLoadCustomToolsWithModules:
    def test_loads_single_tool_via_TOOL(self, tmp_path):
        """A file exporting TOOL = ToolDef(...) is loaded."""
        from froggy.tools import ToolDef, ToolParam

        _write_tool_file(tmp_path, "my_tool.py", """
            from froggy.tools import ToolDef, ToolParam
            TOOL = ToolDef(
                name="my_custom_tool",
                description="A custom tool for testing",
                params=[ToolParam("arg", "string", "An argument")],
            )
        """)

        tools = load_custom_tools(tmp_path)
        assert len(tools) == 1
        assert tools[0].name == "my_custom_tool"

    def test_loads_multiple_tools_via_TOOLS(self, tmp_path):
        """A file exporting TOOLS = [...] loads all entries."""
        _write_tool_file(tmp_path, "multi.py", """
            from froggy.tools import ToolDef
            TOOLS = [
                ToolDef(name="tool_alpha", description="Alpha"),
                ToolDef(name="tool_beta", description="Beta"),
            ]
        """)

        tools = load_custom_tools(tmp_path)
        names = [t.name for t in tools]
        assert "tool_alpha" in names
        assert "tool_beta" in names

    def test_skips_files_starting_with_underscore(self, tmp_path):
        """Files beginning with _ are ignored."""
        _write_tool_file(tmp_path, "_internal.py", """
            from froggy.tools import ToolDef
            TOOL = ToolDef(name="secret_tool", description="Should not load")
        """)

        tools = load_custom_tools(tmp_path)
        assert all(t.name != "secret_tool" for t in tools)

    def test_silently_skips_broken_files(self, tmp_path):
        """A file with a syntax/import error is skipped without crashing."""
        _write_tool_file(tmp_path, "broken.py", "this is not valid python !!!")
        _write_tool_file(tmp_path, "good.py", """
            from froggy.tools import ToolDef
            TOOL = ToolDef(name="good_tool", description="Fine")
        """)

        tools = load_custom_tools(tmp_path)
        assert len(tools) == 1
        assert tools[0].name == "good_tool"

    def test_ignores_files_with_no_tool_export(self, tmp_path):
        """Files that export neither TOOL nor TOOLS are ignored."""
        _write_tool_file(tmp_path, "helper.py", """
            def helper():
                pass
        """)

        tools = load_custom_tools(tmp_path)
        assert tools == []

    def test_ignores_non_tooldef_exports(self, tmp_path):
        """Files that export TOOL of the wrong type are ignored."""
        _write_tool_file(tmp_path, "bad_type.py", """
            TOOL = "not a ToolDef"
        """)

        tools = load_custom_tools(tmp_path)
        assert tools == []

    def test_loads_from_empty_directory(self, tmp_path):
        """An empty directory returns an empty list."""
        tools = load_custom_tools(tmp_path)
        assert tools == []

    def test_multiple_files_loaded_in_sorted_order(self, tmp_path):
        """Multiple files are loaded and names are deterministic (sorted)."""
        for letter in ("c", "a", "b"):
            _write_tool_file(tmp_path, f"tool_{letter}.py", f"""
                from froggy.tools import ToolDef
                TOOL = ToolDef(name="tool_{letter}", description="{letter}")
            """)

        tools = load_custom_tools(tmp_path)
        names = [t.name for t in tools]
        assert names == ["tool_a", "tool_b", "tool_c"]


# ---------------------------------------------------------------------------
# Tests using mocks (run regardless of _TOOLS_AVAILABLE)
# ---------------------------------------------------------------------------


def test_load_custom_tools_with_mocked_tooldef(tmp_path):
    """load_custom_tools integrates with mocked ToolDef gracefully."""
    fake_tool_def = MagicMock()
    fake_tool_def_class = MagicMock(return_value=fake_tool_def)

    _write_tool_file(tmp_path, "mock_tool.py", "TOOL = None")

    # Patch ToolDef and _TOOLS_AVAILABLE so we can test the loader logic
    with patch("froggy.session._TOOLS_AVAILABLE", True), \
         patch("froggy.session.ToolDef", fake_tool_def_class):
        # The file exports TOOL = None, so isinstance check fails — returns []
        tools = load_custom_tools(tmp_path)

    assert tools == []
