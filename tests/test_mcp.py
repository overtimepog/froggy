"""Tests for MCP client integration."""

import textwrap
from unittest.mock import MagicMock

from froggy.mcp_client import (
    MCPManager,
    _mcp_tool_to_tooldef,
    load_mcp_config,
)


class TestLoadMCPConfig:
    def test_returns_empty_when_no_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FROGGY_HOME", str(tmp_path))
        assert load_mcp_config() == []

    def test_parses_valid_config(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FROGGY_HOME", str(tmp_path))
        config = tmp_path / "mcp.yaml"
        config.write_text(textwrap.dedent("""\
            servers:
              fetch:
                command: uvx
                args: [mcp-server-fetch]
              myserver:
                command: python
                args: [server.py, --port, "8080"]
                env:
                  API_KEY: sk-test
        """))
        configs = load_mcp_config()
        assert len(configs) == 2
        assert configs[0].name == "fetch"
        assert configs[0].command == "uvx"
        assert configs[0].args == ["mcp-server-fetch"]
        assert configs[1].name == "myserver"
        assert configs[1].env == {"API_KEY": "sk-test"}

    def test_skips_invalid_entries(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FROGGY_HOME", str(tmp_path))
        config = tmp_path / "mcp.yaml"
        config.write_text(textwrap.dedent("""\
            servers:
              good:
                command: python
                args: [server.py]
              bad_no_command:
                args: [something]
              bad_not_dict: "just a string"
        """))
        configs = load_mcp_config()
        assert len(configs) == 1
        assert configs[0].name == "good"

    def test_handles_corrupt_yaml(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FROGGY_HOME", str(tmp_path))
        config = tmp_path / "mcp.yaml"
        config.write_text("{{{{invalid yaml")
        assert load_mcp_config() == []


class TestMCPToolToToolDef:
    def test_converts_basic_tool(self):
        mock_tool = MagicMock()
        mock_tool.name = "fetch"
        mock_tool.description = "Fetch a URL"
        mock_tool.inputSchema = {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to fetch"},
                "timeout": {"type": "integer", "description": "Timeout in seconds"},
            },
            "required": ["url"],
        }
        td = _mcp_tool_to_tooldef(mock_tool, "myserver")
        assert td.name == "mcp_myserver_fetch"
        assert "[MCP:myserver]" in td.description
        assert len(td.params) == 2
        assert td.params[0].name == "url"
        assert td.params[0].required is True
        assert td.params[1].name == "timeout"
        assert td.params[1].required is False

    def test_handles_no_schema(self):
        mock_tool = MagicMock()
        mock_tool.name = "ping"
        mock_tool.description = "Ping"
        mock_tool.inputSchema = None
        td = _mcp_tool_to_tooldef(mock_tool, "test")
        assert td.name == "mcp_test_ping"
        assert td.params == []

    def test_json_schema_export(self):
        mock_tool = MagicMock()
        mock_tool.name = "add"
        mock_tool.description = "Add numbers"
        mock_tool.inputSchema = {
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "First number"},
                "b": {"type": "integer", "description": "Second number"},
            },
            "required": ["a", "b"],
        }
        td = _mcp_tool_to_tooldef(mock_tool, "math")
        schema = td.to_json_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "mcp_math_add"
        assert "a" in schema["function"]["parameters"]["properties"]


class TestMCPManager:
    def test_is_mcp_tool_false_when_empty(self):
        mgr = MCPManager()
        assert mgr.is_mcp_tool("anything") is False

    def test_connect_all_no_config(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FROGGY_HOME", str(tmp_path))
        mgr = MCPManager()
        tools = mgr.connect_all()
        assert tools == []

    def test_server_tools_unknown(self):
        mgr = MCPManager()
        assert mgr.server_tools("nonexistent") == []

    def test_tool_count_starts_zero(self):
        mgr = MCPManager()
        assert mgr.tool_count == 0


class TestToolExecutorMCPIntegration:
    def test_execute_routes_to_mcp(self):
        from froggy.tool_executor import ToolExecutor

        mock_mgr = MagicMock()
        mock_mgr.is_mcp_tool.return_value = True
        mock_mgr.call_tool.return_value = "MCP result: 42"

        executor = ToolExecutor(
            mcp_manager=mock_mgr,
            confirm_fn=lambda desc, risk: True,
        )

        result = executor.execute("mcp_math_add", a=1, b=2)
        assert result["ok"] is True
        assert "42" in result["output"]
        mock_mgr.call_tool.assert_called_once_with("mcp_math_add", {"a": 1, "b": 2})

    def test_execute_non_mcp_unchanged(self):
        from froggy.tool_executor import ToolExecutor

        mock_mgr = MagicMock()
        mock_mgr.is_mcp_tool.return_value = False

        executor = ToolExecutor(mcp_manager=mock_mgr)
        result = executor.execute("nonexistent_tool")
        assert result["ok"] is False
        assert "Unknown tool" in result["output"]

    def test_mcp_error_handled_gracefully(self):
        from froggy.tool_executor import ToolExecutor

        mock_mgr = MagicMock()
        mock_mgr.is_mcp_tool.return_value = True
        mock_mgr.call_tool.side_effect = RuntimeError("Server crashed")

        executor = ToolExecutor(
            mcp_manager=mock_mgr,
            confirm_fn=lambda desc, risk: True,
        )
        result = executor.execute("mcp_broken_tool")
        assert result["ok"] is False
        assert "MCP error" in result["output"]
