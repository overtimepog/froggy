"""MCP (Model Context Protocol) client integration for froggy.

Connects to MCP servers via stdio transport, discovers their tools,
and bridges them into froggy's ToolDef/ToolRegistry/ToolExecutor system
so local LLMs can call MCP tools natively.

Config lives at ``~/.froggy/mcp.yaml``::

    servers:
      fetch:
        command: uvx
        args: [mcp-server-fetch]
      filesystem:
        command: npx
        args: [-y, "@modelcontextprotocol/server-filesystem", "/tmp"]
      my-server:
        command: python
        args: [my_server.py]
        env:
          API_KEY: "sk-..."
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .paths import froggy_home
from .tools import ToolDef, ToolParam

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class MCPServerConfig:
    """Configuration for a single MCP server."""

    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)


def mcp_config_path() -> Path:
    return froggy_home() / "mcp.yaml"


def load_mcp_config() -> list[MCPServerConfig]:
    """Load MCP server configs from ``~/.froggy/mcp.yaml``."""
    p = mcp_config_path()
    if not p.exists():
        return []
    try:
        data = yaml.safe_load(p.read_text())
    except yaml.YAMLError:
        return []
    if not isinstance(data, dict):
        return []

    servers = data.get("servers", {})
    if not isinstance(servers, dict):
        return []

    result = []
    for name, cfg in servers.items():
        if not isinstance(cfg, dict) or "command" not in cfg:
            continue
        result.append(
            MCPServerConfig(
                name=str(name),
                command=cfg["command"],
                args=[str(a) for a in cfg.get("args", [])],
                env={str(k): str(v) for k, v in cfg.get("env", {}).items()},
            )
        )
    return result


# ---------------------------------------------------------------------------
# MCP tool schema → froggy ToolDef conversion
# ---------------------------------------------------------------------------

_JSON_TYPE_MAP = {
    "string": "string",
    "integer": "integer",
    "number": "number",
    "boolean": "boolean",
    "array": "array",
    "object": "object",
}


def _mcp_tool_to_tooldef(tool: Any, server_name: str) -> ToolDef:
    """Convert an MCP Tool object to a froggy ToolDef."""
    # Prefix tool name with server name to avoid collisions
    froggy_name = f"mcp_{server_name}_{tool.name}"

    params: list[ToolParam] = []
    schema = tool.inputSchema or {}
    properties = schema.get("properties", {})
    required_set = set(schema.get("required", []))

    for pname, pschema in properties.items():
        ptype = _JSON_TYPE_MAP.get(pschema.get("type", "string"), "string")
        pdesc = pschema.get("description", "")
        penum = pschema.get("enum")
        params.append(
            ToolParam(
                name=pname,
                type=ptype,
                description=pdesc,
                required=pname in required_set,
                enum=penum,
            )
        )

    return ToolDef(
        name=froggy_name,
        description=f"[MCP:{server_name}] {tool.description or tool.name}",
        params=params,
    )


# ---------------------------------------------------------------------------
# Live MCP connection manager
# ---------------------------------------------------------------------------


class MCPConnection:
    """Manages a live connection to a single MCP server."""

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self._session = None
        self._read = None
        self._write = None
        self._client_cm = None
        self._session_cm = None
        self._tools: dict[str, Any] = {}  # mcp_name → MCP Tool object
        self._tool_name_map: dict[str, str] = {}  # froggy_name → mcp original name

    async def connect(self) -> list[ToolDef]:
        """Start the MCP server and discover its tools. Returns froggy ToolDefs."""
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        env = {**os.environ, **self.config.env}
        server_params = StdioServerParameters(
            command=self.config.command,
            args=self.config.args,
            env=env,
        )

        self._client_cm = stdio_client(server_params)
        self._read, self._write = await self._client_cm.__aenter__()

        self._session_cm = ClientSession(self._read, self._write)
        self._session = await self._session_cm.__aenter__()
        await self._session.initialize()

        # Discover tools
        tools_result = await self._session.list_tools()
        tooldefs = []
        for tool in tools_result.tools:
            td = _mcp_tool_to_tooldef(tool, self.config.name)
            self._tools[tool.name] = tool
            self._tool_name_map[td.name] = tool.name
            tooldefs.append(td)

        return tooldefs

    async def call_tool(self, froggy_name: str, arguments: dict[str, Any]) -> str:
        """Call an MCP tool by its froggy-prefixed name. Returns result as string."""
        if self._session is None:
            raise RuntimeError(f"MCP server '{self.config.name}' not connected")

        mcp_name = self._tool_name_map.get(froggy_name)
        if mcp_name is None:
            raise ValueError(f"Unknown MCP tool: {froggy_name}")

        from mcp import types

        result = await self._session.call_tool(mcp_name, arguments=arguments)

        # Extract text from result content blocks
        parts = []
        for block in result.content:
            if isinstance(block, types.TextContent):
                parts.append(block.text)
            elif isinstance(block, types.ImageContent):
                parts.append(f"[image: {block.mimeType}]")
            elif isinstance(block, types.EmbeddedResource):
                parts.append(f"[resource: {block.resource.uri}]")
            else:
                parts.append(str(block))

        return "\n".join(parts)

    async def disconnect(self):
        """Shut down the MCP server connection."""
        if self._session_cm is not None:
            try:
                await self._session_cm.__aexit__(None, None, None)
            except Exception:
                pass
        if self._client_cm is not None:
            try:
                await self._client_cm.__aexit__(None, None, None)
            except Exception:
                pass
        self._session = None

    @property
    def is_connected(self) -> bool:
        return self._session is not None

    @property
    def tool_names(self) -> list[str]:
        return list(self._tool_name_map.keys())


# ---------------------------------------------------------------------------
# MCP Manager — orchestrates all server connections
# ---------------------------------------------------------------------------


class MCPManager:
    """Manages multiple MCP server connections and bridges their tools into froggy."""

    def __init__(self):
        self._connections: dict[str, MCPConnection] = {}
        self._tool_to_server: dict[str, str] = {}  # froggy_name → server_name
        self._loop: asyncio.AbstractEventLoop | None = None

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create an event loop for running async MCP operations."""
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop

    def connect_all(self) -> list[ToolDef]:
        """Load config, connect to all MCP servers, return all discovered tools."""
        configs = load_mcp_config()
        if not configs:
            return []

        loop = self._get_loop()
        return loop.run_until_complete(self._connect_all_async(configs))

    async def _connect_all_async(self, configs: list[MCPServerConfig]) -> list[ToolDef]:
        all_tools: list[ToolDef] = []
        for cfg in configs:
            try:
                conn = MCPConnection(cfg)
                tools = await conn.connect()
                self._connections[cfg.name] = conn
                for td in tools:
                    self._tool_to_server[td.name] = cfg.name
                all_tools.extend(tools)
                logger.info("MCP server '%s': %d tools", cfg.name, len(tools))
            except Exception as exc:
                logger.warning("Failed to connect to MCP server '%s': %s", cfg.name, exc)
        return all_tools

    def call_tool(self, froggy_name: str, arguments: dict[str, Any]) -> str:
        """Call an MCP tool synchronously. Returns result string."""
        server_name = self._tool_to_server.get(froggy_name)
        if server_name is None:
            raise ValueError(f"No MCP server for tool: {froggy_name}")

        conn = self._connections[server_name]
        loop = self._get_loop()
        return loop.run_until_complete(conn.call_tool(froggy_name, arguments))

    def is_mcp_tool(self, tool_name: str) -> bool:
        """Check if a tool name belongs to an MCP server."""
        return tool_name in self._tool_to_server

    def disconnect_all(self):
        """Disconnect all MCP servers."""
        if not self._connections:
            return
        loop = self._get_loop()
        loop.run_until_complete(self._disconnect_all_async())

    async def _disconnect_all_async(self):
        for conn in self._connections.values():
            await conn.disconnect()
        self._connections.clear()
        self._tool_to_server.clear()

    @property
    def server_names(self) -> list[str]:
        return list(self._connections.keys())

    @property
    def tool_count(self) -> int:
        return len(self._tool_to_server)

    def server_tools(self, server_name: str) -> list[str]:
        """Return froggy tool names for a specific server."""
        conn = self._connections.get(server_name)
        return conn.tool_names if conn else []
