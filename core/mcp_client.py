"""
Reusable async MCP client layer.

Provides MCPClient (single connection) and MCPClientPool (shared pool of
named connections, one per MCP server key defined in mcp_config.py).

Usage
-----
    async with MCPClientPool(server_keys=["fs_code", "git"]) as pool:
        tools = await pool.list_tools("fs_code")
        result = await pool.call_tool("git", "git_status", {})
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .mcp_config import MCP_SERVERS

logger = logging.getLogger(__name__)


class MCPClient:
    """
    Manages a single stdio MCP session.

    The entire connection lifecycle (stdio_client + ClientSession) is run
    inside a dedicated asyncio Task so that anyio cancel scopes are always
    entered and exited within the same task — avoiding the
    "Attempted to exit cancel scope in a different task" RuntimeError that
    occurs when asyncio.gather() creates sub-tasks for connection setup.
    """

    def __init__(self, server_key: str) -> None:
        self._server_key = server_key
        self._session: ClientSession | None = None
        self._ready_event: asyncio.Event = asyncio.Event()
        self._shutdown_event: asyncio.Event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._connect_error: BaseException | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _run(self, timeout: float) -> None:
        """Long-lived task: open the MCP connection, signal ready, then wait for shutdown."""
        cfg = MCP_SERVERS[self._server_key]
        params = StdioServerParameters(
            command=cfg["command"],
            args=cfg["args"],
            env=cfg.get("env"),
        )
        logger.info("[MCPClient:%s] connecting – %s %s", self._server_key, cfg["command"], cfg["args"])
        try:
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await asyncio.wait_for(session.initialize(), timeout=timeout)
                    self._session = session
                    logger.info("[MCPClient:%s] connected", self._server_key)
                    self._ready_event.set()
                    await self._shutdown_event.wait()
        except Exception as exc:
            self._connect_error = exc
            self._ready_event.set()
        finally:
            self._session = None

    async def connect(self, timeout: float = 30.0) -> None:
        self._task = asyncio.ensure_future(self._run(timeout))
        await self._ready_event.wait()
        if self._connect_error is not None:
            raise self._connect_error

    async def close(self) -> None:
        self._shutdown_event.set()
        if self._task is not None:
            try:
                await self._task
            except Exception as exc:
                logger.warning("[MCPClient:%s] error during shutdown: %s", self._server_key, exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def list_tools(self, timeout: float = 15.0) -> list:
        """Return tool definitions from the MCP server (best-effort)."""
        if not self._session:
            raise RuntimeError(f"MCPClient:{self._server_key} not connected")
        try:
            resp = await asyncio.wait_for(self._session.list_tools(), timeout=timeout)
            return resp.tools
        except asyncio.TimeoutError:
            logger.warning("[MCPClient:%s] list_tools timed out", self._server_key)
            return []
        except Exception as exc:
            logger.warning("[MCPClient:%s] list_tools error: %s", self._server_key, exc)
            return []

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        if not self._session:
            raise RuntimeError(f"MCPClient:{self._server_key} not connected")
        logger.debug("[MCPClient:%s] call_tool %s %s", self._server_key, tool_name, arguments)
        return await self._session.call_tool(tool_name, arguments)


class MCPClientPool:
    """
    Async context manager that owns a pool of named MCPClient instances.

    Parameters
    ----------
    server_keys:
        Subset of keys from mcp_config.MCP_SERVERS to connect on enter.
        Pass None to connect to all registered servers.
    """

    def __init__(self, server_keys: list[str] | None = None) -> None:
        keys = server_keys if server_keys is not None else list(MCP_SERVERS)
        self._clients: dict[str, MCPClient] = {k: MCPClient(k) for k in keys}

    async def __aenter__(self) -> "MCPClientPool":
        # Protect against individual MCP servers hanging indefinitely on connect
        connect_coros = (
            asyncio.wait_for(c.connect(), timeout=60.0) for c in self._clients.values()
        )
        results = await asyncio.gather(*connect_coros, return_exceptions=True)

        errors: dict[str, Exception] = {}
        for key, result in zip(self._clients.keys(), results):
            if isinstance(result, Exception):
                errors[key] = result
                logger.error(
                    "[MCPClientPool] failed to connect '%s': %s", key, result
                )

        if errors:
            failed = ", ".join(errors.keys())
            raise RuntimeError(f"Failed to connect MCP servers: {failed}")

        return self

    async def __aexit__(self, *_: Any) -> None:
        for c in self._clients.values():
            await c.close()

    # ------------------------------------------------------------------
    # Delegation helpers
    # ------------------------------------------------------------------

    def _get(self, server_key: str) -> MCPClient:
        try:
            return self._clients[server_key]
        except KeyError:
            raise KeyError(f"Server '{server_key}' not in pool. Available: {list(self._clients)}")

    async def list_tools(self, server_key: str) -> list:
        return await self._get(server_key).list_tools()

    async def call_tool(self, server_key: str, tool_name: str, arguments: dict[str, Any]) -> Any:
        return await self._get(server_key).call_tool(tool_name, arguments)

    async def list_all_tools(self) -> dict[str, list]:
        """Return {server_key: [tool, ...]} for every server in the pool."""
        results = await asyncio.gather(*(c.list_tools() for c in self._clients.values()))
        return dict(zip(self._clients.keys(), results))
