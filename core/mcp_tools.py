"""
Thin MCP-backed tool wrappers for AutoGen agents.

These functions are simple adapters:
- AutoGen sees them as Python tools.
- Internally they call MCP servers via `MCPClientPool.call_tool`.
"""
from __future__ import annotations

import asyncio
import functools
import inspect
from typing import Any, Callable

from core.mcp_client import MCPClientPool


BOARD_ROOT = "data/project_board"
DOCS_ROOT = "data/knowledge_base"
CODE_ROOT = "data/workspace"


def _ensure_under_root(path: str, root: str) -> str:
    """Normalize tool paths so they are always under the configured root."""
    if not path or path in (".", "/"):
        return root
    # Already rooted
    if path.startswith(root + "/") or path == root:
        return path
    # Strip leading slash and join under root
    return f"{root}/{path.lstrip('/')}"


async def _fs_call(pool: MCPClientPool, server_key: str, tool: str, args: dict[str, Any]) -> Any:
    result = await pool.call_tool(server_key, tool, args)
    # MCP results carry content blocks; unwrap text for simplicity.
    if hasattr(result, "content"):
        blocks = result.content
        if blocks and hasattr(blocks[0], "text"):
            return blocks[0].text
    return result


# ---------------------------------------------------------------------------
# Filesystem tools (server-filesystem MCP)
# ---------------------------------------------------------------------------

async def board_read_file(pool: MCPClientPool, path: str) -> str:
    """Read a ticket or document from the project board (data/project_board/)."""
    norm = _ensure_under_root(path, BOARD_ROOT)
    return await _fs_call(pool, "fs_board", "read_file", {"path": norm})


async def board_write_file(pool: MCPClientPool, path: str, content: str) -> str:
    """Create or update a ticket/document on the project board."""
    norm = _ensure_under_root(path, BOARD_ROOT)
    return await _fs_call(pool, "fs_board", "write_file", {"path": norm, "content": content})


async def board_create_directory(pool: MCPClientPool, path: str) -> str:
    """Create a directory on the project board."""
    norm = _ensure_under_root(path, BOARD_ROOT)
    return await _fs_call(pool, "fs_board", "create_directory", {"path": norm})


async def board_list_directory(pool: MCPClientPool, path: str = "") -> str:
    """List entries on the project board."""
    norm = _ensure_under_root(path, BOARD_ROOT)
    return await _fs_call(pool, "fs_board", "list_directory", {"path": norm})


async def board_get_file_info(pool: MCPClientPool, path: str) -> str:
    """Get metadata for a board path."""
    norm = _ensure_under_root(path, BOARD_ROOT)
    return await _fs_call(pool, "fs_board", "get_file_info", {"path": norm})


async def board_read_multiple_files(pool: MCPClientPool, paths: list[str]) -> str:
    """Read multiple board files at once."""
    norm_paths = [_ensure_under_root(p, BOARD_ROOT) for p in paths]
    return await _fs_call(pool, "fs_board", "read_multiple_files", {"paths": norm_paths})


async def docs_read_file(pool: MCPClientPool, path: str) -> str:
    """Read a knowledge-base document (data/knowledge_base/)."""
    norm = _ensure_under_root(path, DOCS_ROOT)
    return await _fs_call(pool, "fs_docs", "read_file", {"path": norm})


async def docs_write_file(pool: MCPClientPool, path: str, content: str) -> str:
    """Create or update a knowledge-base document."""
    norm = _ensure_under_root(path, DOCS_ROOT)
    return await _fs_call(pool, "fs_docs", "write_file", {"path": norm, "content": content})


async def docs_create_directory(pool: MCPClientPool, path: str) -> str:
    """Create a directory in the knowledge base."""
    norm = _ensure_under_root(path, DOCS_ROOT)
    return await _fs_call(pool, "fs_docs", "create_directory", {"path": norm})


async def docs_list_directory(pool: MCPClientPool, path: str = "") -> str:
    """List entries in the knowledge base."""
    norm = _ensure_under_root(path, DOCS_ROOT)
    return await _fs_call(pool, "fs_docs", "list_directory", {"path": norm})


async def docs_get_file_info(pool: MCPClientPool, path: str) -> str:
    """Get metadata for a knowledge-base path."""
    norm = _ensure_under_root(path, DOCS_ROOT)
    return await _fs_call(pool, "fs_docs", "get_file_info", {"path": norm})


async def docs_read_multiple_files(pool: MCPClientPool, paths: list[str]) -> str:
    """Read multiple knowledge-base files at once."""
    norm_paths = [_ensure_under_root(p, DOCS_ROOT) for p in paths]
    return await _fs_call(pool, "fs_docs", "read_multiple_files", {"paths": norm_paths})


async def code_read_file(pool: MCPClientPool, path: str) -> str:
    norm = _ensure_under_root(path, CODE_ROOT)
    return await _fs_call(pool, "fs_code", "read_file", {"path": norm})


async def code_list_directory(pool: MCPClientPool, path: str = "") -> str:
    norm = _ensure_under_root(path, CODE_ROOT)
    return await _fs_call(pool, "fs_code", "list_directory", {"path": norm})


async def code_get_file_info(pool: MCPClientPool, path: str) -> str:
    norm = _ensure_under_root(path, CODE_ROOT)
    return await _fs_call(pool, "fs_code", "get_file_info", {"path": norm})


async def code_read_multiple_files(pool: MCPClientPool, paths: list[str]) -> str:
    norm_paths = [_ensure_under_root(p, CODE_ROOT) for p in paths]
    return await _fs_call(pool, "fs_code", "read_multiple_files", {"paths": norm_paths})


async def code_write_file(pool: MCPClientPool, path: str, content: str) -> str:
    norm = _ensure_under_root(path, CODE_ROOT)
    return await _fs_call(pool, "fs_code", "write_file", {"path": norm, "content": content})


async def code_create_directory(pool: MCPClientPool, path: str) -> str:
    norm = _ensure_under_root(path, CODE_ROOT)
    return await _fs_call(pool, "fs_code", "create_directory", {"path": norm})


async def code_move_file(pool: MCPClientPool, source: str, destination: str) -> str:
    norm_src = _ensure_under_root(source, CODE_ROOT)
    norm_dst = _ensure_under_root(destination, CODE_ROOT)
    return await _fs_call(pool, "fs_code", "move_file", {"source": norm_src, "destination": norm_dst})


async def code_search_files(pool: MCPClientPool, path: str, pattern: str) -> str:
    norm = _ensure_under_root(path, CODE_ROOT)
    return await _fs_call(pool, "fs_code", "search_files", {"path": norm, "pattern": pattern})


BOARD_TOOLS = [
    board_read_file,
    board_write_file,
    board_create_directory,
    board_list_directory,
    board_get_file_info,
    board_read_multiple_files,
]

DOCS_TOOLS = [
    docs_read_file,
    docs_write_file,
    docs_create_directory,
    docs_list_directory,
    docs_get_file_info,
    docs_read_multiple_files,
]

CODE_READ_TOOLS = [
    code_read_file,
    code_list_directory,
    code_get_file_info,
    code_read_multiple_files,
    code_search_files,
]

CODE_WRITE_TOOLS = [
    *CODE_READ_TOOLS,
    code_write_file,
    code_create_directory,
    code_move_file,
]


# ---------------------------------------------------------------------------
# Git tools (mcp-server-git)
# ---------------------------------------------------------------------------

async def _git_call(pool: MCPClientPool, tool: str, args: dict[str, Any]) -> str:
    result = await pool.call_tool("git", tool, args)
    if hasattr(result, "content"):
        blocks = result.content
        if blocks and hasattr(blocks[0], "text"):
            return blocks[0].text
    return str(result)


async def git_status(pool: MCPClientPool) -> str:
    return await _git_call(pool, "git_status", {})


async def git_diff_unstaged(pool: MCPClientPool) -> str:
    return await _git_call(pool, "git_diff_unstaged", {})


async def git_diff_staged(pool: MCPClientPool) -> str:
    return await _git_call(pool, "git_diff_staged", {})


async def git_diff(pool: MCPClientPool, target: str) -> str:
    return await _git_call(pool, "git_diff", {"target": target})


async def git_log(pool: MCPClientPool, max_count: int = 10) -> str:
    return await _git_call(pool, "git_log", {"max_count": max_count})


async def git_show(pool: MCPClientPool, revision: str) -> str:
    return await _git_call(pool, "git_show", {"revision": revision})


async def git_add(pool: MCPClientPool, files: list[str]) -> str:
    return await _git_call(pool, "git_add", {"files": files})


async def git_commit(pool: MCPClientPool, message: str) -> str:
    return await _git_call(pool, "git_commit", {"message": message})


async def git_create_branch(pool: MCPClientPool, branch_name: str, start_point: str | None = None) -> str:
    args: dict[str, Any] = {"branch_name": branch_name}
    if start_point:
        args["start_point"] = start_point
    return await _git_call(pool, "git_create_branch", args)


async def git_checkout(pool: MCPClientPool, branch_name: str) -> str:
    return await _git_call(pool, "git_checkout", {"branch_name": branch_name})


GIT_READ_TOOLS = [
    git_status,
    git_diff_unstaged,
    git_diff_staged,
    git_diff,
    git_log,
    git_show,
]

GIT_WRITE_TOOLS = [
    *GIT_READ_TOOLS,
    git_add,
    git_commit,
    git_create_branch,
    git_checkout,
]


# ---------------------------------------------------------------------------
# Playwright / browser tools (@playwright/mcp)
# ---------------------------------------------------------------------------

async def _pw_call(pool: MCPClientPool, tool: str, args: dict[str, Any]) -> str:
    result = await pool.call_tool("playwright", tool, args)
    if hasattr(result, "content"):
        blocks = result.content
        if blocks and hasattr(blocks[0], "text"):
            return blocks[0].text
    return str(result)


async def browser_navigate(pool: MCPClientPool, url: str) -> str:
    return await _pw_call(pool, "browser_navigate", {"url": url})


async def browser_screenshot(pool: MCPClientPool) -> str:
    return await _pw_call(pool, "browser_screenshot", {})


async def browser_click(pool: MCPClientPool, selector: str) -> str:
    return await _pw_call(pool, "browser_click", {"selector": selector})


async def browser_type(pool: MCPClientPool, selector: str, text: str) -> str:
    return await _pw_call(pool, "browser_type", {"selector": selector, "text": text})


async def browser_get_text(pool: MCPClientPool, selector: str) -> str:
    return await _pw_call(pool, "browser_get_text", {"selector": selector})


async def browser_evaluate(pool: MCPClientPool, script: str) -> str:
    return await _pw_call(pool, "browser_evaluate", {"script": script})


async def browser_close(pool: MCPClientPool) -> str:
    return await _pw_call(pool, "browser_close", {})


PLAYWRIGHT_TOOLS = [
    browser_navigate,
    browser_screenshot,
    browser_click,
    browser_type,
    browser_get_text,
    browser_evaluate,
    browser_close,
]


# ---------------------------------------------------------------------------
# bind_tools helper (captures MCPClientPool for AutoGen)
# ---------------------------------------------------------------------------

def bind_tools(pool: MCPClientPool, *tool_fns: Callable) -> list[Callable]:
    """
    Bind `pool` as the first argument of each tool function and return a list
    of AutoGen-compatible async callables.

    The returned callables:
    - keep the original ``__name__`` and ``__doc__``
    - have a signature that matches everything *after* the ``pool`` parameter
    - are always async (sync functions are wrapped with asyncio.to_thread)
    """
    bound: list[Callable] = []
    for fn in tool_fns:
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())
        inner_params = params[1:]  # drop the pool param
        new_sig = sig.replace(parameters=inner_params)

        if asyncio.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def _async_wrapper(*args: Any, _fn=fn, _pool=pool, **kwargs: Any) -> Any:
                return await _fn(_pool, *args, **kwargs)

            _async_wrapper.__signature__ = new_sig  # type: ignore[attr-defined]
            bound.append(_async_wrapper)
        else:

            @functools.wraps(fn)
            async def _sync_wrapper(*args: Any, _fn=fn, _pool=pool, **kwargs: Any) -> Any:
                return await asyncio.to_thread(_fn, _pool, *args, **kwargs)

            _sync_wrapper.__signature__ = new_sig  # type: ignore[attr-defined]
            bound.append(_sync_wrapper)

    return bound


__all__ = [
    "BOARD_TOOLS",
    "DOCS_TOOLS",
    "CODE_READ_TOOLS",
    "CODE_WRITE_TOOLS",
    "GIT_READ_TOOLS",
    "GIT_WRITE_TOOLS",
    "PLAYWRIGHT_TOOLS",
    "bind_tools",
]

