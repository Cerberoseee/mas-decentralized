"""
MCP server configurations for each logical domain.

Each entry maps a server key to the command + args needed to launch
the corresponding MCP stdio server.  Paths are resolved at import time
from the same BASE_DIR used by agents/config.py so every piece of code
references the same workspace folders.
"""
from __future__ import annotations

import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))

BOARD_PATH = os.path.join(BASE_DIR, "project_board")
DOCS_PATH = os.path.join(BASE_DIR, "knowledge_base")
CODE_PATH = os.path.join(BASE_DIR, "workspace")

# ---------------------------------------------------------------------------
# Server registry
# Each value is passed directly to StdioServerParameters(command=, args=).
# ---------------------------------------------------------------------------

MCP_SERVERS: dict[str, dict] = {
    # @modelcontextprotocol/server-filesystem instances – one per scoped root.
    "fs_board": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", BOARD_PATH],
    },
    "fs_docs": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", DOCS_PATH],
    },
    "fs_code": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", CODE_PATH],
    },
    # mcp-server-git (pip install mcp-server-git)
    "git": {
        "command": "mcp-server-git",
        "args": ["--repository", CODE_PATH],
    },
    # @playwright/mcp  (npx @playwright/mcp@latest)
    "playwright": {
        "command": "npx",
        "args": ["-y", "@playwright/mcp@latest"],
    },
}

# Convenience: which server keys each role may access.
ROLE_SERVERS: dict[str, list[str]] = {
    "project_manager": ["fs_board", "fs_docs"],
    "architect": ["fs_board", "fs_docs", "fs_code"],
    "engineer": ["fs_board", "fs_docs", "fs_code", "git"],
    "code_reviewer": ["fs_docs", "fs_code", "git"],
    "qa": ["fs_board", "fs_code", "playwright"],
}
