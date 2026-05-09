"""
Engineer agent, backed by mini-swe-agent.

Receives implementation tasks from the ProjectManager (guided by the
Architect's design), writes or refactors code in the workspace, commits
via git, and reports back.
"""
from __future__ import annotations

import asyncio
import itertools
import json
import logging
import os
import shlex
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import yaml
from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import BaseChatMessage, HandoffMessage, TextMessage
from autogen_core import CancellationToken

from core.mcp_client import MCPClientPool
from core.mcp_config import BOARD_PATH, CODE_PATH, DOCS_PATH, ROLE_SERVERS
from core.swebench import get_role_system_message
from core.telemetry import record_tool_event

logger = logging.getLogger(__name__)

os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.environ.setdefault(
    "MSWEA_GLOBAL_CONFIG_DIR",
    str(_PROJECT_ROOT / "logs" / "mini_swe_agent_config"),
)
Path(os.environ["MSWEA_GLOBAL_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)

from minisweagent import package_dir  # noqa: E402
from minisweagent.agents.default import DefaultAgent  # noqa: E402
from minisweagent.environments.local import LocalEnvironment  # noqa: E402
from minisweagent.models.litellm_model import LitellmModel  # noqa: E402


_DEFAULT_CONFIG_PATH = Path(package_dir) / "config" / "default.yaml"
_DEFAULT_CONFIG_CACHE: dict[str, Any] | None = None
_ESCALATION_MARKER = "ESCALATE_TO_PROJECT_MANAGER"


def _load_default_config() -> dict[str, Any]:
    """Load mini-swe-agent's bundled default templates once per process."""
    global _DEFAULT_CONFIG_CACHE
    if _DEFAULT_CONFIG_CACHE is None:
        _DEFAULT_CONFIG_CACHE = yaml.safe_load(_DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    return _DEFAULT_CONFIG_CACHE


def _resolve_model_name() -> str:
    """Return the LiteLLM-compatible model used by mini-swe-agent."""
    return os.environ.get("MINI_AGENT_MODEL") or os.environ.get("AUTOGEN_MODEL") or "gpt-4o"


def _resolve_cost_limit() -> float:
    raw = os.environ.get("MINI_AGENT_COST_LIMIT")
    if raw is None:
        return 3.0
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid MINI_AGENT_COST_LIMIT=%r, using 3.0", raw)
        return 3.0


def _resolve_step_limit() -> int:
    raw = os.environ.get("MINI_AGENT_STEP_LIMIT")
    if raw is None:
        return 0
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid MINI_AGENT_STEP_LIMIT=%r, using 0", raw)
        return 0


def _resolve_mcp_timeout() -> float:
    raw = os.environ.get("MINI_AGENT_MCP_TIMEOUT")
    if raw is None:
        return 120.0
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid MINI_AGENT_MCP_TIMEOUT=%r, using 120", raw)
        return 120.0


class MCPLocalEnvironment(LocalEnvironment):
    """Local bash environment plus MCP pseudo-command support."""

    def __init__(
        self,
        *,
        pool: MCPClientPool,
        parent_loop: asyncio.AbstractEventLoop,
        allowed_servers: tuple[str, ...] | None = None,
        mcp_timeout: float = 120.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._pool = pool
        self._parent_loop = parent_loop
        self._allowed_servers = (
            tuple(allowed_servers)
            if allowed_servers is not None
            else tuple(ROLE_SERVERS.get("engineer", []))
        )
        self._mcp_timeout = mcp_timeout

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        command = (action.get("command") or "").lstrip()
        if command.startswith("mcp_call"):
            return self._handle_mcp_call(command)
        return super().execute(action, cwd, timeout=timeout)

    def _handle_mcp_call(self, command: str) -> dict[str, Any]:
        try:
            tokens = shlex.split(command, posix=True)
        except ValueError as exc:
            return self._mcp_error(f"failed to parse mcp_call: {exc}")

        if len(tokens) < 3 or len(tokens) > 4:
            return self._mcp_error(
                "Usage: mcp_call <server> <tool> [JSON_ARGS]\n"
                f"got {len(tokens) - 1} arg(s) after 'mcp_call'."
            )

        _, server, tool, *rest = tokens
        json_args = rest[0] if rest else "{}"

        if self._allowed_servers and server not in self._allowed_servers:
            return self._mcp_error(
                f"server '{server}' is not in this agent's allowed list "
                f"({list(self._allowed_servers)})."
            )

        try:
            arguments = json.loads(json_args) if json_args else {}
        except json.JSONDecodeError as exc:
            return self._mcp_error(f"JSON_ARGS is not valid JSON: {exc}\nGot: {json_args!r}")

        if not isinstance(arguments, dict):
            return self._mcp_error(f"JSON_ARGS must be a JSON object, got {type(arguments).__name__}")

        try:
            future = asyncio.run_coroutine_threadsafe(
                self._pool.call_tool(server, tool, arguments),
                self._parent_loop,
            )
            raw_result = future.result(timeout=self._mcp_timeout)
        except Exception as exc:  # noqa: BLE001
            record_tool_event(
                tool,
                False,
                server_key=server,
                via="mini_swe_agent",
                error=str(exc),
            )
            return {
                "output": f"mcp_call {server} {tool} FAILED: {exc}",
                "returncode": 1,
                "exception_info": traceback.format_exc(),
            }

        record_tool_event(tool, True, server_key=server, via="mini_swe_agent")
        return {
            "output": self._stringify_mcp_result(raw_result),
            "returncode": 0,
            "exception_info": "",
        }

    @staticmethod
    def _mcp_error(message: str) -> dict[str, Any]:
        return {
            "output": f"mcp_call ERROR: {message}",
            "returncode": 2,
            "exception_info": "",
        }

    @staticmethod
    def _stringify_mcp_result(result: Any) -> str:
        if hasattr(result, "content"):
            blocks = result.content
            if blocks and hasattr(blocks[0], "text"):
                return blocks[0].text
        return str(result)


_DEFAULT_BRIEFING = f"""\
You are Charlie, the Software Engineer for a decentralized multi-agent
SDLC mesh. The Architect, ProjectManager, CodeReviewer, or QA may hand
work to you. Your job is to implement the requested code change in the
workspace, validate it, commit it, and hand off to CodeReviewer.

Mesh workflow:
- Default path: Architect/ProjectManager -> Engineer -> CodeReviewer.
- If CodeReviewer or QA sends issues back, fix them and hand off to
  CodeReviewer again.
- Only escalate to ProjectManager if you are genuinely blocked by
  unclear scope, missing design, or a decision only PM can make. If you
  must escalate, start your final submission with:
  {_ESCALATION_MARKER}

Operating rules:
- Treat the workspace (your cwd) as the project root.
- Do not modify files outside the data directories.
- For greenfield work, scaffold a complete project structure with real
  working code. Avoid TODO stubs and placeholders.
- For bug fixes, prefer minimal edits to existing files.
- Read relevant ticket(s) and design docs before changing code.
- Update assigned ticket(s): TO DO -> IN PROGRESS when you start, and
  IN REVIEW when you hand off to CodeReviewer.
- Commit your changes before final submission.
"""


def _build_engineer_briefing() -> str:
    return get_role_system_message("engineer", _DEFAULT_BRIEFING)


def _build_paths_block() -> str:
    return (
        "## Workspace layout\n"
        f"- Workspace (cwd):   {CODE_PATH}\n"
        f"- Project board:     {BOARD_PATH}\n"
        f"- Knowledge base:    {DOCS_PATH}\n"
    )


def _build_mcp_block() -> str:
    return (
        "## MCP tools (in addition to bash)\n"
        "You can issue MCP tool calls as actions. Syntax:\n\n"
        "    mcp_call <server> <tool> '<JSON_ARGS>'\n\n"
        "Available servers:\n"
        "- fs_board  -> data/project_board/   (tickets and index)\n"
        "- fs_docs   -> data/knowledge_base/  (design and architecture docs)\n"
        "- fs_code   -> data/workspace/       (source code)\n"
        "- git       -> workspace repo        (status, diff, add, commit, etc.)\n\n"
        "Filesystem tool names: read_file, write_file, list_directory, "
        "create_directory, get_file_info, read_multiple_files, move_file, "
        "search_files. Use absolute paths.\n\n"
        "Git tool names: git_status, git_diff_unstaged, git_diff_staged, "
        "git_diff, git_log, git_show, git_add, git_commit, git_create_branch, "
        "git_checkout. The repo path is set automatically.\n\n"
        "Examples:\n"
        "    mcp_call fs_board list_directory '{\"path\":\"" + BOARD_PATH + "\"}'\n"
        "    mcp_call fs_board write_file '{\"path\":\"" + BOARD_PATH + "/tickets/T-XXX.md\",\"content\":\"...\"}'\n"
        "    mcp_call git git_status '{}'\n"
        "    mcp_call git git_add '{\"files\":[\"main.py\"]}'\n"
        "    mcp_call git git_commit '{\"message\":\"feat: implement T-XXX\"}'\n\n"
        "Bash and mcp_call coexist. Pick whichever is more ergonomic per step.\n"
    )


def _format_messages_for_task(messages: Sequence[BaseChatMessage]) -> str:
    if not messages:
        return "(no task message provided)"
    chunks: list[str] = []
    for msg in messages:
        source = getattr(msg, "source", "?")
        content = getattr(msg, "content", None)
        chunks.append(f"--- from {source} ---\n{content if content is not None else str(msg)}".rstrip())
    return "\n\n".join(chunks)


def _build_task_prompt(messages: Sequence[BaseChatMessage]) -> str:
    return (
        f"{_build_engineer_briefing()}\n"
        f"{_build_paths_block()}\n"
        f"{_build_mcp_block()}\n"
        "## Incoming request from the team\n"
        f"{_format_messages_for_task(messages)}\n"
    )


class _MiniEngineerAgent(BaseChatAgent):
    """AutoGen ChatAgent whose on_messages runs a mini-swe-agent loop."""

    DEFAULT_DESCRIPTION = (
        "Charlie, the Software Engineer. Implemented by mini-swe-agent with "
        "bash plus scoped MCP access; hands off to CodeReviewer by default."
    )

    def __init__(
        self,
        *,
        pool: MCPClientPool,
        name: str = "Engineer",
        description: str | None = None,
        review_target: str = "CodeReviewer",
        escalation_target: str = "ProjectManager",
    ) -> None:
        super().__init__(name=name, description=description or self.DEFAULT_DESCRIPTION)
        self._pool = pool
        self._review_target = review_target
        self._escalation_target = escalation_target
        self._turn_counter = itertools.count(1)

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (HandoffMessage, TextMessage)

    async def on_messages(
        self,
        messages: Sequence[BaseChatMessage],
        cancellation_token: CancellationToken,
    ) -> Response:
        del cancellation_token
        turn = next(self._turn_counter)
        task_prompt = _build_task_prompt(messages)
        traj_path = self._trajectory_path(turn)

        loop = asyncio.get_running_loop()
        started = time.monotonic()
        try:
            result = await asyncio.to_thread(self._run_mini_agent, task_prompt, traj_path, loop)
        except Exception as exc:  # noqa: BLE001
            logger.exception("mini-swe-agent crashed during turn %d", turn)
            elapsed = time.monotonic() - started
            content = (
                f"Engineer (mini-swe-agent) FAILED on turn {turn} after "
                f"{elapsed:.1f}s: {type(exc).__name__}: {exc}"
            )
            return Response(
                chat_message=HandoffMessage(
                    source=self.name,
                    target=self._escalation_target,
                    content=content,
                )
            )

        elapsed = time.monotonic() - started
        target, result = self._choose_handoff_target(result)
        content = self._format_summary(turn, elapsed, result, target)
        return Response(
            chat_message=HandoffMessage(
                source=self.name,
                target=target,
                content=content,
            )
        )

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        del cancellation_token
        self._turn_counter = itertools.count(1)

    def _run_mini_agent(
        self,
        task_prompt: str,
        traj_path: Path,
        parent_loop: asyncio.AbstractEventLoop,
    ) -> dict[str, Any]:
        config = _load_default_config()
        agent_cfg = dict(config.get("agent", {}))
        env_cfg = dict(config.get("environment", {}))
        model_cfg = dict(config.get("model", {}))

        agent_cfg["cost_limit"] = _resolve_cost_limit()
        agent_cfg["step_limit"] = _resolve_step_limit()
        agent_cfg["output_path"] = traj_path

        workspace = os.environ.get("MAS_WORKSPACE_PATH", CODE_PATH)
        env_cfg.setdefault("env", {})
        env_cfg["cwd"] = workspace

        model_name = _resolve_model_name()
        logger.info(
            "[Engineer/mini] starting model=%s cwd=%s cost_limit=%s step_limit=%s",
            model_name,
            workspace,
            agent_cfg["cost_limit"],
            agent_cfg["step_limit"],
        )

        env = MCPLocalEnvironment(
            pool=self._pool,
            parent_loop=parent_loop,
            mcp_timeout=_resolve_mcp_timeout(),
            **env_cfg,
        )
        model = LitellmModel(model_name=model_name, **model_cfg)
        inner = DefaultAgent(model, env, **agent_cfg)

        try:
            extra = inner.run(task_prompt) or {}
        except Exception as exc:  # noqa: BLE001
            logger.exception("[Engineer/mini] DefaultAgent.run raised")
            return {
                "exit_status": type(exc).__name__,
                "submission": "",
                "error": str(exc),
                "trajectory": str(traj_path),
                "n_calls": getattr(inner, "n_calls", 0),
                "cost": getattr(inner, "cost", 0.0),
            }

        return {
            "exit_status": extra.get("exit_status", ""),
            "submission": extra.get("submission", ""),
            "trajectory": str(traj_path),
            "n_calls": getattr(inner, "n_calls", 0),
            "cost": getattr(inner, "cost", 0.0),
        }

    def _choose_handoff_target(self, result: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        cleaned = dict(result)
        submission = (cleaned.get("submission") or "").strip()
        if result.get("exit_status") != "Submitted" or "error" in result:
            return self._escalation_target, cleaned
        if submission.startswith(_ESCALATION_MARKER):
            cleaned["submission"] = submission.removeprefix(_ESCALATION_MARKER).lstrip(" :\n\t")
            return self._escalation_target, cleaned
        return self._review_target, cleaned

    @staticmethod
    def _trajectory_path(turn: int) -> Path:
        logs_dir = _PROJECT_ROOT / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        run_id = os.environ.get("MAS_RUN_ID")
        stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
        suffix = run_id or stamp
        return logs_dir / f"mini_traj_{suffix}_turn{turn:02d}.json"

    @staticmethod
    def _format_summary(turn: int, elapsed: float, result: dict[str, Any], target: str) -> str:
        submission = (result.get("submission") or "").strip()
        if len(submission) > 4000:
            submission = submission[:4000] + "\n... [truncated; see trajectory]"

        success = result.get("exit_status") == "Submitted" and "error" not in result
        status = "succeeded" if success else "did not submit cleanly"
        lines = [
            f"Engineer (mini-swe-agent) {status} on turn {turn}; handoff target={target}.",
            (
                f"exit_status={result.get('exit_status', '?')} "
                f"steps={result.get('n_calls', 0)} "
                f"cost=${result.get('cost', 0.0):.4f} "
                f"elapsed={elapsed:.1f}s"
            ),
            f"trajectory: {result.get('trajectory')}",
            "",
            "Engineer summary:",
            submission or "(no submission text returned)",
        ]
        if "error" in result:
            lines.append("")
            lines.append(f"ERROR: {result['error']}")
        return "\n".join(lines)


class Engineer:
    """Engineer role wrapper; self.agent is mini-swe-agent backed."""

    def __init__(self, pool: MCPClientPool) -> None:
        self._pool = pool
        self.agent = _MiniEngineerAgent(pool=pool)
