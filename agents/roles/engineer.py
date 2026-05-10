"""
Engineer agent — backed by mini-swe-agent.

Loads cost/step/env from mini-swe-agent's ``default.yaml``, but **replaces**
agent/model templates: bundled yaml documents `` ```mswea_bash_command``` **
markdown fences**, while ``LitellmModel`` only executes OpenAI-style ``bash``
tool calls — without that alignment every turn fails with "No tool calls found".
``mcp_call`` is intercepted locally; real bash commands are routed to either
``LocalEnvironment`` (host) or ``DockerEnvironment`` (SWE-bench container).

Docker mode (set ``MINI_AGENT_USE_DOCKER=1``)
---------------------------------------------
When enabled the Engineer starts a per-turn Docker container from the
official SWE-bench per-instance image (derived from ``MAS_EVAL_TASK_ID``).
Every bash command the LLM generates executes via ``docker exec … bash -lc``
so the conda env inside the container is auto-activated — no host-side venv,
no C compilation, no PEP 668.  ``mcp_call`` lines are still dispatched to the
host-side ``MCPClientPool`` as before.

Receives implementation tasks from the ProjectManager, writes or refactors
code in the workspace, commits via git, and hands off to CodeReviewer.
SWE-bench runs emit ``patch.diff`` from
``main._write_patch_if_configured`` after the workflow.
"""
from __future__ import annotations

import asyncio
import itertools
import json
import logging
import os
import shlex
import subprocess
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import yaml

from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import (
    BaseChatMessage,
    HandoffMessage,
    TextMessage,
)
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
from minisweagent.environments.docker import DockerEnvironment  # noqa: E402
from minisweagent.environments.local import LocalEnvironment  # noqa: E402
from minisweagent.models.litellm_model import LitellmModel  # noqa: E402


_DEFAULT_CONFIG_PATH = Path(package_dir) / "config" / "default.yaml"
_DEFAULT_CONFIG_CACHE: dict[str, Any] | None = None
_ESCALATION_MARKER = "ESCALATE_TO_PROJECT_MANAGER"


def _load_default_config() -> dict[str, Any]:
    """Load mini-swe-agent's bundled default templates (cached)."""
    global _DEFAULT_CONFIG_CACHE
    if _DEFAULT_CONFIG_CACHE is None:
        _DEFAULT_CONFIG_CACHE = yaml.safe_load(
            _DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
        )
    return _DEFAULT_CONFIG_CACHE


def _resolve_model_name() -> str:
    """LiteLLM-compatible model name for the inner mini-swe-agent."""
    return (
        os.environ.get("MINI_AGENT_MODEL")
        or os.environ.get("AUTOGEN_MODEL")
        or "gpt-4o"
    )


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


def _resolve_max_engineer_turns() -> int:
    """Maximum number of Engineer turns before a hard stop is issued.

    Prevents destructive loops where the model repeatedly overwrites source
    files with stubs or otherwise makes things worse across many re-invocations.
    Override with ``MINI_AGENT_MAX_ENGINEER_TURNS`` (default: 5).
    """
    raw = os.environ.get("MINI_AGENT_MAX_ENGINEER_TURNS")
    if raw is None:
        return 5
    try:
        val = int(raw)
        if val < 1:
            raise ValueError("must be >= 1")
        return val
    except ValueError:
        logger.warning("Invalid MINI_AGENT_MAX_ENGINEER_TURNS=%r, using 5", raw)
        return 5


def _resolve_mini_cmd_timeout(env_cfg: dict[str, Any]) -> None:
    """Optional override of per-step shell timeout from MINI_AGENT_CMD_TIMEOUT (seconds)."""
    raw = os.environ.get("MINI_AGENT_CMD_TIMEOUT", "").strip()
    if not raw:
        return
    try:
        env_cfg["timeout"] = int(raw)
    except ValueError:
        logger.warning("Invalid MINI_AGENT_CMD_TIMEOUT=%r, ignoring", raw)


def _workspace_venv_bin(workspace: str) -> Path | None:
    """Return ``.../bin`` (or ``.../Scripts``) if a local venv exists under the workspace."""
    root = Path(workspace)
    for name in ("venv", ".venv"):
        for sub in ("bin", "Scripts"):
            candidate = root / name / sub
            is_dir = candidate.is_dir()
            logger.info("[venv-detect] checking %s → is_dir=%s", candidate, is_dir)
            if is_dir:
                return candidate
    logger.info("[venv-detect] no venv found under workspace=%s", workspace)
    return None


def _posix_bash_executable() -> str | None:
    """Use bash for ``shell=True`` so agents can use ``source`` (``sh`` has no ``source``)."""
    if os.name != "posix":
        return None
    candidate = Path("/bin/bash")
    return str(candidate) if candidate.is_file() else None


def _merge_mini_env_with_venv_on_path(workspace_root: str, base_env: dict[str, str]) -> dict[str, str]:
    """Return a copy of ``base_env`` with workspace ``venv``/``.venv`` prepended to ``PATH`` when present."""
    if os.environ.get("MAS_MINI_AGENT_SKIP_VENV_PATH", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return dict(base_env)
    venv_bin = _workspace_venv_bin(workspace_root)
    if venv_bin is None:
        return dict(base_env)
    out = dict(base_env)
    path = out.get("PATH", "")
    vbin_s = os.fsdecode(venv_bin)
    sep = os.pathsep
    if path == vbin_s or path.startswith(vbin_s + sep):
        out.setdefault("VIRTUAL_ENV", str(venv_bin.parent))
        return out
    out["PATH"] = f"{venv_bin}{sep}{path}"
    out.setdefault("VIRTUAL_ENV", str(venv_bin.parent))
    return out


def _prepend_workspace_venv_to_path(env_cfg: dict[str, Any], workspace: str) -> None:
    """Put workspace ``venv``/``.venv`` first on PATH so ``pip``/``pytest`` match the project interpreter.

    Without this, models often invoke system ``pip`` (PEP 668) while errors in the log refer to
    ``workspace/venv/...``, which wastes turns. Set ``MAS_MINI_AGENT_SKIP_VENV_PATH=1`` to disable.
    """
    if os.environ.get("MAS_MINI_AGENT_SKIP_VENV_PATH", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return
    venv_bin = _workspace_venv_bin(workspace)
    if venv_bin is None:
        return
    env = env_cfg.setdefault("env", {})
    if not isinstance(env, dict):
        return
    # Copy so we do not mutate a shared reference from default.yaml.
    env_cfg["env"] = dict(env)
    env = env_cfg["env"]
    base_path = env.get("PATH", os.environ.get("PATH", ""))
    env["PATH"] = f"{venv_bin}{os.pathsep}{base_path}"
    env.setdefault("VIRTUAL_ENV", str(venv_bin.parent))
    logger.info(
        "[Engineer/mini] prepended workspace venv to PATH: %s (VIRTUAL_ENV=%s)",
        venv_bin,
        env.get("VIRTUAL_ENV"),
    )


def _tool_choice_for_litellm() -> dict[str, str]:
    """Optional ``tool_choice`` for litellm; default forces one tool call per turn."""
    if "MINI_AGENT_TOOL_CHOICE" not in os.environ:
        return {"tool_choice": "required"}
    raw = os.environ["MINI_AGENT_TOOL_CHOICE"].strip()
    if raw.lower() in ("", "no", "none", "false", "0", "off", "unset"):
        return {}
    return {"tool_choice": raw}


def _swebench_image_name(instance_id: str) -> str:
    """Return the Docker Hub image name for a SWE-bench instance.

    Docker doesn't allow ``__`` in image names; the SWE-bench project
    substitutes them with ``_1776_`` (a magic token unlikely to clash).
    """
    docker_id = instance_id.replace("__", "_1776_")
    tag = os.environ.get("MINI_AGENT_DOCKER_IMAGE_TAG", "latest")
    return f"docker.io/swebench/sweb.eval.x86_64.{docker_id}:{tag}".lower()


def _resolve_docker_image() -> str | None:
    """Return the SWE-bench Docker image to use, or None for host execution.

    Activated by setting ``MINI_AGENT_USE_DOCKER=1`` in the environment.
    Requires ``MAS_EVAL_TASK_ID`` to be set (injected by the eval harness).
    """
    if os.environ.get("MINI_AGENT_USE_DOCKER", "").strip().lower() not in ("1", "true", "yes"):
        return None
    instance_id = os.environ.get("MAS_EVAL_TASK_ID", "").strip()
    if not instance_id:
        logger.warning(
            "[Engineer/mini] MINI_AGENT_USE_DOCKER=1 but MAS_EVAL_TASK_ID is not set; "
            "falling back to LocalEnvironment"
        )
        return None
    return _swebench_image_name(instance_id)


# LitellmModel parses OpenAI ``bash`` tool_calls only — not ```mswea_bash_command``` fences from default.yaml.
_MINI_LLM_SYSTEM_TEMPLATE = """\
You are a helpful assistant that can interact with a computer.

Every turn you MUST call the provided `bash` tool exactly once. The runtime does not execute
markdown, prose-only answers, or ```mswea_bash_command``` code fences — only the `bash` tool call
runs shell commands.

Pass a JSON object with a single key: `command` (the shell string). You may add brief reasoning
in normal assistant text, but the action that runs is always the `bash` tool call.

Rules for `command`:
- It runs with **bash** (POSIX), not plain `sh`, so `source venv/bin/activate` works when needed.
- If the workspace has `venv/` or `.venv/`, that environment's `bin` is prepended to `PATH` for each command (after it exists), so `pip`/`pytest` match the project interpreter—avoid bare system `pip` on PEP 668 hosts.
- Combine steps with `&&` or `||` on one line when needed.
- MCP tools: start the command with `mcp_call <server> <tool> '<JSON>'` (see task message for servers).
- **Never** run `git push` — there is no remote; commits stay local only.
- To finish the task, your **last** `bash` call must be **only**: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`
  (do not chain other commands on that final step).
"""

_MINI_LLM_FORMAT_ERROR_TEMPLATE = """\
Format error:

<error>
{{error}}
</error>

The API requires a **`bash` tool call** with JSON like: {"command": "your shell command here"}.
Your last response had {{actions|length}} tool call(s) (need exactly one valid `bash` call).

Do not use markdown code fences for commands; they are ignored.

To abandon the task early (only if appropriate), call `bash` with:
{"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"}
"""

_MINI_LLM_INSTANCE_TEMPLATE = """\
Please solve this issue: {{task}}

Execute work by calling the **`bash` tool** each turn (see system message). Optional MCP lines use
`mcp_call` inside the shell command string.

## Recommended workflow

1. Analyze the codebase by finding and reading relevant files.
2. Create a small script or command to reproduce the issue when practical.
3. Edit the source code to resolve the issue.
4. Verify the fix (run targeted tests or your repro).
5. Commit when ready (`mcp_call git ...` or the platform `git` CLI in `bash`). Do **not** push — there is no remote.
6. Finish by calling `bash` with command exactly: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`
   (that step alone; after that you cannot continue).

## Important rules

1. One `bash` tool call per turn — no prose-only replies.
2. Directory changes are not persistent across turns; use `cd /path && ...` in `command` when needed.

<system_information>
{{system}} {{release}} {{version}} {{machine}}
</system_information>

## Useful patterns (inside the `command` string)

### Create / overwrite a file

Use heredoc or your platform's file redirection inside the `command` value.

### Edit with sed

{% if system == "Darwin" %}
<important>
You are on macOS: use `sed -i ''` for in-place edits.
</important>
{% endif %}

Example: `sed -i 's/old/new/g' path/to/file.py` (on Linux/Git Bash). On macOS use `sed -i '' 's/old/new/g' ...`.

### View part of a file

Example: `nl -ba path/to/file.py | sed -n '10,30p'`
"""


def _apply_litellm_tool_templates(agent_cfg: dict[str, Any], model_cfg: dict[str, Any]) -> None:
    """Align prompts with LitellmModel (tool calls), overriding incompatible default.yaml fences."""
    agent_cfg["system_template"] = _MINI_LLM_SYSTEM_TEMPLATE
    agent_cfg["instance_template"] = _MINI_LLM_INSTANCE_TEMPLATE
    model_cfg["format_error_template"] = _MINI_LLM_FORMAT_ERROR_TEMPLATE
    merged_mk = dict(model_cfg.get("model_kwargs") or {})
    merged_mk.update(_tool_choice_for_litellm())
    model_cfg["model_kwargs"] = merged_mk


# ---------------------------------------------------------------------------
# MCP-aware environment
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Shared MCP dispatch mixin
# ---------------------------------------------------------------------------


class _MCPDispatchMixin:
    """Mixin that intercepts ``mcp_call`` commands and routes them to the
    host-side ``MCPClientPool``.

    Concrete subclasses must set the following attributes in their
    ``__init__`` before calling any ``execute``:

    * ``_pool`` — ``MCPClientPool``
    * ``_parent_loop`` — ``asyncio.AbstractEventLoop``
    * ``_allowed_servers`` — ``tuple[str, ...]``
    * ``_mcp_timeout`` — ``float``

    Action grammar for mcp_call::

        mcp_call <server_key> <tool_name> [JSON_ARGS]

    The MCP call is dispatched onto the parent asyncio loop via
    ``run_coroutine_threadsafe`` because mini-swe-agent runs synchronously
    in a worker thread while the ``MCPClientPool`` lives on the main loop.
    """

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
            return self._mcp_error(
                f"JSON_ARGS is not valid JSON: {exc}\nGot: {json_args!r}"
            )

        if not isinstance(arguments, dict):
            return self._mcp_error(
                f"JSON_ARGS must be a JSON object, got {type(arguments).__name__}"
            )

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
        """Mirror what core.mcp_tools._fs_call does for tool results."""
        if hasattr(result, "content"):
            blocks = result.content
            if blocks and hasattr(blocks[0], "text"):
                return blocks[0].text
        return str(result)


# ---------------------------------------------------------------------------
# Environment implementations
# ---------------------------------------------------------------------------


class MCPLocalEnvironment(_MCPDispatchMixin, LocalEnvironment):
    """``LocalEnvironment`` that intercepts ``mcp_call`` pseudo-commands.

    Real bash commands run on the host via ``subprocess``; ``mcp_call``
    lines are dispatched to the host-side ``MCPClientPool``.
    """

    def __init__(
        self,
        *,
        pool: MCPClientPool,
        parent_loop: asyncio.AbstractEventLoop,
        allowed_servers: tuple[str, ...] | None = None,
        mcp_timeout: float = 120.0,
        **kwargs: Any,
    ) -> None:
        LocalEnvironment.__init__(self, **kwargs)
        self._pool = pool
        self._parent_loop = parent_loop
        self._allowed_servers = (
            tuple(allowed_servers)
            if allowed_servers is not None
            else tuple(ROLE_SERVERS.get("engineer", []))
        )
        self._mcp_timeout = mcp_timeout

    # The base class signature in mini-swe-agent v2 is:
    #   execute(self, action, cwd="", *, timeout=None) -> dict
    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        command = (action.get("command") or "").lstrip()
        if command.startswith("mcp_call"):
            return self._handle_mcp_call(command)
        return self._execute_shell_like_local_env(action, cwd, timeout=timeout)

    def _execute_shell_like_local_env(
        self, action: dict, cwd: str = "", *, timeout: int | None = None
    ) -> dict[str, Any]:
        """Mirror ``LocalEnvironment.execute`` but (1) merge venv PATH every call and (2) use bash on POSIX.

        Startup-time PATH injection misses ``venv/`` created mid-run; checking the filesystem each
        turn fixes PEP 668 loops from system ``pip``. Using ``/bin/bash`` fixes ``source: not found``.
        """
        shell_cmd = action.get("command", "")
        cwd_resolved = cwd or self.config.cwd or os.getcwd()
        merged = {**os.environ, **self.config.env}
        env = _merge_mini_env_with_venv_on_path(cwd_resolved, merged)
        run_kw: dict[str, Any] = {
            "shell": True,
            "text": True,
            "cwd": cwd_resolved,
            "env": env,
            "timeout": timeout or self.config.timeout,
            "encoding": "utf-8",
            "errors": "replace",
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
        }
        bash_exe = _posix_bash_executable()
        if bash_exe:
            run_kw["executable"] = bash_exe
        try:
            result = subprocess.run(shell_cmd, **run_kw)
            output: dict[str, Any] = {
                "output": result.stdout,
                "returncode": result.returncode,
                "exception_info": "",
            }
        except Exception as exc:  # noqa: BLE001 — match LocalEnvironment
            raw_output = getattr(exc, "output", None)
            raw_output = (
                raw_output.decode("utf-8", errors="replace")
                if isinstance(raw_output, bytes)
                else (raw_output or "")
            )
            output = {
                "output": raw_output,
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {exc}",
                "extra": {"exception_type": type(exc).__name__, "exception": str(exc)},
            }
        self._check_finished(output)
        return output


class MCPDockerEnvironment(_MCPDispatchMixin, DockerEnvironment):
    """``DockerEnvironment`` that intercepts ``mcp_call`` pseudo-commands.

    Real bash commands execute via ``docker exec … bash -lc`` inside the
    SWE-bench per-instance container.  The login shell (``-l``) auto-activates
    the conda ``testbed`` environment so ``pytest``/``pip``/``python`` all
    resolve to the correct pre-compiled interpreter — no host-side venv or C
    compilation required.

    ``mcp_call`` lines are dispatched to the host-side ``MCPClientPool``
    without touching the container, preserving git-MCP, fs-MCP, etc.
    """

    def __init__(
        self,
        *,
        pool: MCPClientPool,
        parent_loop: asyncio.AbstractEventLoop,
        allowed_servers: tuple[str, ...] | None = None,
        mcp_timeout: float = 120.0,
        **kwargs: Any,
    ) -> None:
        DockerEnvironment.__init__(self, **kwargs)  # pulls image + starts container
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
        # Route to DockerEnvironment.execute → docker exec … bash -lc
        return DockerEnvironment.execute(self, action, cwd, timeout=timeout)


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


_DEFAULT_BRIEFING = """\
You are Charlie, the Software Engineer for a multi-agent SDLC team. The
Project Manager (Alice) has already produced tickets and any design docs.
Your job is to make the code in the workspace satisfy what they asked for,
then commit and hand off to CodeReviewer.

Operating rules:
- Treat the workspace (your cwd) as the project root. Do NOT modify
  anything outside the data/ directories.
- For a greenfield request: scaffold a complete project structure and
  implement real working code (no TODO stubs, no placeholders).
- For a bug-fix request: prefer minimal edits to existing files.
- Read the relevant ticket(s) and any design docs the team produced
  before writing code. Update each ticket's Status field and append a
  brief Update note as your work progresses.
- Commit your changes via git when they are ready for review.
- If you are genuinely blocked, start your final submission with
  ESCALATE_TO_PROJECT_MANAGER.
"""


def _build_engineer_briefing() -> str:
    """Per-role briefing — switches to SWE-bench mode if MAS_MODE=swebench."""
    return get_role_system_message("engineer", _DEFAULT_BRIEFING)


def _build_paths_block() -> str:
    # In Docker mode the Engineer's bash commands run inside the SWE-bench
    # container where the repo lives at /testbed, not at the host CODE_PATH.
    # Showing the host path here confuses the agent into navigating somewhere
    # that doesn't exist inside the container.
    if os.environ.get("MINI_AGENT_USE_DOCKER", "").strip().lower() in ("1", "true", "yes"):
        workspace_path = "/testbed"
    else:
        workspace_path = CODE_PATH
    return (
        "## Workspace layout\n"
        f"- Workspace (cwd):   {workspace_path}\n"
        f"- Project board:     {BOARD_PATH}\n"
        f"- Knowledge base:    {DOCS_PATH}\n"
    )


def _build_mcp_block() -> str:
    return (
        "## MCP tools (in addition to bash)\n"
        "You can issue MCP tool calls as actions. Syntax:\n\n"
        "    mcp_call <server> <tool> '<JSON_ARGS>'\n\n"
        "Available servers (and their scope):\n"
        "- fs_board  →  data/project_board/   (tickets and the index)\n"
        "- fs_docs   →  data/knowledge_base/  (design + architecture docs)\n"
        "- fs_code   →  data/workspace/       (the project source code)\n"
        "- git       →  the workspace repo    (status / add / commit / log / etc.)\n\n"
        "Filesystem tool names: read_file, write_file, list_directory,\n"
        "create_directory, get_file_info, read_multiple_files, move_file,\n"
        "search_files. Paths must be absolute.\n\n"
        "Git tool names: git_status, git_diff_unstaged, git_diff_staged,\n"
        "git_diff, git_log, git_show, git_add, git_commit, git_create_branch,\n"
        "git_checkout. The repo path is set automatically.\n\n"
        "Examples:\n"
        "    mcp_call fs_board list_directory '{\"path\":\"" + BOARD_PATH + "\"}'\n"
        "    mcp_call fs_board write_file '{\"path\":\"" + BOARD_PATH + "/tickets/T-XXX.md\",\"content\":\"...\"}'\n"
        "    mcp_call git git_status '{}'\n"
        "    mcp_call git git_add '{\"files\":[\"main.py\"]}'\n"
        "    mcp_call git git_commit '{\"message\":\"feat: implement T-XXX\"}'\n\n"
        "Bash and mcp_call coexist — pick whichever is more ergonomic per\n"
        "step. Direct git CLI commands also work (the workspace IS a git repo).\n"
    )


def _format_messages_for_task(messages: Sequence[BaseChatMessage]) -> str:
    """Render the incoming Swarm messages into a task block for mini-swe-agent."""
    if not messages:
        return "(no task message provided)"
    chunks: list[str] = []
    for msg in messages:
        source = getattr(msg, "source", "?")
        content = getattr(msg, "content", None)
        if content is None:
            content = str(msg)
        chunks.append(f"--- from {source} ---\n{content}".rstrip())
    return "\n\n".join(chunks)


def _build_task_prompt(messages: Sequence[BaseChatMessage]) -> str:
    return (
        f"{_build_engineer_briefing()}\n"
        f"{_build_paths_block()}\n"
        f"{_build_mcp_block()}\n"
        "## Incoming request from the team\n"
        f"{_format_messages_for_task(messages)}\n"
    )


# ---------------------------------------------------------------------------
# AutoGen agent wrapper
# ---------------------------------------------------------------------------


class _MiniEngineerAgent(BaseChatAgent):
    """AutoGen ``ChatAgent`` whose ``on_messages`` runs a mini-swe-agent loop."""

    DEFAULT_DESCRIPTION = (
        "Charlie, the Software Engineer. Backed by mini-swe-agent: "
        "delegates each task to a fresh bash-driven agent loop with MCP "
        "tool access, then hands control to CodeReviewer (or ProjectManager on escalation)."
    )

    def __init__(
        self,
        *,
        pool: MCPClientPool,
        name: str = "Engineer",
        description: str | None = None,
        handoff_target: str = "CodeReviewer",
        escalation_target: str = "ProjectManager",
    ) -> None:
        super().__init__(name=name, description=description or self.DEFAULT_DESCRIPTION)
        self._pool = pool
        self._handoff_target = handoff_target
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
        del cancellation_token  # mini-swe-agent runs to completion or limit
        turn = next(self._turn_counter)

        max_turns = _resolve_max_engineer_turns()
        if turn > max_turns:
            logger.warning(
                "[Engineer] Hard stop: turn %d exceeds max_engineer_turns=%d. "
                "Escalating to ProjectManager to prevent destructive loops.",
                turn,
                max_turns,
            )
            content = (
                f"ESCALATE_TO_PROJECT_MANAGER: Engineer hard-stop triggered on turn {turn} "
                f"(limit: {max_turns}). "
                "The agent has been re-invoked too many times without resolving the task — "
                "continuing risks further destructive changes (e.g. overwriting source files with stubs). "
                "Please mark the task as failed or declare the best available state as the final submission."
            )
            return Response(
                chat_message=HandoffMessage(
                    source=self.name,
                    target=self._escalation_target,
                    content=content,
                )
            )

        task_prompt = _build_task_prompt(messages)
        traj_path = self._trajectory_path(turn)

        loop = asyncio.get_running_loop()
        started = time.monotonic()
        try:
            result = await asyncio.to_thread(
                self._run_mini_agent, task_prompt, traj_path, loop
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("mini-swe-agent crashed during turn %d", turn)
            elapsed = time.monotonic() - started
            content = (
                f"Engineer (mini-swe-agent) FAILED on turn {turn} after "
                f"{elapsed:.1f}s: {type(exc).__name__}: {exc}"
            )
            handoff = HandoffMessage(
                source=self.name,
                target=self._escalation_target,
                content=content,
            )
            return Response(chat_message=handoff)

        elapsed = time.monotonic() - started
        target, result = self._choose_handoff_target(result)
        content = self._format_summary(turn, elapsed, result, target)
        handoff = HandoffMessage(
            source=self.name,
            target=target,
            content=content,
        )
        return Response(chat_message=handoff)

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        del cancellation_token
        self._turn_counter = itertools.count(1)

    def _choose_handoff_target(self, result: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        cleaned = dict(result)
        submission = (cleaned.get("submission") or "").strip()
        if result.get("exit_status") != "Submitted" or "error" in result:
            return self._escalation_target, cleaned
        if submission.startswith(_ESCALATION_MARKER):
            cleaned["submission"] = submission.removeprefix(_ESCALATION_MARKER).lstrip(" :\n\t")
            return self._escalation_target, cleaned
        return self._handoff_target, cleaned

    # ------------------------------------------------------------------
    # mini-swe-agent driver
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_patch_from_docker(env: MCPDockerEnvironment, *, initial_head: str | None = None) -> None:
        """Write patch.diff by running ``git diff`` inside the live container.

        Called right after ``DefaultAgent.run()`` completes, while the
        container is still alive (it sleeps for 2 h by default).

        ``initial_head`` is the git SHA recorded *before* the agent ran.
        Diffing against it (rather than ``base_commit``) is essential because
        SWE-bench Docker images sometimes contain commits on top of
        ``base_commit`` that were added during image construction (e.g.
        environment pin-downs).  Using ``initial_head`` isolates only the
        changes the agent actually made in this session.
        """
        patch_path_str = os.environ.get("MAS_EVAL_PATCH_PATH")
        if not patch_path_str:
            return
        container_id = getattr(env, "container_id", None)
        if not container_id:
            logger.warning("[Engineer/mini] No container_id on MCPDockerEnvironment; cannot extract patch.")
            return

        if initial_head:
            diff_ref = initial_head
        else:
            # Fallback: try base_commit from task context, then bare git diff.
            diff_ref = None
            ctx_path = os.environ.get("MAS_EVAL_TASK_CONTEXT_PATH")
            if ctx_path:
                try:
                    ctx = json.loads(Path(ctx_path).read_text(encoding="utf-8"))
                    diff_ref = ctx.get("base_commit")
                except Exception:  # noqa: BLE001
                    pass

        diff_cmd = f"git diff {diff_ref}" if diff_ref else "git diff"
        result = subprocess.run(
            ["docker", "exec", container_id, "bash", "-lc", diff_cmd],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if result.returncode == 0:
            patch_path = Path(patch_path_str)
            patch_path.parent.mkdir(parents=True, exist_ok=True)
            patch_path.write_text(result.stdout, encoding="utf-8")
            logger.info(
                "[Engineer/mini] Patch extracted from container %s → %s (%d bytes)",
                container_id[:12],
                patch_path,
                len(result.stdout),
            )
        else:
            logger.warning(
                "[Engineer/mini] git diff inside container failed (rc=%d): %s",
                result.returncode,
                result.stderr[:300],
            )

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
        _apply_litellm_tool_templates(agent_cfg, model_cfg)

        _resolve_mini_cmd_timeout(env_cfg)
        env_cfg.setdefault("env", {})
        model_name = _resolve_model_name()
        mcp_timeout = _resolve_mcp_timeout()

        docker_image = _resolve_docker_image()
        if docker_image:
            logger.info(
                "[Engineer/mini] Docker mode: image=%s cost_limit=%s step_limit=%s",
                docker_image,
                agent_cfg["cost_limit"],
                agent_cfg["step_limit"],
            )
            # Only pass DockerEnvironmentConfig-compatible keys; venv PATH
            # injection is unnecessary — bash -lc activates conda automatically.
            docker_kwargs: dict[str, Any] = {
                "image": docker_image,
                "cwd": "/testbed",
                "env": env_cfg.get("env", {}),
            }
            if "timeout" in env_cfg:
                docker_kwargs["timeout"] = env_cfg["timeout"]
            env: MCPLocalEnvironment | MCPDockerEnvironment = MCPDockerEnvironment(
                pool=self._pool,
                parent_loop=parent_loop,
                mcp_timeout=mcp_timeout,
                **docker_kwargs,
            )
        else:
            workspace = os.environ.get("MAS_WORKSPACE_PATH", CODE_PATH)
            env_cfg["cwd"] = workspace
            _prepend_workspace_venv_to_path(env_cfg, workspace)
            logger.info(
                "[Engineer/mini] Host mode: model=%s cwd=%s cost_limit=%s step_limit=%s",
                model_name,
                workspace,
                agent_cfg["cost_limit"],
                agent_cfg["step_limit"],
            )
            env = MCPLocalEnvironment(
                pool=self._pool,
                parent_loop=parent_loop,
                mcp_timeout=mcp_timeout,
                **env_cfg,
            )
        model = LitellmModel(model_name=model_name, **model_cfg)
        inner = DefaultAgent(model, env, **agent_cfg)

        # Record the container's HEAD *before* the agent runs so that the
        # patch diff only covers the agent's own changes, not any commits that
        # the SWE-bench image may have added on top of base_commit during build.
        initial_head: str | None = None
        if isinstance(env, MCPDockerEnvironment):
            head_result = subprocess.run(
                ["docker", "exec", env.container_id, "git", "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=15, check=False,
            )
            if head_result.returncode == 0:
                initial_head = head_result.stdout.strip()
                logger.info("[Engineer/mini] Container initial HEAD: %s", initial_head)
            else:
                logger.warning("[Engineer/mini] Could not read container HEAD: %s", head_result.stderr[:200])

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

        # Extract patch before the container is cleaned up.
        if isinstance(env, MCPDockerEnvironment):
            self._extract_patch_from_docker(env, initial_head=initial_head)

        return {
            "exit_status": extra.get("exit_status", ""),
            "submission": extra.get("submission", ""),
            "trajectory": str(traj_path),
            "n_calls": getattr(inner, "n_calls", 0),
            "cost": getattr(inner, "cost", 0.0),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Public Engineer wrapper (preserves the eng.agent interface used by main.py)
# ---------------------------------------------------------------------------


class Engineer:
    """Engineer role wrapper. ``self.agent`` is an AutoGen ``ChatAgent``.

    Despite the wrapper, the Engineer is *implemented by* mini-swe-agent.
    Each Swarm handoff to "Engineer" boots a fresh ``DefaultAgent`` loop
    with bash + ``mcp_call`` actions and runs it to completion.
    """

    def __init__(self, pool: MCPClientPool) -> None:
        self._pool = pool
        self.agent = _MiniEngineerAgent(pool=pool)
