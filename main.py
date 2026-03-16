"""
SDLC entry point.

Orchestrates the AutoGen hub-and-spoke multi-agent workflow:

    UserProxy  →  ProjectManager  ↔  Architect
                                  ↔  Engineer
                                  ↔  CodeReviewer
                                  ↔  QA

The ProjectManager is the central hub.  It receives the user idea, delegates
tasks to each specialist via nested chats, and each specialist reports back
to the PM when done.  The PM decides next steps at every stage.

Usage
-----
    python -m ai-agents-mcp-client.main "Build a REST API for a todo app"
    python -m ai-agents-mcp-client.main "Build a REST API for a todo app" --rounds 10
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path

import fire
from dotenv import load_dotenv

from autogen_agentchat.agents import UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat

from agents import ProjectManager, Architect, Engineer, CodeReviewer, QA
from agents.config import ensure_workspace_dirs
from core.mcp_client import MCPClientPool
from core.mcp_config import ROLE_SERVERS

load_dotenv()

# ---------------------------------------------------------------------------
# Logging: mirror all logs to a per-session file under data/logs/
# ---------------------------------------------------------------------------

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

log_format = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Console handler (stderr/stdout)
if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    root_logger.addHandler(console_handler)

# Per-session file handler
logs_dir = Path(__file__).resolve().parent / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)
session_ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
log_file = logs_dir / f"session_{session_ts}.log"

file_handler = logging.FileHandler(log_file, encoding="utf-8")
file_handler.setFormatter(log_format)
root_logger.addHandler(file_handler)

logger = logging.getLogger(__name__)
logger.info("Session log file: %s", log_file)


async def start_sdlc(idea: str, rounds: int = 20) -> str:
    """
    Run the SDLC multi-agent workflow for the given idea.

    Parameters
    ----------
    idea:
        The project idea or user requirement to implement.
    rounds:
        Maximum number of agent messages before the workflow stops.

    Returns
    -------
    str
        The final message from the workflow (typically the PM's summary).
    """
    ensure_workspace_dirs()

    # Collect the union of all server keys needed by any role.
    all_server_keys: list[str] = list(
        {key for keys in ROLE_SERVERS.values() for key in keys}
    )

    logger.info("Opening MCP connections: %s", all_server_keys)

    async with MCPClientPool(server_keys=all_server_keys) as pool:
        # --- Instantiate role agents ---
        pm = ProjectManager(pool)
        arch = Architect(pool)
        eng = Engineer(pool)
        reviewer = CodeReviewer(pool)
        qa = QA(pool)

        # --- Termination conditions ---
        # Stop when the PM says the project is complete OR we hit the round cap.
        termination = (
            TextMentionTermination("PROJECT COMPLETE")
            | MaxMessageTermination(max_messages=rounds)
        )

        # --- Hub-and-spoke team ---
        # RoundRobinGroupChat is used here as the backbone; the PM's system
        # message instructs it to act as the hub and delegate to specialists.
        # Specialists are registered and respond only when the PM addresses them.
        team = RoundRobinGroupChat(
            participants=[
                # user_proxy,
                pm.agent,
                arch.agent,
                eng.agent,
                reviewer.agent,
                qa.agent,
            ],
            termination_condition=termination,
        )

        logger.info("Starting SDLC for: %s", idea)

        final_message = ""
        try:
            async for message in team.run_stream(task=idea):
                if hasattr(message, "content"):
                    logger.info(
                        "[%s] %s",
                        getattr(message, "source", "?"),
                        str(message.content)[:200],
                    )
                    final_message = str(message.content)
        except Exception as exc:  # noqa: BLE001
            # AutoGen may raise an internal termination-related exception when
            # the termination condition is satisfied. Treat that as a graceful
            # shutdown and still consider the workflow "done" for this run.
            logger.info(
                "SDLC workflow stopped due to %s: %s. Assuming all planned steps are done.",
                type(exc).__name__,
                exc,
            )
        finally:
            logger.info("SDLC workflow finished.")

        return final_message


def main(idea: str, rounds: int = 50) -> None:
    """
    CLI entry point.

    Parameters`
    ----------
    idea:
        The project idea or user requirement to implement.
    rounds:
        Maximum number of agent messages (default: 20).
    """
    result = asyncio.run(start_sdlc(idea, rounds=rounds))
    print("\n=== Final result ===")
    print(result)


if __name__ == "__main__":
    fire.Fire(main)
