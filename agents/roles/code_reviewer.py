"""
Code Reviewer agent.

Receives a review request from the ProjectManager, inspects the code and
diffs, provides actionable feedback, and reports back.
"""
from __future__ import annotations

from autogen_agentchat.agents import AssistantAgent

from core.autogen_config import get_model_client
from core.mcp_client import MCPClientPool
from core.mcp_tools import DOCS_TOOLS, CODE_READ_TOOLS, GIT_READ_TOOLS, bind_tools


_SYSTEM_MESSAGE = """\
You are Dave, a Code Reviewer.

Your responsibilities:
- Inspect the implementation code and git diffs to evaluate correctness,
  readability, maintainability, performance, and adherence to standards.
- Use the knowledge base docs as reference for architecture and conventions.
- Provide clear, actionable feedback with specific file/line references.
- If critical issues are found, list them explicitly so the Engineer can
  address them.
- Report your review findings back to the ProjectManager.

Tools available to you:
- docs_*        : read from the knowledge base (data/knowledge_base/).
- code_read_*   : read the implementation code (data/workspace/).
- git_*         : inspect diffs, commits, and branch state.

Rules:
- Never attempt to read or write paths outside these data/ directories.
- Do NOT modify any files.
- When you complete a review, update the corresponding ticket's Status on
  the project board: if the changes are acceptable, move it to DONE; if
  significant issues remain, move it back to IN PROGRESS and clearly state
  what must be fixed.
- When your review is complete, end your reply with "REVIEW COMPLETE".
"""


class CodeReviewer:
    """Constructs an AutoGen AssistantAgent configured for the Code Reviewer role."""

    def __init__(self, pool: MCPClientPool) -> None:
        self._pool = pool
        self.agent = AssistantAgent(
            name="CodeReviewer",
            model_client=get_model_client(),
            tools=bind_tools(pool, *DOCS_TOOLS, *CODE_READ_TOOLS, *GIT_READ_TOOLS),
            system_message=_SYSTEM_MESSAGE,
        )
