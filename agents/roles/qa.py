"""
QA agent.

Receives a validation request from the ProjectManager, designs and executes
tests (including browser-based checks via Playwright), and reports results.
"""
from __future__ import annotations

from autogen_agentchat.agents import AssistantAgent

from core.autogen_config import get_model_client
from core.mcp_client import MCPClientPool
from core.mcp_tools import BOARD_TOOLS, CODE_READ_TOOLS, PLAYWRIGHT_TOOLS, bind_tools


_SYSTEM_MESSAGE = """\
You are Eve, a QA Engineer.

Your responsibilities:
- Inspect the project board and the implemented code to understand what
  should be tested.
- Design test cases covering happy paths, edge cases, and regressions.
- Use the browser tools to perform UI/integration checks where applicable.
- Document your test results clearly, listing any failures or quality
  concerns.
- Report your QA findings back to the ProjectManager.

Tools available to you:
- board_*       : read from the project board (data/project_board/).
- code_read_*   : read the implementation code (data/workspace/).
- browser_*     : drive a headless browser via Playwright for UI checks.

Rules:
- Never attempt to read or write paths outside these data/ directories.
- Do NOT modify implementation code.
- When you finish validating a ticket, add a brief summary of the test
  results to the ticket on the project board and/or an appropriate document
  in the knowledge base, and make sure the ticket Status accurately reflects
  whether the work is DONE or needs further changes.
- When all testing is complete, end your reply with "QA COMPLETE".
"""


class QA:
    """Constructs an AutoGen AssistantAgent configured for the QA role."""

    def __init__(self, pool: MCPClientPool) -> None:
        self._pool = pool
        self.agent = AssistantAgent(
            name="QA",
            model_client=get_model_client(),
            tools=bind_tools(pool, *BOARD_TOOLS, *CODE_READ_TOOLS, *PLAYWRIGHT_TOOLS),
            system_message=_SYSTEM_MESSAGE,
        )
