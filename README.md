# mas-decentralized

An AutoGen-based multi-agent system that runs a full Software Development Life Cycle (SDLC) workflow from a single natural-language idea. A team of AI agents — each with a distinct role and access to scoped MCP tool servers — collaborates to plan, design, implement, review, and QA software projects autonomously.

---

## How it works

The system uses a **mesh** architecture built on [AutoGen](https://microsoft.github.io/autogen/) `Swarm`. Agents hand off directly to their natural downstream peer; the **Project Manager** acts as a strategic orchestrator rather than a routing hub, re-entering the flow only on escalation or final completion.

```
User idea
    │
    ▼
ProjectManager (Alice)
    │
    ▼
Architect (Bob) ──────────────────────────────────────────┐
    │                                                      │ escalate
    ▼                                                      │
Engineer (Charlie) ◄─────────────────────────────── CodeReviewer (Dave)
    │   ▲                                                  │
    │   └──────────────────────────── QA (Eve) (bugs)      │ approve
    │                                     │                │
    │                                     └────────────────┘
    │                                          │
    └──────── (commit) ────────────────────────┘
                                               │
                                    ProjectManager ← final report
```

Specialists loop back without PM involvement:
- **CodeReviewer → Engineer** when issues are found
- **QA → Engineer** when bugs are found
- **Any agent → PM** only when genuinely blocked

Each agent is backed by an LLM and has access to a curated set of [MCP](https://modelcontextprotocol.io/) tool servers that scope what it can read and write.

### Agent roles

| Agent | Persona | Responsibilities | MCP access |
|---|---|---|---|
| **ProjectManager** | Alice | Breaks down ideas into tickets, kicks off the workflow, writes the final summary | `fs_board`, `fs_docs` |
| **Architect** | Bob | Produces system design and architecture docs, hands off directly to Engineer | `fs_board`, `fs_docs`, `fs_code` |
| **Engineer** | Charlie | **Implemented by [`mini-swe-agent`](https://mini-swe-agent.com/)** — runs a fresh bash-driven coding loop inside the workspace, commits, then hands off to CodeReviewer | `fs_board`, `fs_docs`, `fs_code`, `git` (via `mcp_call`) |
| **CodeReviewer** | Dave | Reviews code; routes to QA (approved) or back to Engineer (issues) | `fs_board`, `fs_docs`, `fs_code`, `git` |
| **QA** | Eve | Tests the implementation; routes back to Engineer (bugs) or to PM (all clear) | `fs_board`, `fs_code` |

> The Engineer is not an AutoGen `AssistantAgent` with direct tool calling. A
> Swarm handoff to `Engineer` starts a [`mini-swe-agent`](https://mini-swe-agent.com/)
> `DefaultAgent` inside `data/workspace/`. The mini-agent can run normal bash
> commands and can also call the existing scoped MCP servers with:
>
> ```bash
> mcp_call <server> <tool> '<JSON_ARGS>'
> ```
>
> By default, successful Engineer turns emit a `HandoffMessage` to
> `CodeReviewer`. If the mini-agent is blocked, it can start its final
> submission with `ESCALATE_TO_PROJECT_MANAGER` to escalate instead. Full
> trajectories are written to `logs/mini_traj_<run-id>_turn<NN>.json`.

### Handoff graph

| From | To | Condition |
|---|---|---|
| ProjectManager | Architect | Start new work |
| ProjectManager | Engineer | Skip design (trivial change / hotfix) |
| ProjectManager | CodeReviewer / QA | Out-of-band intervention |
| Architect | Engineer | Design complete *(primary path)* |
| Architect | ProjectManager | Blocked — requirements unclear |
| Engineer | CodeReviewer | Implementation complete *(primary path)* |
| Engineer | ProjectManager | Blocked — scope/design missing |
| CodeReviewer | QA | Code approved *(primary path)* |
| CodeReviewer | Engineer | Issues found |
| CodeReviewer | ProjectManager | Scope/business decision needed |
| QA | ProjectManager | All tests pass — final report *(primary path)* |
| QA | Engineer | Bugs found |

### MCP servers

| Key | Server | Scoped path |
|---|---|---|
| `fs_board` | `@modelcontextprotocol/server-filesystem` | `data/project_board/` |
| `fs_docs` | `@modelcontextprotocol/server-filesystem` | `data/knowledge_base/` |
| `fs_code` | `@modelcontextprotocol/server-filesystem` | `data/workspace/` |
| `git` | `mcp-server-git` | `data/workspace/` |

---

## Project structure

```
mas-decentralized/
├── main.py                  # CLI entry point & SDLC orchestration
├── agents/
│   ├── __init__.py
│   ├── config.py            # Workspace directory setup
│   └── roles/
│       ├── project_manager.py
│       ├── architect.py
│       ├── engineer.py      # mini-swe-agent backed (BaseChatAgent + MCPLocalEnvironment)
│       ├── code_reviewer.py
│       └── qa.py
├── core/
│   ├── mcp_client.py        # MCPClient / MCPClientPool (async lifecycle management)
│   ├── mcp_config.py        # MCP server registry & role-to-server mapping
│   ├── mcp_tools.py         # Tool bindings exposed to AutoGen agents
│   └── autogen_config.py    # LLM model client configuration
├── data/
│   ├── workspace/           # Agent-generated code lives here (git-tracked)
│   ├── project_board/       # Kanban tickets written by the PM
│   └── knowledge_base/      # Architecture and design docs
├── logs/                    # Per-session log files
├── .env                     # API keys and model config (not committed)
└── pyproject.toml
```

---

## Prerequisites

- **Python 3.13+**
- **Node.js 18+** (for `npx`-based MCP servers)
- **uv** (`pip install uv`)
- **mcp-server-git** (`pip install mcp-server-git`)
- An **OpenAI API key**

---

## Setup

**1. Clone and install dependencies**

```bash
git clone <repo-url>
cd mas-decentralized
uv sync
```

**2. Configure environment variables**

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

```env
OPENAI_API_KEY=sk-...

# Model to use (default: gpt-4o)
AUTOGEN_MODEL=gpt-4o

# LLM temperature (default: 0.1)
AUTOGEN_TEMPERATURE=0.1

# Optional: model used by the Engineer's mini-swe-agent worker.
# Defaults to AUTOGEN_MODEL. Must be a LiteLLM-compatible name
# (e.g. "gpt-4o", "openai/gpt-4o", "anthropic/claude-sonnet-4-5-20250929").
# MINI_AGENT_MODEL=gpt-4o

# Optional: cost ceiling per mini-swe-agent run, in USD (default: 3.0).
# MINI_AGENT_COST_LIMIT=3.0

# Optional: hard step ceiling per mini-swe-agent run (default: 0 = unlimited).
# MINI_AGENT_STEP_LIMIT=0

# Optional: timeout (seconds) for a single mcp_call from inside mini-swe-agent
# (default: 120).
# MINI_AGENT_MCP_TIMEOUT=120
```

**3. Initialise the workspace git repo** (first run only, **required for git MCP tools**)

```bash
git init data/workspace
```

---

## Usage

```bash
uv run python main.py "<your idea>" [--rounds N]
```

| Argument | Description | Default |
|---|---|---|
| `idea` | Natural-language description of what to build | *(required)* |
| `--rounds` | Maximum number of agent messages before the workflow stops | `100` |

### Examples

```bash
# Build a CLI todo app
uv run python main.py "Build a CLI Todo App with JSON storage"

# Build a REST API with a limited number of agent turns
uv run python main.py "Build a REST API for a blog" --rounds 30
```

The workflow stops automatically when the ProjectManager outputs `PROJECT COMPLETE` or the round limit is reached.

### Output

- **Terminal** — live log of every agent message
- **`logs/session_<timestamp>.log`** — full session log persisted to disk
- **`logs/mini_traj_<run-id>_turn<NN>.json`** — full mini-swe-agent trajectory for each Engineer turn
- **`data/workspace/`** — all generated source code, committed to git
- **`data/project_board/`** — Kanban tickets tracking the work
- **`data/knowledge_base/`** — architecture and design documents

---

## Development

### Running with a different model

Set `AUTOGEN_MODEL` in `.env` to any OpenAI-compatible model name (e.g. `gpt-4o`, `gpt-4-turbo`).

### Adding a new agent role

1. Create `agents/roles/<role>.py` following the pattern of existing roles.
2. Declare `Handoff` objects for each peer the role can hand off to (primary path + `ProjectManager` as escalation fallback).
3. Update the `Handoff` lists of any existing agents that should be able to route to the new role.
4. Add the new class to `agents/__init__.py`.
5. Register the role's allowed MCP servers in `core/mcp_config.py` under `ROLE_SERVERS`.
6. Instantiate the agent in `main.py` and add it to the `Swarm` participants list.

### Adding a new MCP server

Add an entry to `MCP_SERVERS` in `core/mcp_config.py`:

```python
"my_server": {
    "command": "npx",
    "args": ["-y", "@my-org/mcp-server", "/some/path"],
},
```

Then list it under the relevant roles in `ROLE_SERVERS`.
