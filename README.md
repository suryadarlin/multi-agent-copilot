
# 🤖 AI Engineering Copilot

An AI-powered multi-agent software engineering assistant that automates the software development lifecycle using specialized AI agents.

AI Engineering Copilot — A multi-agent AI system that automates software planning, code generation, code review, testing, security analysis, debugging, and workflow orchestration using Python, FastAPI, and Google Gemini.

Capstone submission — 2026 Google + Kaggle AI Agents Intensive Course.


## 1. Project Overview

The AI Software Engineering Copilot takes a single natural-language requirement (e.g. *"Build a FastAPI JWT auth system"*) and autonomously carries it from specification to a tested, security-reviewed, working codebase — without a human in the loop for routine fixes.

It demonstrates four core agentic-systems concepts from the course:

| Concept | Where it shows up |
|---|---|
| **Multi-agent orchestration** | Planner → Code → Critic → Test → Security → Debug → Auto-Fix, coordinated by `agents/orchestrator.py` |
| **MCP (Model Context Protocol) tool use** | `mcp_tools/` exposes sandboxed code execution, shell commands, file I/O, and test running as callable tools |
| **Context management & progressive disclosure** | `Context Manager` and `Memory Store` (Phase 4) persist only the relevant slice of project state to each agent, instead of dumping the full history into every prompt |
| **Self-repair loop** | Failed tests or security findings route to the Debug Agent → Auto Fix Agent, which patches and re-verifies automatically |



## 2. Architecture
User Requirement
        │
        ▼
 Orchestrator Agent
        │
 ┌──────┼─────────────┐
 ▼      ▼             ▼
Planner Code      Critic
Agent   Agent      Agent
 │        │           │
 └──────┬─┴───────────┘
        ▼
 Test Agent
        ▼
 Security Agent
        ▼
 Debug Agent
        ▼
 Final Output

```
                    ┌──────────────┐
   User Prompt ───► │ Orchestrator │
                    └──────┬───────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │   Planner Agent   │  decomposes requirement into subtasks
                  └────────┬──────────┘
                           ▼
                  ┌──────────────────┐
                  │    Code Agent     │  generates implementation
                  └────────┬──────────┘
                           ▼
                  ┌──────────────────┐
                  │   Critic Agent    │  reviews quality / style / correctness
                  └────────┬──────────┘
                           ▼
                  ┌──────────────────┐        ┌─────────────────────┐
                  │  MCP Tool Layer   │ ─────► │ Sandbox Execution    │
                  │ (code/shell/file/  │        │ (code_executor,      │
                  │  test runners)    │        │  shell_runner)       │
                  └────────┬──────────┘        └─────────────────────┘
                           ▼
                  ┌──────────────────┐
                  │    Test Agent     │  runs pytest via test_runner.py
                  └────────┬──────────┘
                           ▼
                  ┌──────────────────┐
                  │  Security Agent   │  scans for vulnerable patterns
                  └────────┬──────────┘
                           ▼
                  ┌──────────────────┐
                  │   Debug Agent     │  diagnoses failures
                  └────────┬──────────┘
                           ▼
                  ┌──────────────────┐
                  │  Auto Fix Agent   │──┐ retries failed stages
                  └────────┬──────────┘  │
                           ▼              │
                       ✅ Success ◄───────┘
```

All agent communication is mediated by a shared `Context Manager`, backed by a `Memory Store` (SQLite), so each agent receives only what it needs (progressive disclosure) rather than the entire conversation history.




## 3. Agent Roles

| Agent | Responsibility |
|---|---|
| **Orchestrator** | Drives the pipeline, sequences agents, handles retries |
| **Planner Agent** | Breaks the requirement into ordered subtasks |
| **Code Agent** | Generates source code for each subtask |
| **Critic Agent** | Reviews generated code for correctness, style, and design issues before execution |
| **Test Agent** | Runs the generated test suite via the MCP `test_runner` tool and reports pass/fail/coverage |
| **Security Agent** | Scans generated code for vulnerable patterns (secrets, injection risks, unsafe subprocess calls) |
| **Debug Agent** | Diagnoses root causes of test or security failures |
| **Auto Fix Agent** | Applies a targeted patch and triggers re-verification |




## 4. MCP Tool Layer

| Tool | Purpose |
|---|---|
| `code_executor.py` | Executes AI-generated Python in an isolated temp file with a timeout, capturing stdout/stderr/exit code |
| `shell_runner.py` | Runs whitelisted shell commands (`pip install`, `pytest`, `python`, etc.), blocking dangerous ones (`rm -rf`, `sudo`, `shutdown`) |
| `file_reader.py` | Reads/lists project files with path-traversal protection |
| `test_runner.py` | Executes `pytest`, parses results, and reports structured pass/fail/coverage data |

---

## 5. Installation (local, no Docker)

```bash
git clone <your-repo-url>
cd multi-agent-copilot

python3.11 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env            # add your GEMINI_API_KEY

streamlit run ui/dashboard.py
```

The dashboard opens at `http://localhost:8501`.




## 6. Docker

```bash
cd deployment
docker compose up --build
```

This builds the multi-stage production image, mounts persistent volumes for SQLite memory, generated output, and logs, and exposes the dashboard on port `8501` (configurable via `APP_PORT` in `.env`).

To build/run the image directly without Compose:

```bash
docker build -f deployment/Dockerfile -t ai-engineering-copilot .
docker run -p 8501:8501 --env-file .env ai-engineering-copilot
```

---

## 7. Deployment (CI/CD → Google Cloud Run)

`.github/workflows/deploy.yml` runs on every push to `main`:

1. **test** — `pytest` with coverage
2. **lint** — `ruff` + `black --check`
3. **security-scan** — `bandit` (static analysis) + `pip-audit` (dependency CVEs)
4. **build-and-push** — builds the Docker image and pushes it to Artifact Registry
5. **deploy** — deploys the new image to Cloud Run

Required GitHub secrets:

| Secret | Purpose |
|---|---|
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | Keyless auth to GCP |
| `GCP_SERVICE_ACCOUNT` | Deploy service account |
| `GCP_PROJECT_ID` | Target GCP project |
| `GEMINI_API_KEY` | Stored in Secret Manager, injected at deploy time |

---

## 8. Monitoring

| Module | What it tracks |
|---|---|
| `monitoring/logger.py` | Structured JSON logs for agent events, workflow stages, and errors (stdout + file, Cloud Logging-compatible) |
| `monitoring/metrics.py` | Execution time, retry count, test success rate, bug frequency, security issue count, auto-fix success rate — persisted to `logs/metrics_snapshot.json` and surfaced on the dashboard |

---

## 9. Demo Instructions

1. Run the dashboard (`streamlit run ui/dashboard.py` or via Docker).
2. Enter a requirement, e.g. `"Build a FastAPI JWT auth system with refresh tokens"`.
3. Click **Run Autonomous Workflow** and watch each agent execute live, including the auto-fix loop if tests or security checks initially fail.
4. Review execution metrics (time, coverage, security issues, retries).
5. Download the generated project as a ZIP from the **Project Artifacts** panel.

---

## 10. Capstone Submission Notes

This project targets the course's emphasis on **agentic patterns over single-shot generation**: planning, tool use via MCP, context management with progressive disclosure, and a closed-loop self-repair cycle rather than a one-pass "generate and hope" pipeline. The dashboard is built to make every one of those mechanisms visible to a judge in real time, not just claimed in writing.

