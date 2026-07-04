# Planner Skill

## Purpose
Decomposes a raw, natural-language software request into a structured
engineering specification: target frameworks, required modules, dependency
graph, and risk surface. This skill is loaded exclusively by the
PlannerAgent at the start of the pipeline, before any code is written.

## Capabilities
- Parse free-form software requirements into structured JSON
- Infer backend/frontend frameworks and datastore from intent
- Generate an ordered task breakdown with module-level dependencies
- Resolve implicit dependency packages from named frameworks
- Surface early architectural and security risks before implementation

## Required Tools
- file_reader.py (to inspect existing project structure, if present)
- gemini_client.py (LLM reasoning backend)

## Execution Rules
1. This skill loads BEFORE any other skill in the pipeline.
2. It must never write or modify source files — planning only.
3. Output must conform strictly to the ProjectPlan schema (see
   `planner_agent.py`); free-text responses are rejected and retried once.
4. If requirement analysis fails twice, the skill escalates to the
   Orchestrator with a `planning_failed` status rather than guessing.
5. Context passed to this skill is limited to the raw user request plus
   any CRITICAL-priority context entries — no historical agent chatter.

## Constraints
- Must not invent dependencies unrelated to the stated framework/database.
- Must not exceed a 30-second LLM call budget.
- Must not fabricate file paths that conflict with existing project structure.

## Expected Outputs
- A `ProjectPlan` object: frameworks, database, required_modules,
  dependencies, risks, task_breakdown.

## Example Prompt
"Build a FastAPI authentication system with JWT and PostgreSQL, including
rate limiting on the login endpoint."