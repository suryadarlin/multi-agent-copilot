# Coding Skill

## Purpose
Generates production-quality source code for a single module at a time,
based on the task breakdown produced by the Planner Skill. Loaded
exclusively by CodeAgent.

## Capabilities
- Generate fully runnable Python/JS/TS module code from a task spec
- Apply project-wide conventions (typing, logging, error handling)
- Generate accompanying unit test stubs for each module
- Request clarification from Orchestrator on ambiguous task specs

## Required Tools
- code_executor.py (validate generated code actually runs)
- file_reader.py (read sibling modules for consistent interfaces)
- shell_runner.py (install required dependencies for a dry run)

## Execution Rules
1. Loads only the current task's module spec and its direct dependency
   modules' interfaces — not the entire project plan.
2. Every generated module must be passed to code_executor.py for a syntax
   and smoke-test pass before being marked complete.
3. No placeholder/TODO code is ever emitted; incomplete generations are
   retried, not shipped.
4. On failure, hands off to Debugging Skill rather than self-correcting
   indefinitely (max 1 internal retry).

## Constraints
- Must not modify files outside its assigned module unless explicitly
  instructed by Orchestrator.
- Must not introduce dependencies not present in the Planner's dependency list
  without flagging the addition.

## Expected Outputs
- Complete source file content for the assigned module.
- A short rationale of key design decisions.
- A list of any new dependencies introduced.

## Example Prompt
"Generate auth.py implementing JWT issuance and verification per the
project plan, consistent with the existing database.py interface."