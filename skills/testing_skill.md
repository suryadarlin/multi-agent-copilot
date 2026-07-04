# Testing Skill

## Purpose
Drives automated test generation and execution for completed modules,
loaded exclusively by TestAgent after Security Skill has cleared a module.

## Capabilities
- Generate pytest test suites covering core logic, edge cases, and failure modes
- Execute test suites in the sandboxed runner and collect structured results
- Compute coverage percentage per module and project-wide
- Classify failures as flaky, deterministic, or environment-caused

## Required Tools
- test_runner.py (pytest execution and coverage collection)
- code_executor.py (isolated execution sandbox)
- file_reader.py (inspect module under test for signature extraction)

## Execution Rules
1. Loads only the target module's source and its public interface — not
   the full project context.
2. Generated tests must be deterministic; tests relying on wall-clock time,
   network access, or random seeds without fixing the seed are rejected.
3. On test failure, hands off structured failure data to Debugging Skill;
   does not attempt fixes itself.
4. Coverage below 70% on a CRITICAL-risk module (per Planner risk flags)
   triggers an additional test-generation pass, up to 2 extra passes.

## Constraints
- Must not modify production source files.
- Must not mark a module complete with any failing test still present.

## Expected Outputs
- Generated test file(s).
- Structured result: `{tests_run, passed, failed, coverage, failure_summary}`.

## Example Prompt
"Generate and run a pytest suite for auth.py covering JWT issuance,
expiry, and invalid-token rejection paths."