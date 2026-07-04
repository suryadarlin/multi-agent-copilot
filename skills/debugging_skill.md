# Debugging Skill

## Purpose
Diagnoses runtime failures and test failures surfaced by code_executor.py
and test_runner.py, and proposes targeted fixes. Loaded exclusively by
DebugAgent and AutoFixAgent.

## Capabilities
- Parse stack traces and pytest failure output into structured error signatures
- Query MemoryStore for previously successful fixes to matching error signatures
- Propose minimal, targeted code diffs (not full rewrites)
- Distinguish between logic errors, dependency errors, and environment errors
- Detect repeated-failure loops and escalate instead of looping indefinitely

## Required Tools
- test_runner.py (re-run tests after applying a candidate fix)
- code_executor.py (verify fix resolves the runtime exception)
- file_reader.py (inspect failing module and its dependents)
- memory_store.py (fix history lookup and persistence)

## Execution Rules
1. On any failure, first compute an error_signature and query
   `memory_store.retrieve_previous_solution()` before generating a new fix.
2. If a prior solution exists and is applicable, reuse it and re-verify
   rather than regenerating from scratch.
3. If no prior solution exists, generate the smallest viable diff to
   resolve the failure — never a full module rewrite on first attempt.
4. Maximum of 3 auto-fix attempts per module per pipeline run before
   escalating to a human-readable failure report.
5. Every successful fix is persisted via `memory_store.store_fix_history()`.

## Constraints
- Must not modify modules unrelated to the failing test/trace.
- Must not disable or skip failing tests as a "fix".

## Expected Outputs
- Applied diff (if auto-fixed) or an escalation report (if unresolved).
- Updated error_signature record in MemoryStore.

## Example Prompt
"Test suite reports ImportError: cannot import name 'get_db' from
'database' in test_auth.py — diagnose and fix."