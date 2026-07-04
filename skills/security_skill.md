# Security Skill

## Purpose
Performs static and behavioral security review of generated code prior to
execution or merge, loaded exclusively by the Critic/Security Agent stage.

## Capabilities
- Detect SQL injection patterns in raw query construction
- Detect hardcoded secrets, API keys, and credentials
- Detect JWT misconfiguration (weak algorithms, missing expiry, static secrets)
- Detect unsafe shell/subprocess execution (command injection vectors)
- Detect insecure deserialization (pickle, eval, exec on untrusted input)
- Flag missing input validation on external-facing endpoints

## Required Tools
- security_scanner.py (pattern-based static analysis)
- file_reader.py (read target source files)
- shell_runner.py (run sandboxed dependency vulnerability scans, e.g. pip-audit)

## Execution Rules
1. Loads only after a module has passed code_executor.py smoke testing.
2. Runs as a read-only review pass — never modifies source directly;
   emits findings for Auto Fix Agent to act on.
3. Each finding includes severity (critical/high/medium/low), location,
   and a remediation suggestion.
4. Any CRITICAL finding blocks promotion to Test Agent until resolved.

## Constraints
- Must not execute untrusted code outside the sandboxed executor.
- Must not suppress findings based on agent self-reported confidence alone.

## Expected Outputs
- A findings list: `{severity, rule_id, file, line, description, remediation}`
- An overall pass/fail gate decision for the reviewed module.

## Example Prompt
"Review auth.py and database.py for SQL injection, hardcoded secrets, and
unsafe JWT configuration before promoting to the test stage."