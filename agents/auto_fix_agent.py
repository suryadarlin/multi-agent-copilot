"""
auto_fix_agent.py

AutoFixAgent: orchestrates the self-repair loop. Consumes structured
diagnostics from TestAgent, SecurityAgent, and DebugAgent, asks the LLM
(via GeminiClient + PromptManager) for a targeted patch, applies it to the
in-memory project source map, and triggers re-validation until the project
is clean or an iteration ceiling is reached.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from agents.debug_agent import DebugAgent, DebugAnalysis
from agents.security_agent import SecurityAgent
from agents.test_agent import TestAgent
from llm.gemini_client import GeminiClient
from llm.prompt_manager import PromptManager

logger = logging.getLogger("ai_engineering_copilot.agents.auto_fix_agent")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class AutoFixError(Exception):
    """Raised when the auto-fix loop cannot proceed."""


@dataclass
class FixAttempt:
    iteration: int
    target_file: str
    error_type: str
    fix_applied: bool
    diff_summary: str = ""


@dataclass
class FixSession:
    max_iterations: int
    attempts: list[FixAttempt] = field(default_factory=list)
    patched_files: set[str] = field(default_factory=set)
    resolved: bool = False
    remaining_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        last_iteration = self.attempts[-1].iteration if self.attempts else 0
        return {
            "iteration": last_iteration,
            "fix_applied": any(a.fix_applied for a in self.attempts),
            "patched_files": sorted(self.patched_files),
            "remaining_errors": self.remaining_errors,
            "resolved": self.resolved,
            "history": [
                {
                    "iteration": a.iteration,
                    "target_file": a.target_file,
                    "error_type": a.error_type,
                    "fix_applied": a.fix_applied,
                    "diff_summary": a.diff_summary,
                }
                for a in self.attempts
            ],
        }


class AutoFixAgent:
    """
    Drives the iterative repair loop:

        while errors_exist and iteration < max_iterations:
            diagnose -> generate_fix (LLM) -> patch_code -> rerun_validation
    """

    def __init__(
        self,
        project_files: dict[str, str],
        test_agent: TestAgent | None = None,
        security_agent: SecurityAgent | None = None,
        debug_agent: DebugAgent | None = None,
        gemini_client: GeminiClient | None = None,
        prompt_manager: PromptManager | None = None,
        max_iterations: int = 5,
    ) -> None:
        self.project_files = dict(project_files)
        self.test_agent = test_agent
        self.security_agent = security_agent or SecurityAgent()
        self.debug_agent = debug_agent or DebugAgent()
        self.gemini_client = gemini_client or GeminiClient()
        self.prompt_manager = prompt_manager or PromptManager()
        self.max_iterations = max_iterations
        self._iteration = 0
        logger.info(
            "AutoFixAgent initialized with %d file(s), max_iterations=%d",
            len(self.project_files),
            max_iterations,
        )

    async def run_repair_loop(self, traceback_text: str | None = None) -> FixSession:
        """
        Top-level loop combining diagnosis, fix generation, patching, and
        re-validation, bounded by max_iterations.
        """
        session = FixSession(max_iterations=self.max_iterations)
        current_traceback = traceback_text

        while self._iteration < self.max_iterations:
            self._iteration += 1
            logger.info("AutoFix iteration %d/%d starting", self._iteration, self.max_iterations)

            problem = await self.analyze_problem(current_traceback)
            if problem is None:
                logger.info("No outstanding problems detected; repair loop converged.")
                session.resolved = True
                break

            error_type, faulty_module, root_cause, suggested_fix = problem

            try:
                patch = await self.generate_fix(faulty_module, error_type, root_cause, suggested_fix)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Fix generation failed for %s", faulty_module)
                session.attempts.append(
                    FixAttempt(
                        iteration=self._iteration,
                        target_file=faulty_module,
                        error_type=error_type,
                        fix_applied=False,
                        diff_summary=f"generation_failed: {exc}",
                    )
                )
                session.remaining_errors.append(f"{faulty_module}: {root_cause}")
                continue

            applied = self.patch_code(faulty_module, patch)
            self.track_fix_iteration(session, faulty_module, error_type, applied, patch)

            validation = await self.rerun_validation()
            current_traceback = validation.get("traceback")

            if validation.get("clean", False):
                session.resolved = True
                session.remaining_errors = []
                logger.info("Repair loop resolved all known issues at iteration %d", self._iteration)
                break

            session.remaining_errors = validation.get("errors", [])

        if not session.resolved and session.remaining_errors:
            logger.warning(
                "AutoFix loop ended after %d iterations with %d unresolved issue(s)",
                self._iteration,
                len(session.remaining_errors),
            )

        return session

    async def analyze_problem(
        self, traceback_text: str | None
    ) -> tuple[str, str, str, str] | None:
        """
        Combines DebugAgent traceback analysis with a SecurityAgent scan to
        decide what the next fix target should be. Returns None when no
        problems remain. Runtime/test errors take priority over security
        findings since broken code cannot be meaningfully security-reviewed.
        """
        if traceback_text:
            analysis: DebugAnalysis = self.debug_agent.analyze_traceback(traceback_text)
            if analysis.error_type.value != "UnknownError" or analysis.confidence >= 0.4:
                return (
                    analysis.error_type.value,
                    analysis.faulty_module,
                    analysis.root_cause,
                    analysis.suggested_fix,
                )

        security_report = self.security_agent.scan_project(self.project_files)
        if security_report["critical"] > 0 or security_report["high"] > 0:
            top_issue = next(
                (i for i in security_report["issues"] if i["severity"] in ("critical", "high")),
                None,
            )
            if top_issue:
                return (
                    f"SecurityIssue:{top_issue['rule_id']}",
                    top_issue["file"],
                    top_issue["message"],
                    "Refactor flagged code to remove the vulnerability per security rule guidance.",
                )

        return None

    async def generate_fix(
        self, target_file: str, error_type: str, root_cause: str, suggested_fix: str
    ) -> str:
        """
        Builds a fix prompt and requests a corrected file body from the LLM.
        """
        original_source = self.project_files.get(target_file, "")
        prompt = self.prompt_manager.build_fix_prompt(
            file_name=target_file,
            original_source=original_source,
            error_type=error_type,
            root_cause=root_cause,
            suggested_fix=suggested_fix,
        )
        response = await self.gemini_client.generate_response(prompt, response_kind="code_fix")
        patched_source = self.gemini_client.validate_response(response, expect="code")
        return patched_source

    def patch_code(self, target_file: str, patched_source: str) -> bool:
        """Applies the patch to the in-memory project source map."""
        if not patched_source or not patched_source.strip():
            logger.warning("Empty patch received for %s; skipping apply.", target_file)
            return False
        self.project_files[target_file] = patched_source
        logger.info("Patched %s (%d bytes)", target_file, len(patched_source))
        return True

    async def rerun_validation(self) -> dict[str, Any]:
        """
        Re-runs tests (if a TestAgent is configured) and a fresh security
        scan against the patched project_files, returning a normalized
        status dict the repair loop can branch on.
        """
        errors: list[str] = []
        traceback_text: str | None = None

        if self.test_agent is not None:
            try:
                test_report = self.test_agent.run_tests()
                if test_report.failed > 0:
                    errors.extend(
                        f"{f.module}::{f.test_name}: {f.message}" for f in test_report.failures
                    )
                    if test_report.failures:
                        traceback_text = test_report.failures[0].traceback or test_report.raw_stderr
            except Exception as exc:  # noqa: BLE001
                logger.exception("Validation test run failed")
                errors.append(f"test_execution_error: {exc}")

        security_report = self.security_agent.scan_project(self.project_files)
        if security_report["critical"] > 0 or security_report["high"] > 0:
            errors.append(
                f"security: {security_report['critical']} critical, {security_report['high']} high severity issue(s)"
            )

        return {"clean": len(errors) == 0, "errors": errors, "traceback": traceback_text}

    def track_fix_iteration(
        self,
        session: FixSession,
        target_file: str,
        error_type: str,
        fix_applied: bool,
        patch: str,
    ) -> None:
        diff_summary = f"{len(patch.splitlines())} line(s) replaced in {target_file}" if fix_applied else "no change applied"
        session.attempts.append(
            FixAttempt(
                iteration=self._iteration,
                target_file=target_file,
                error_type=error_type,
                fix_applied=fix_applied,
                diff_summary=diff_summary,
            )
        )
        if fix_applied:
            session.patched_files.add(target_file)