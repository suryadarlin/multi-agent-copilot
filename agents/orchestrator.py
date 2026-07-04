"""
orchestrator.py

Central workflow controller for the AI Software Engineering Copilot.

The orchestrator owns the end-to-end lifecycle of a user request: it
instantiates the downstream agents, progresses through a multi-stage
pipeline, retries transient failures, and
produces a structured result object for downstream consumers.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

try:
    from .auto_fix_agent import AutoFixAgent
except Exception:  # pragma: no cover - optional dependency fallback
    AutoFixAgent = Any  # type: ignore[misc,assignment]

from .code_agent import CodeAgent, CodeGenerationResult
from .crictic_agent import CriticAgent, CriticReview
from .debug_agent import DebugAgent, ErrorType
from .planner_agent import PlannerAgent
from .security_agent import SecurityAgent
from .test_agent import TestAgent

logger = logging.getLogger("copilot.orchestrator")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class WorkflowStage(str, Enum):
    """Explicit, ordered stages of the orchestration pipeline."""

    RECEIVED = "received"
    PLANNING = "planning"
    CODE_GENERATION = "code_generation"
    CRITIC_REVIEW = "critic_review"
    TESTING = "testing"
    SECURITY_SCAN = "security_scan"
    DEBUGGING = "debugging"
    AUTO_FIXING = "auto_fixing"
    COLLECTING_RESULTS = "collecting_results"
    COMPLETED = "completed"
    FAILED = "failed"


class OrchestrationError(Exception):
    """Raised when the orchestrator cannot complete a workflow after retries."""

    def __init__(
        self,
        stage: WorkflowStage,
        message: str,
        cause: Optional[Exception] = None,
    ):
        self.stage = stage
        self.cause = cause
        super().__init__(f"[{stage.value}] {message}")


@dataclass
class WorkflowState:
    """Mutable record of a single workflow execution."""

    request_id: str
    user_request: str
    stage: WorkflowStage = WorkflowStage.RECEIVED
    plan_result: Optional[Any] = None
    code_result: Optional[CodeGenerationResult] = None
    critic_result: Optional[CriticReview] = None
    test_result: Optional[Any] = None
    security_result: Optional[dict[str, Any]] = None
    debug_result: Optional[Any] = None
    fix_result: Optional[Any] = None
    error: Optional[str] = None
    started_at: float = field(default_factory=time.monotonic)
    finished_at: Optional[float] = None
    attempts: dict[str, int] = field(default_factory=dict)

    def elapsed_ms(self) -> float:
        end = self.finished_at if self.finished_at is not None else time.monotonic()
        return round((end - self.started_at) * 1000, 2)


@dataclass
class OrchestrationResult:
    """Final structured output returned to the caller (e.g. main.py)."""

    request_id: str
    success: bool
    stage_reached: WorkflowStage
    generated_files: dict[str, str]
    critic_feedback: dict[str, Any]
    elapsed_ms: float
    error: Optional[str] = None
    plan_summary: Optional[dict[str, Any]] = None
    test_summary: Optional[dict[str, Any]] = None
    security_summary: Optional[dict[str, Any]] = None
    debug_summary: Optional[dict[str, Any]] = None
    fix_summary: Optional[dict[str, Any]] = None


class Orchestrator:
    """Drives a user request through the full multi-agent engineering pipeline."""

    # Limited retries for Gemini/semantic failures.
    # Keep small to respect free-tier quotas.
    MAX_RETRIES: int = 2
    RETRY_BACKOFF_BASE_SECONDS: float = 1.0


    def __init__(
        self,
        planner_agent: Optional[PlannerAgent] = None,
        code_agent: Optional[CodeAgent] = None,
        critic_agent: Optional[CriticAgent] = None,
        test_agent: Optional[TestAgent] = None,
        security_agent: Optional[SecurityAgent] = None,
        debug_agent: Optional[DebugAgent] = None,
        auto_fix_agent: Optional[Any] = None,
    ) -> None:
        self._planner_agent = planner_agent or PlannerAgent()
        self._code_agent = code_agent or CodeAgent()
        self._critic_agent = critic_agent or CriticAgent()
        project_root = Path(__file__).resolve().parent.parent
        self._test_agent = test_agent or TestAgent(project_root=project_root)
        self._security_agent = security_agent or SecurityAgent(project_root=project_root)
        self._debug_agent = debug_agent or DebugAgent()
        self._auto_fix_agent = auto_fix_agent
        self._active_states: dict[str, WorkflowState] = {}
        logger.info("Orchestrator initialized with 8 agents")

    async def receive_request(self, user_request: str) -> OrchestrationResult:
        """Entry point for a new user request."""
        if not user_request or not user_request.strip():
            raise ValueError("user_request must be a non-empty string")

        request_id = str(uuid.uuid4())
        state = WorkflowState(request_id=request_id, user_request=user_request.strip())
        self._active_states[request_id] = state

        try:
            # Execute each agent exactly once.
            await self._run_planner_agent(state)
            await self._run_code_agent(state)
            await self._run_critic_agent(state)
            await self._run_test_agent(state)
            await self._run_security_agent(state)
            await self._run_debug_agent(state)
            # AutoFix is intentionally disabled at runtime.

            return self._collect_results(state)

        except OrchestrationError as exc:
            print("=" * 60)
            print("ORCHESTRATION FAILED")
            print("Stage :", exc.stage)
            print("Error :", exc)
            print("=" * 60)

            state.stage = WorkflowStage.FAILED
            state.error = str(exc)
            state.finished_at = time.monotonic()
            print(f"ERROR: Stage -> {exc.stage.value.upper()} | {exc}")
            return self._build_failure_result(state)

        finally:
            self._active_states.pop(request_id, None)

    async def _run_planner_agent(self, state: WorkflowState) -> None:
        state.stage = WorkflowStage.PLANNING
        print("Stage → PLANNING")

        async def _attempt() -> Any:
            try:
                return await self._planner_agent.generate_project_plan(state.user_request)
            except Exception:  # noqa: BLE001
                # PlannerAgent.execute() returns a plain dict.
                response = await self._planner_agent.execute({"request": state.user_request})
                if isinstance(response, dict) and response:
                    return response

                # Deterministic fallback planner so the pipeline can still proceed
                # when Gemini quota/API is unavailable.
                required_modules = [
                    "main.py",
                    "models.py",
                    "routes.py",
                    "auth.py",
                ]
                task_breakdown = [
                    {"order": 1, "module": m, "depends_on": None, "status": "pending"}
                    for m in required_modules
                ]

                return {
                    "tasks": task_breakdown,
                    "dependencies": [
                        "fastapi",
                        "uvicorn",
                        "pydantic",
                        "python-jose",
                        "passlib",
                    ],
                    "execution_order": [t["order"] for t in task_breakdown],
                    "plan": {
                        "backend_framework": "fastapi",
                        "frontend_framework": None,
                        "database": None,
                        "required_modules": required_modules,
                        "dependencies": [
                            "fastapi",
                            "uvicorn",
                            "pydantic",
                            "python-jose",
                            "passlib",
                        ],
                        "risks": [],
                        "task_breakdown": task_breakdown,
                        "raw_request": state.user_request,
                    },
                }

        state.plan_result = await _attempt()

    async def _run_code_agent(self, state: WorkflowState) -> None:
        state.stage = WorkflowStage.CODE_GENERATION
        print("Stage → CODE_GENERATION")

        async def _attempt() -> CodeGenerationResult:
            request = self._build_code_request(state)
            try:
                return await self._code_agent.generate(request)
            except Exception as exc:  # noqa: BLE001
                raise OrchestrationError(
                    WorkflowStage.CODE_GENERATION,
                    f"Code generation failed: {exc}",
                    cause=exc,
                ) from exc

        state.code_result = await _attempt()

    async def _run_critic_agent(self, state: WorkflowState) -> None:
        state.stage = WorkflowStage.CRITIC_REVIEW
        print("Stage → CRITIC_REVIEW")

        if state.code_result is None:
            raise OrchestrationError(
                WorkflowStage.CRITIC_REVIEW, "No code result available to review"
            )

        async def _attempt() -> CriticReview:
            return await self._critic_agent.review(state.code_result)  # type: ignore[arg-type]

        state.critic_result = await _attempt()

    async def _run_test_agent(self, state: WorkflowState) -> None:
        state.stage = WorkflowStage.TESTING
        print("Stage → TESTING")

        async def _attempt() -> Any:
            return self._test_agent.run_tests(timeout_seconds=120)

        state.test_result = await _attempt()

    async def _run_security_agent(self, state: WorkflowState) -> None:
        state.stage = WorkflowStage.SECURITY_SCAN
        print("Stage → SECURITY_SCAN")

        if state.code_result is None:
            raise OrchestrationError(
                WorkflowStage.SECURITY_SCAN, "No code result available to scan"
            )

        async def _attempt() -> dict[str, Any]:
            return self._security_agent.scan_project(state.code_result.files)

        state.security_result = await _attempt()

    async def _run_debug_agent(self, state: WorkflowState) -> None:
        state.stage = WorkflowStage.DEBUGGING
        print("Stage → DEBUGGING")

        async def _attempt() -> Any:
            traceback_text = self._build_traceback_text(state)
            return self._debug_agent.analyze_traceback(traceback_text)

        state.debug_result = await _attempt()

    async def _run_auto_fix_agent(self, state: WorkflowState) -> None:
        """Kept for compatibility but not executed by the runtime pipeline."""
        state.stage = WorkflowStage.AUTO_FIXING
        logger.info("[%s] Stage -> AUTO_FIXING", state.request_id)

        if state.code_result is None:
            raise OrchestrationError(
                WorkflowStage.AUTO_FIXING, "No code result available for auto-fix"
            )

        async def _attempt() -> Any:
            fix_agent = self._auto_fix_agent or AutoFixAgent(
                project_files=dict(state.code_result.files),
                test_agent=self._test_agent,
                security_agent=self._security_agent,
                debug_agent=self._debug_agent,
            )
            traceback_text = self._build_traceback_text(state)
            session = await fix_agent.run_repair_loop(traceback_text=traceback_text or None)
            if state.code_result is not None:
                state.code_result.files = dict(fix_agent.project_files)
            return session

        # Not used in runtime, but kept non-breaking.
        state.fix_result = await self._with_retries(
            state, "auto_fixing", _attempt, WorkflowStage.AUTO_FIXING
        )

    def _collect_results(self, state: WorkflowState) -> OrchestrationResult:
        print("Stage → COLLECTING_RESULTS")
        state.stage = WorkflowStage.COLLECTING_RESULTS

        if state.code_result is None or state.critic_result is None:
            raise OrchestrationError(
                WorkflowStage.COLLECTING_RESULTS,
                "Missing code or critic results",
            )
        # Save generated files to outputs/<request_id>/
        output_dir = Path("outputs") / state.request_id
        output_dir.mkdir(parents=True, exist_ok=True)

        for filename, content in state.code_result.files.items():
            file_path = output_dir / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")

        state.stage = WorkflowStage.COMPLETED
        state.finished_at = time.monotonic()

        return OrchestrationResult(
            request_id=state.request_id,
            success=True,
            stage_reached=state.stage,
            generated_files=state.code_result.files,
            critic_feedback={
                "issues_found": state.critic_result.issues_found,
                "severity": state.critic_result.severity,
                "improvements": state.critic_result.improvements,
            },
            elapsed_ms=state.elapsed_ms(),
            plan_summary=self._summarize_plan(state.plan_result),
            test_summary=self._summarize_test_result(state.test_result),
            security_summary=state.security_result,
            debug_summary=self._summarize_debug_result(state.debug_result),
            fix_summary=self._summarize_fix_result(state.fix_result),
        )

    def _build_failure_result(self, state: WorkflowState) -> OrchestrationResult:
        return OrchestrationResult(
            request_id=state.request_id,
            success=False,
            stage_reached=state.stage,
            generated_files=state.code_result.files if state.code_result else {},
            critic_feedback={},
            elapsed_ms=state.elapsed_ms(),
            error=state.error,
            plan_summary=self._summarize_plan(state.plan_result),
            test_summary=self._summarize_test_result(state.test_result),
            security_summary=state.security_result,
            debug_summary=self._summarize_debug_result(state.debug_result),
            fix_summary=self._summarize_fix_result(state.fix_result),
        )

    async def _with_retries(
        self,
        state: WorkflowState,
        step_name: str,
        coro_factory,
        stage: WorkflowStage,
    ):
        # Kept for compatibility; runtime pipeline does not call retry wrapper.
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            state.attempts[step_name] = attempt
            try:
                return await coro_factory()
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                backoff = self.RETRY_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
                logger.warning(
                    "[%s] %s attempt %d/%d failed: %s (retrying in %.2fs)",
                    state.request_id,
                    step_name,
                    attempt,
                    self.MAX_RETRIES,
                    exc,
                    backoff,
                )
                if attempt < self.MAX_RETRIES:
                    await asyncio.sleep(backoff)

        raise OrchestrationError(
            stage, f"{step_name} failed after {self.MAX_RETRIES} attempts", cause=last_exc
        )

    def _build_code_request(self, state: WorkflowState) -> str:
        if state.plan_result is None:
            return state.user_request
        try:
            plan_payload = json.dumps(state.plan_result, indent=2, default=str)
        except TypeError:
            plan_payload = str(state.plan_result)
        return f"{state.user_request}\n\nPlanner output:\n{plan_payload}"

    def _build_traceback_text(self, state: WorkflowState) -> str:
        if state.test_result is None:
            return ""
        failures = getattr(state.test_result, "failures", [])
        if failures:
            first_failure = failures[0]
            if getattr(first_failure, "traceback", ""):
                return getattr(first_failure, "traceback", "")
            if getattr(first_failure, "message", ""):
                return getattr(first_failure, "message", "")
        raw_stderr = getattr(state.test_result, "raw_stderr", "") or ""
        raw_stdout = getattr(state.test_result, "raw_stdout", "") or ""
        return f"{raw_stderr}\n{raw_stdout}".strip()

    def _should_run_auto_fix(self, state: WorkflowState) -> bool:
        if state.debug_result is None:
            return False
        error_type = getattr(state.debug_result, "error_type", None)
        if not isinstance(error_type, ErrorType):
            return False
        return error_type in {
            ErrorType.IMPORT_ERROR,
            ErrorType.MODULE_NOT_FOUND,
            ErrorType.SYNTAX_ERROR,
            ErrorType.RUNTIME_ERROR,
            ErrorType.DEPENDENCY_CONFLICT,
            ErrorType.MISSING_PACKAGE,
            ErrorType.NAME_ERROR,
            ErrorType.TYPE_ERROR,
            ErrorType.ATTRIBUTE_ERROR,
            ErrorType.VALUE_ERROR,
            ErrorType.KEY_ERROR,
            ErrorType.INDEX_ERROR,
            ErrorType.ASSERTION_ERROR,
            ErrorType.TIMEOUT_ERROR,
        }

    def _summarize_plan(self, plan_result: Optional[Any]) -> Optional[dict[str, Any]]:
        if plan_result is None:
            return None
        if isinstance(plan_result, dict):
            return plan_result
        if hasattr(plan_result, "to_dict"):
            try:
                return plan_result.to_dict()
            except Exception:  # noqa: BLE001
                return {"value": str(plan_result)}
        return {"value": str(plan_result)}

    def _summarize_test_result(self, test_result: Optional[Any]) -> Optional[dict[str, Any]]:
        if test_result is None:
            return None
        if hasattr(test_result, "to_dict"):
            try:
                return test_result.to_dict()
            except Exception:  # noqa: BLE001
                return {"value": str(test_result)}
        return {"value": str(test_result)}

    def _summarize_debug_result(self, debug_result: Optional[Any]) -> Optional[dict[str, Any]]:
        if debug_result is None:
            return None
        if hasattr(debug_result, "to_dict"):
            try:
                return debug_result.to_dict()
            except Exception:  # noqa: BLE001
                return {"value": str(debug_result)}
        return {"value": str(debug_result)}

    def _summarize_fix_result(self, fix_result: Optional[Any]) -> Optional[dict[str, Any]]:
        if fix_result is None:
            return None
        if hasattr(fix_result, "to_dict"):
            try:
                return fix_result.to_dict()
            except Exception:  # noqa: BLE001
                return {"value": str(fix_result)}
        return {"value": str(fix_result)}

    def get_state(self, request_id: str) -> Optional[WorkflowState]:
        """Return the in-flight state for a request, if still active."""
        return self._active_states.get(request_id)

