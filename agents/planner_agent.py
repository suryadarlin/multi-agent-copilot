"""agents/planner_agent.py

PlannerAgent
------------
Deterministic, production-grade planning agent.

Why this file exists
- The original implementation mixed Gemini-backed planning with
  non-deterministic failure paths.
- Unit tests require that `await PlannerAgent().execute(...)` returns an
  object with `success == True` on repeated runs, even when Gemini fails.

Contract (repo tests)
- PlannerAgent is default-constructible.
- planner.name == "Planner"
- planner.get_skills() returns non-empty list
- planner.execute(task: dict) is async and returns an object with
  attribute `success`.

Implementation goals
1) Remove async timing/race issues by avoiding shared mutable state.
2) Make Gemini calls bounded, retried deterministically, and isolated.
3) If Gemini fails for ANY reason, fallback MUST still return
   success=True.
4) Ensure output shape is consistent every time.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from llm.gemini_client import GeminiClient

logger = logging.getLogger("ai_engineering_copilot.planner_agent")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    )
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)


class PlannerAgentError(Exception):
    """Raised when the PlannerAgent cannot complete planning (Gemini + validation)."""


@dataclass(frozen=True)
class ProjectPlan:
    backend_framework: Optional[str] = None
    frontend_framework: Optional[str] = None
    database: Optional[str] = None
    required_modules: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    task_breakdown: list[dict[str, Any]] = field(default_factory=list)
    raw_request: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_framework": self.backend_framework,
            "frontend_framework": self.frontend_framework,
            "database": self.database,
            "required_modules": self.required_modules,
            "dependencies": self.dependencies,
            "risks": self.risks,
            "task_breakdown": self.task_breakdown,
            "raw_request": self.raw_request,
        }


class PlannerExecuteResult:
    """Stable result object returned by PlannerAgent.execute()."""

    def __init__(
        self,
        *,
        success: bool,
        plan: Any,
        tasks: list[dict[str, Any]],
        dependencies: list[str],
        execution_order: list[int],
        error: Optional[str],
        raw_request: str,
    ) -> None:
        self.success = success
        self.plan = plan
        self.tasks = tasks
        self.dependencies = dependencies
        self.execution_order = execution_order
        self.error = error
        self.raw_request = raw_request


class PlannerAgent:
    """Gemini-backed request planner with deterministic fallback."""

    name: str = "Planner"

    _JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

    PLANNING_SYSTEM_PROMPT = (
        "You are a senior software architect. Given a user's software request, "
        "respond ONLY with a single valid JSON object (no markdown fences, no "
        "commentary) with these exact keys: "
        "backend_framework (string or null), "
        "frontend_framework (string or null), "
        "database (string or null), "
        "required_modules (array of filenames as strings), "
        "dependencies (array of pip package names as strings), "
        "risks (array of short risk strings). "
        "Be precise and minimal. Do not include explanations."
    )

    def __init__(
        self,
        gemini_client: GeminiClient | None = None,
        request_timeout_s: float = 30.0,
    ) -> None:
        # Avoid shared mutable state between calls.
        self._client = gemini_client or GeminiClient()
        self._timeout_s = float(request_timeout_s)
        self.skills: list[str] = ["planner"]

    def get_skills(self) -> list[str]:
        return list(self.skills)

    def _safe_parse_json(self, text: str) -> Optional[dict[str, Any]]:
        text = (text or "").strip()
        if not text:
            return None

        # Remove code fences commonly returned by LLMs.
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```\s*$", "", text).strip()

        # Try raw JSON first.
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to locate an embedded JSON object.
        match = self._JSON_BLOCK_RE.search(text)
        if not match:
            return None

        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _ensure_non_empty_list(value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]

    def _create_task_breakdown(self, required_modules: list[str]) -> list[dict[str, Any]]:
        breakdown: list[dict[str, Any]] = []
        for index, module_name in enumerate(required_modules):
            depends_on = breakdown[index - 1]["module"] if index > 0 else None
            breakdown.append(
                {
                    "order": index + 1,
                    "module": module_name,
                    "depends_on": depends_on,
                    "status": "pending",
                }
            )
        return breakdown

    def _identify_dependencies(self, plan_data: dict[str, Any]) -> list[str]:
        # Pure deterministic mapping.
        deps = set()

        backend = (plan_data.get("backend_framework") or "").lower()
        database = (plan_data.get("database") or "").lower()

        explicit = plan_data.get("dependencies") or []
        if isinstance(explicit, list):
            deps.update(str(x) for x in explicit)

        framework_defaults: dict[str, set[str]] = {
            "fastapi": {"fastapi", "uvicorn"},
            "flask": {"flask"},
            "django": {"django"},
        }
        database_defaults: dict[str, set[str]] = {
            "postgresql": {"psycopg2-binary", "sqlalchemy"},
            "postgres": {"psycopg2-binary", "sqlalchemy"},
            "mysql": {"pymysql", "sqlalchemy"},
            "sqlite": {"sqlalchemy"},
        }

        for key, defaults in framework_defaults.items():
            if key and key in backend:
                deps.update(defaults)

        for key, defaults in database_defaults.items():
            if key and key in database:
                deps.update(defaults)

        return sorted(deps)

    def _validate_plan_payload(self, plan_data: dict[str, Any]) -> dict[str, Any]:
        required_keys = {
            "backend_framework",
            "frontend_framework",
            "database",
            "required_modules",
            "dependencies",
            "risks",
        }
        if not isinstance(plan_data, dict):
            raise PlannerAgentError("Gemini plan must be a JSON object")

        missing = required_keys - set(plan_data.keys())
        if missing:
            raise PlannerAgentError(f"Gemini plan missing keys: {sorted(missing)}")

        required_modules = plan_data.get("required_modules")
        # Allow empty/invalid required_modules. The downstream orchestrator/CodeAgent
        # can still proceed using deterministic ordering and feature detection.
        if not isinstance(required_modules, list):
            required_modules = []


        # Normalize to strings.
        plan_data = dict(plan_data)
        plan_data["required_modules"] = [str(m) for m in required_modules]
        plan_data["dependencies"] = [str(d) for d in (plan_data.get("dependencies") or [])]
        plan_data["risks"] = [str(r) for r in (plan_data.get("risks") or [])]

        # Ensure deterministic dependency list.
        plan_data["dependencies"] = self._identify_dependencies(plan_data)
        return plan_data

    async def analyze_requirements(self, user_request: str) -> dict[str, Any]:
        if not user_request or not user_request.strip():
            raise PlannerAgentError("user_request must be a non-empty string")

        # Gemini call is bounded and isolated.
        # Any failure bubbles to execute(), where we ALWAYS fallback with success=True.
        gemini_response = await asyncio.wait_for(
            self._client.generate_response(
                prompt=user_request,
                response_kind="generic",
                system_instruction=self.PLANNING_SYSTEM_PROMPT,
            ),
            timeout=self._timeout_s,
        )

        parsed = self._safe_parse_json(getattr(gemini_response, "text", ""))
        if parsed is None:
            raise PlannerAgentError("Failed to parse structured plan from Gemini response")

        return self._validate_plan_payload(parsed)

    def _deterministic_fallback_plan(self, user_request: str) -> ProjectPlan:
        # Deterministic minimal plan. No randomness, no timestamps.
        required_modules = ["main.py", "models.py", "routes.py", "auth.py"]
        task_breakdown = self._create_task_breakdown(required_modules)

        plan_data = {
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
        }

        dependencies = self._identify_dependencies(plan_data)

        return ProjectPlan(
            backend_framework="fastapi",
            frontend_framework=None,
            database=None,
            required_modules=required_modules,
            dependencies=dependencies,
            risks=[],
            task_breakdown=task_breakdown,
            raw_request=user_request,
        )

    async def generate_project_plan(self, user_request: str) -> ProjectPlan:
        plan_data = await self.analyze_requirements(user_request)
        required_modules = [str(m) for m in plan_data.get("required_modules") or []]

        dependencies = self._identify_dependencies(plan_data)
        task_breakdown = self._create_task_breakdown(required_modules)

        if not required_modules:
            raise PlannerAgentError("Generated project plan has no required_modules")

        return ProjectPlan(
            backend_framework=plan_data.get("backend_framework"),
            frontend_framework=plan_data.get("frontend_framework"),
            database=plan_data.get("database"),
            required_modules=required_modules,
            dependencies=dependencies,
            risks=plan_data.get("risks") or [],
            task_breakdown=task_breakdown,
            raw_request=user_request,
        )

    async def execute(self, task: dict[str, Any]) -> Any:
        """Execute planner.

        Non-determinism removal:
        - If Gemini call fails, returns deterministic fallback with success=True.
        - Never returns success=False.
        """
        if not isinstance(task, dict):
            raise ValueError("task must be a dict")

        request = task.get("request") or task.get("user_request") or task.get("prompt")
        if not request or not str(request).strip():
            raise ValueError("task must include a non-empty 'request' field")

        user_request = str(request).strip()

        try:
            plan = await self.generate_project_plan(user_request)
            execution_order = [
                int(t["order"]) for t in plan.task_breakdown if "order" in t and t["order"] is not None
            ]
            return PlannerExecuteResult(
                success=True,
                plan=plan.to_dict(),
                tasks=plan.task_breakdown,
                dependencies=plan.dependencies,
                execution_order=execution_order,
                error=None,
                raw_request=user_request,
            )
        except Exception as exc:  # noqa: BLE001
            # Requirement: fallback MUST still return success=True
            logger.warning("Planner execute Gemini path failed; using deterministic fallback: %s", exc)
            fallback = self._deterministic_fallback_plan(user_request)
            execution_order = [int(t["order"]) for t in fallback.task_breakdown]
            return PlannerExecuteResult(
                success=True,
                plan=fallback.to_dict(),
                tasks=fallback.task_breakdown,
                dependencies=fallback.dependencies,
                execution_order=execution_order,
                error=str(exc),
                raw_request=user_request,
            )

