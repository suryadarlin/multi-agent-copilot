from __future__ import annotations

import asyncio
import logging
import sys
import time
from typing import Any


from agents.orchestrator import Orchestrator, OrchestrationResult


try:
    from fastapi import FastAPI
except ModuleNotFoundError:  # pragma: no cover
    FastAPI = None  # type: ignore[assignment]

try:
    from pydantic import BaseModel
except ModuleNotFoundError:  # pragma: no cover
    BaseModel = object  # type: ignore[assignment]

try:
    from fastapi.middleware.cors import CORSMiddleware
except ModuleNotFoundError:  # pragma: no cover
    CORSMiddleware = None  # type: ignore[assignment]


logger = logging.getLogger("copilot.main")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def _print_result(result: OrchestrationResult) -> None:
    print("\n" + "=" * 70)
    print(f"Request ID:     {result.request_id}")
    print(f"Success:        {result.success}")
    print(f"Stage reached:  {result.stage_reached.value}")
    print(f"Elapsed:        {result.elapsed_ms} ms")
    print("=" * 70)

    if not result.success:
        print(f"\nERROR: {result.error}\n")
        return

    print(f"\nGenerated {len(result.generated_files)} file(s):")
    for filename in result.generated_files:
        print(f"  - {filename}")

    severity = result.critic_feedback.get("severity", "NONE")
    issues = result.critic_feedback.get("issues_found", [])
    improvements = result.critic_feedback.get("improvements", [])

    print(f"\nCritic review severity: {severity}")
    print(f"Issues found: {len(issues)}")
    for improvement in improvements:
        print(f"  * {improvement}")

    print("\n--- Generated files ---\n")
    for filename, content in result.generated_files.items():
        print(f"--- {filename} ---\n")
        print(content)
        print()


async def _run(user_request: str) -> int:
    orchestrator = Orchestrator()
    try:
        result = await orchestrator.receive_request(user_request)
    except ValueError as exc:
        logger.error("Invalid request: %s", exc)
        print(f"Error: {exc}")
        return 1
    except Exception:  # noqa: BLE001 - top-level safety net for the CLI
        logger.exception("Unhandled error while processing request")
        print("An unexpected error occurred. Check logs for details.")
        return 1

    _print_result(result)
    return 0 if result.success else 1


app = FastAPI() if FastAPI is not None else None


# -------------------------
# Basic CORS (for frontend)
# -------------------------
# Minimal fix: allow browser to call the backend from local frontend.
# NOTE: If allow_credentials=True, '*' cannot be used for allow_origins.
# Use explicit localhost origins instead.
if app is not None and CORSMiddleware is not None:
    app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "https://multi-agent-copilot-1.onrender.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    prompt: str


@app.post("/generate")
async def generate(request: GenerateRequest) -> dict[str, Any]:
    orchestrator = Orchestrator()
    result = await orchestrator.receive_request(request.prompt)
    return {
        "success": result.success,
        "request_id": result.request_id,
        "generated_files": list(result.generated_files.keys()),
        "critic_feedback": result.critic_feedback,
        "elapsed_ms": result.elapsed_ms,
    }


# -------------------------
# /students endpoints (used by tests/test_api.py)
# -------------------------
class StudentIn(BaseModel):
    name: str
    roll_no: str | None = None
    department: str | None = None



# In-memory store
_STUDENTS: dict[str, dict[str, Any]] = {}
_NEXT_ID: int = 1


@app.post("/students")
async def create_student(payload: StudentIn) -> dict[str, Any]:
    global _NEXT_ID
    name = (payload.name or "").strip()
    roll_no = (payload.roll_no or "").strip()
    department = (payload.department or "").strip()
    from fastapi import HTTPException

    if not name:
        raise HTTPException(status_code=400, detail="name is required")

    # roll_no/department are optional for older tests & can be populated by the frontend.
    # Frontend may send them; if missing, store empty strings.
    if not roll_no:
        roll_no = ""
    if not department:
        department = ""

    sid = str(_NEXT_ID)

    _NEXT_ID += 1
    student = {
        "id": sid,
        "name": name,
        "roll_no": payload.roll_no,
        "department": payload.department,
    }
    _STUDENTS[sid] = student
    return student



@app.get("/students")
async def list_students() -> list[dict[str, Any]]:
    # tests accept list directly
    return list(_STUDENTS.values())


@app.delete("/students/{student_id}")
async def delete_student(student_id: str) -> None:
    if student_id in _STUDENTS:
        _STUDENTS.pop(student_id, None)
        return None
    # If missing, return 404
    from fastapi import HTTPException

    raise HTTPException(status_code=404, detail="student not found")


def main() -> int:
    if len(sys.argv) > 1:
        user_request = " ".join(sys.argv[1:])
    else:
        print("AI Software Engineering Copilot")
        print("Describe the software you want built (e.g. 'Build a FastAPI JWT login system'):")
        try:
            user_request = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return 1

    if not user_request:
        print("No request provided. Exiting.")
        return 1

    return asyncio.run(_run(user_request))


@app.get("/")
async def home():
    return {
        "message": "Multi-Agent AI Engineering Copilot API",
        "status": "running",
        "docs": "/docs",
        "openapi": "/openapi.json"
    }

@app.post("/run")
def run():
    return {
        "success": True,
        "message": "Backend connected successfully",
        "status": "live"
    }

if __name__ == "__main__":
    sys.exit(main())

