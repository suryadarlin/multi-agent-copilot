from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agents.orchestrator import Orchestrator, OrchestrationResult


# -------------------------
# Logging
# -------------------------
logger = logging.getLogger("copilot.main")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# -------------------------
# App
# -------------------------
app = FastAPI()

# -------------------------
# CORS (FIXED - CRITICAL)
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://multi-agent-copilot-1.onrender.com",
        "http://localhost:5500"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Schemas
# -------------------------
class GenerateRequest(BaseModel):
    prompt: str


class StudentIn(BaseModel):
    name: str
    roll_no: str | None = None
    department: str | None = None


# -------------------------
# AI Orchestrator Endpoint
# -------------------------
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
# Students API
# -------------------------
_STUDENTS: dict[str, dict[str, Any]] = {}
_NEXT_ID: int = 1


@app.post("/students")
async def create_student(payload: StudentIn):
    global _NEXT_ID

    if not payload.name.strip():
        raise HTTPException(status_code=400, detail="name is required")

    sid = str(_NEXT_ID)
    _NEXT_ID += 1

    student = {
        "id": sid,
        "name": payload.name,
        "roll_no": payload.roll_no or "",
        "department": payload.department or "",
    }

    _STUDENTS[sid] = student
    return student


@app.get("/students")
async def list_students():
    return list(_STUDENTS.values())


@app.delete("/students/{student_id}")
async def delete_student(student_id: str):
    if student_id not in _STUDENTS:
        raise HTTPException(status_code=404, detail="student not found")

    del _STUDENTS[student_id]
    return {"success": True}


# -------------------------
# Health Check
# -------------------------
@app.get("/")
async def home():
    return {
        "status": "running",
        "message": "Multi-Agent Copilot API",
        "docs": "/docs"
    }


@app.post("/run")
async def run():
    return {"success": True, "status": "live"}


# -------------------------
# CLI mode (optional local run)
# -------------------------
async def _run(user_request: str) -> int:
    orchestrator = Orchestrator()
    result = await orchestrator.receive_request(user_request)

    print(result)
    return 0 if result.success else 1


def main():
    if len(sys.argv) > 1:
        user_request = " ".join(sys.argv[1:])
        return asyncio.run(_run(user_request))

    print("Run with API via FastAPI or pass a prompt")
    return 0


if __name__ == "__main__":
    sys.exit(main())