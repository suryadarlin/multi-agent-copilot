# FastAPI adapter layer exposing the existing Python backend.
# Does not modify any agent/orchestrator internals.

from __future__ import annotations

import asyncio
import io
import time
import zipfile
import traceback
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agents.orchestrator import Orchestrator, WorkflowStage
from agents.orchestrator import OrchestrationResult
from validate_project import main as health_main

app = FastAPI(title="Multi-Agent AI Engineering Copilot API")

# Allow local dev; no auth/management added.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    prompt: str


class GenerateResponse(BaseModel):
    request_id: str
    success: bool
    stage_reached: str
    elapsed_ms: float
    generated_files: Dict[str, str]
    critic_feedback: Dict[str, Any]


# In-flight state (single-process)
_active: Dict[str, Dict[str, Any]] = {}


def _stage_to_key(stage: WorkflowStage) -> str:
    mapping = {
        WorkflowStage.PLANNING: "planning",
        WorkflowStage.CODE_GENERATION: "code_generation",
        WorkflowStage.CRITIC_REVIEW: "critic_review",
        WorkflowStage.SECURITY_SCAN: "security_scan",
        WorkflowStage.TESTING: "testing",
        WorkflowStage.DEBUGGING: "debugging",
        WorkflowStage.AUTO_FIXING: "auto_fix",
        WorkflowStage.COMPLETED: "completed",
        WorkflowStage.FAILED: "failed",
    }
    return mapping.get(stage, "received")


async def _run_workflow(request_id: str, prompt: str) -> None:
    orch = Orchestrator()
    _active[request_id] = {
        "stage": "received",
        "logs": [],
        "generated_files": {},
        "success": False,
        "critic_feedback": {},
        "elapsed_ms": 0.0,
        "error": None,
        "started_at": time.time(),
        "finished": False,
    }

    # We cannot change orchestrator internals to emit per-agent logs.
    # We therefore surface the final OrchestrationResult only, and set stage
    # as orchestrator's stage_reached.
    try:
        result: OrchestrationResult = await orch.receive_request(prompt)
        _active[request_id]["stage"] = _stage_to_key(result.stage_reached)
        _active[request_id]["generated_files"] = result.generated_files
        _active[request_id]["success"] = result.success
        _active[request_id]["critic_feedback"] = result.critic_feedback
        _active[request_id]["elapsed_ms"] = result.elapsed_ms

        # Provide a minimal logs trace compatible with the frontend.
        # Each stage maps to a single log line when reached.
        _active[request_id]["logs"] = [
            {
                "agent": "Orchestrator",
                "status": "success" if result.success else "failed",
                "stage": result.stage_reached.value,
                "message": f"Workflow {result.stage_reached.value}",
                "detail": {
                    "request_id": result.request_id,
                },
                "timestamp": time.time(),
            }
        ]

    except Exception as exc:  # noqa: BLE001
        _active[request_id]["stage"] = "failed"
        _active[request_id]["error"] = str(exc)
        _active[request_id]["logs"].append(
            {
                "agent": "Orchestrator",
                "status": "failed",
                "stage": "failed",
                "message": "Fatal workflow error",
                "detail": {"traceback": traceback.format_exc()},
                "timestamp": time.time(),
            }
        )
    finally:
        _active[request_id]["finished"] = True


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    prompt = (req.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")

    request_id = str(int(time.time() * 1000))

    # Run async so the client can immediately poll /status.
    asyncio.create_task(_run_workflow(request_id, prompt))

    # Return initial stub response.
    return GenerateResponse(
        request_id=request_id,
        success=False,
        stage_reached="received",
        elapsed_ms=0.0,
        generated_files={},
        critic_feedback={},
    )


@app.get("/status")
async def status(request_id: str) -> Dict[str, Any]:
    if request_id not in _active:
        raise HTTPException(status_code=404, detail="Unknown request_id")
    return {
        "request_id": request_id,
        "stage": _active[request_id]["stage"],
        "finished": _active[request_id]["finished"],
        "success": _active[request_id]["success"],
        "elapsed_ms": _active[request_id]["elapsed_ms"],
        "error": _active[request_id]["error"],
    }


@app.get("/logs")
async def logs(request_id: str) -> Dict[str, Any]:
    if request_id not in _active:
        raise HTTPException(status_code=404, detail="Unknown request_id")
    return {"request_id": request_id, "logs": _active[request_id]["logs"]}


@app.get("/health")
async def health() -> Dict[str, Any]:
    # validate_project.py prints score; do not change it, just reuse.
    # Run synchronously (fast enough for monitoring).
    # Return an interpreted score.
    try:
        # validate_project.py returns exit code (0 for >=70)
        code = health_main()
        return {"passed": code == 0}
    except Exception as exc:  # noqa: BLE001
        return {"passed": False, "error": str(exc)}


def _build_zip_bytes(generated_files: Dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        if not generated_files:
            zf.writestr("README.txt", "No generated files")
        for filename, content in generated_files.items():
            zf.writestr(filename, content)
    buffer.seek(0)
    return buffer.read()


@app.get("/download")
async def download(request_id: str) -> Dict[str, Any]:
    if request_id not in _active:
        raise HTTPException(status_code=404, detail="Unknown request_id")
    if not _active[request_id]["finished"]:
        raise HTTPException(status_code=409, detail="Workflow not finished")
    zip_bytes = _build_zip_bytes(_active[request_id]["generated_files"])

    # Returning bytes via JSON is not ideal; use base64 would require extra work.
    # For simplicity, return as a data URL payload.
    import base64

    b64 = base64.b64encode(zip_bytes).decode("ascii")
    return {
        "request_id": request_id,
        "filename": f"copilot_output_{request_id}.zip",
        "content_type": "application/zip",
        "base64_zip": b64,
    }

