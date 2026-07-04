"""
monitoring/logger.py
=====================

Production-grade structured logging for the AI Software Engineering
Copilot. Emits JSON-formatted log records so they can be ingested by
Cloud Logging, Loki, or any log aggregator without a custom parser.

Provides distinct loggers/helpers for:
    - agent execution events (per-agent lifecycle)
    - workflow-level events (the orchestrator pipeline as a whole)
    - error events (exceptions, with stack traces)
    - execution timing (duration of any traced block)
"""

from __future__ import annotations

import json
import logging
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

LOG_DIR = Path("/app/logs") if Path("/app").exists() else Path("./logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)


class JSONFormatter(logging.Formatter):
    """Renders every log record as a single line of JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Attach any structured extras passed via `extra={...}`
        for key, value in record.__dict__.items():
            if key in (
                "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
                "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
                "created", "msecs", "relativeCreated", "thread", "threadName",
                "processName", "process", "message",
            ):
                continue
            payload[key] = value

        if record.exc_info:
            payload["exception"] = "".join(traceback.format_exception(*record.exc_info))

        return json.dumps(payload, default=str)


def _build_logger(name: str, filename: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured (avoid duplicate handlers on reimport)

    logger.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(JSONFormatter())
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(LOG_DIR / filename)
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger


agent_logger = _build_logger("copilot.agents", "agent.log")
workflow_logger = _build_logger("copilot.workflow", "workflow.log")
error_logger = _build_logger("copilot.errors", "error.log")


@dataclass
class AgentLogEvent:
    agent_name: str
    status: str
    requirement_id: Optional[str] = None
    duration_ms: Optional[float] = None
    retry_count: int = 0
    detail: dict[str, Any] = field(default_factory=dict)


def log_agent_event(event: AgentLogEvent) -> None:
    agent_logger.info(
        f"{event.agent_name} -> {event.status}",
        extra={
            "agent_name": event.agent_name,
            "status": event.status,
            "requirement_id": event.requirement_id,
            "duration_ms": event.duration_ms,
            "retry_count": event.retry_count,
            "detail": event.detail,
        },
    )


def log_workflow_event(stage: str, status: str, requirement_id: Optional[str] = None, **extra: Any) -> None:
    workflow_logger.info(
        f"workflow:{stage} -> {status}",
        extra={"stage": stage, "status": status, "requirement_id": requirement_id, **extra},
    )


def log_error(message: str, exc: Optional[BaseException] = None, **extra: Any) -> None:
    if exc is not None:
        error_logger.error(message, exc_info=exc, extra=extra)
    else:
        error_logger.error(message, extra=extra)


@contextmanager
def timed_operation(operation_name: str, logger_to_use: logging.Logger = workflow_logger) -> Iterator[dict[str, Any]]:
    """
    Context manager that logs the start/end and duration of an operation.

    Usage:
        with timed_operation("code_generation") as ctx:
            ... do work ...
            ctx["files_generated"] = 3
    """
    start = time.perf_counter()
    ctx: dict[str, Any] = {}
    logger_to_use.info(f"{operation_name} started", extra={"operation": operation_name, "phase": "start"})
    try:
        yield ctx
    except Exception as exc:  # noqa: BLE001 - intentionally broad, re-raised below
        duration_ms = (time.perf_counter() - start) * 1000
        log_error(f"{operation_name} failed", exc=exc, operation=operation_name, duration_ms=duration_ms)
        raise
    else:
        duration_ms = (time.perf_counter() - start) * 1000
        logger_to_use.info(
            f"{operation_name} completed",
            extra={"operation": operation_name, "phase": "end", "duration_ms": duration_ms, **ctx},
        )