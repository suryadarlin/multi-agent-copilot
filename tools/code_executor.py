"""
code_executor.py

Secure execution sandbox for AI-generated Python code.
Part of the MCP Tool Layer for the AI Software Engineering Copilot.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger("mcp_tools.code_executor")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logger.addHandler(_handler)


class CodeExecutionError(Exception):
    """Raised when code execution infrastructure itself fails (not user code errors)."""


@dataclass
class ExecutionResult:
    success: bool
    stdout: str
    stderr: str
    exit_code: int
    execution_time_ms: int
    timed_out: bool = False
    error_message: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


class CodeExecutor:
    """
    Executes AI-generated Python code in an isolated subprocess with
    timeout protection, output capture, and guaranteed cleanup.
    """

    def __init__(
        self,
        work_dir: Optional[str] = None,
        timeout_seconds: int = 15,
        python_executable: Optional[str] = None,
    ) -> None:
        self._base_dir = Path(work_dir) if work_dir else Path(tempfile.gettempdir()) / "mcp_code_exec"
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = timeout_seconds
        self.python_executable = python_executable or sys.executable
        self._active_files: list[Path] = []

    def write_temp_code(self, code: str) -> Path:
        """Writes generated code to a uniquely named temp file and returns its path."""
        if not isinstance(code, str) or not code.strip():
            raise CodeExecutionError("Cannot execute empty or non-string code payload.")

        file_id = uuid.uuid4().hex
        temp_path = self._base_dir / f"snippet_{file_id}.py"

        try:
            temp_path.write_text(code, encoding="utf-8")
        except OSError as exc:
            raise CodeExecutionError(f"Failed to write temp code file: {exc}") from exc

        self._active_files.append(temp_path)
        logger.info("Wrote temp code file: %s", temp_path)
        return temp_path

    def execute_python_code(self, code: str) -> ExecutionResult:
        """
        Writes the given code to disk, executes it in a subprocess,
        captures output, and cleans up regardless of outcome.
        """
        temp_path: Optional[Path] = None
        start_time = time.monotonic()

        try:
            temp_path = self.write_temp_code(code)
            result = self._run_subprocess(temp_path)
            return result
        except CodeExecutionError as exc:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            logger.error("Code execution infrastructure failure: %s", exc)
            return ExecutionResult(
                success=False,
                stdout="",
                stderr="",
                exit_code=-1,
                execution_time_ms=elapsed_ms,
                timed_out=False,
                error_message=str(exc),
            )
        finally:
            if temp_path is not None:
                self.cleanup_temp_files([temp_path])

    def _run_subprocess(self, file_path: Path) -> ExecutionResult:
        start_time = time.monotonic()
        try:
            completed = subprocess.run(
                [self.python_executable, str(file_path)],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                cwd=str(self._base_dir),
            )
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            return self.capture_execution_output(completed, elapsed_ms, timed_out=False)

        except subprocess.TimeoutExpired as exc:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            logger.warning("Execution timed out after %sms for %s", elapsed_ms, file_path)
            return ExecutionResult(
                success=False,
                stdout=exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or ""),
                stderr=exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or ""),
                exit_code=-1,
                execution_time_ms=elapsed_ms,
                timed_out=True,
                error_message=f"Execution exceeded timeout of {self.timeout_seconds}s",
            )
        except OSError as exc:
            raise CodeExecutionError(f"Subprocess failed to start: {exc}") from exc

    def capture_execution_output(
        self,
        completed: subprocess.CompletedProcess,
        elapsed_ms: int,
        timed_out: bool,
    ) -> ExecutionResult:
        """Normalizes a CompletedProcess into a structured ExecutionResult."""
        success = completed.returncode == 0
        return ExecutionResult(
            success=success,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
            exit_code=completed.returncode,
            execution_time_ms=elapsed_ms,
            timed_out=timed_out,
            error_message=None if success else "Process exited with non-zero status.",
        )

    def cleanup_temp_files(self, files: Optional[list[Path]] = None) -> None:
        """Removes temp files created during execution. Cleans all tracked files if none specified."""
        targets = files if files is not None else list(self._active_files)
        for path in targets:
            try:
                if path.exists():
                    path.unlink()
                if path in self._active_files:
                    self._active_files.remove(path)
            except OSError as exc:
                logger.warning("Failed to clean up temp file %s: %s", path, exc)

    def __del__(self):
        try:
            self.cleanup_temp_files()
        except Exception:
            pass