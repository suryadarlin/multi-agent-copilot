"""
shell_runner.py

Sandboxed shell command execution for the MCP Tool Layer.
Enforces a strict allow-list and pattern-based deny-list before
ever invoking subprocess, so dangerous commands never reach the OS.
"""

from __future__ import annotations

import logging
import re
import shlex
import subprocess
import time
from dataclasses import dataclass, asdict
from typing import Optional

logger = logging.getLogger("mcp_tools.shell_runner")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logger.addHandler(_handler)


ALLOWED_BASE_COMMANDS: frozenset[str] = frozenset(
    {
        "pip",
        "pytest",
        "python",
        "python3",
        "ls",
        "dir",
        "uvicorn",
        "echo",
        "cat",
        "pwd",
        "git",
    }
)

# Regex patterns matched against the full raw command string.
BLOCKED_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"\brm\s+-rf\b", re.IGNORECASE),
    re.compile(r"\bshutdown\b", re.IGNORECASE),
    re.compile(r"\bformat\b", re.IGNORECASE),
    re.compile(r"\bsudo\b", re.IGNORECASE),
    re.compile(r"\bmkfs\b", re.IGNORECASE),
    re.compile(r"\bdd\s+if=", re.IGNORECASE),
    re.compile(r":\(\)\s*\{.*\};:", re.IGNORECASE),  # fork bomb
    re.compile(r"[;&|`]"),  # command chaining / injection separators
    re.compile(r"\$\("),    # command substitution
    re.compile(r">\s*/dev/sd"),  # raw disk writes
)


@dataclass
class ShellResult:
    command: str
    success: bool
    stdout: str
    stderr: str
    blocked: bool
    block_reason: Optional[str] = None
    exit_code: Optional[int] = None
    execution_time_ms: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)


class ShellRunner:
    """
    Executes whitelisted shell commands with strict validation,
    input sanitization, and timeout enforcement.
    """

    def __init__(self, timeout_seconds: int = 30, cwd: Optional[str] = None) -> None:
        self.timeout_seconds = timeout_seconds
        self.cwd = cwd

    def sanitize_input(self, command: str) -> str:
        """Strips and normalizes whitespace; rejects null bytes and control characters."""
        if not isinstance(command, str):
            raise ValueError("Command must be a string.")

        if "\x00" in command:
            raise ValueError("Null byte detected in command input.")

        cleaned = command.strip()
        cleaned = re.sub(r"[\r\n]+", " ", cleaned)

        if not cleaned:
            raise ValueError("Command is empty after sanitization.")

        return cleaned

    def validate_command(self, command: str) -> tuple[bool, Optional[str]]:
        """
        Validates a sanitized command against the deny-list and allow-list.
        Returns (is_valid, reason_if_invalid).
        """
        for pattern in BLOCKED_PATTERNS:
            if pattern.search(command):
                return False, f"Command matched blocked pattern: {pattern.pattern}"

        try:
            tokens = shlex.split(command)
        except ValueError as exc:
            return False, f"Command failed to parse safely: {exc}"

        if not tokens:
            return False, "No command tokens found."

        base_command = tokens[0].lower()
        # Strip a path prefix if present (e.g. /usr/bin/python -> python)
        base_command = base_command.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]

        if base_command not in ALLOWED_BASE_COMMANDS:
            return False, f"Command '{base_command}' is not in the allowed command whitelist."

        return True, None

    def execute_command(self, raw_command: str) -> ShellResult:
        """
        Sanitizes, validates, and executes a shell command if permitted.
        Always returns a structured ShellResult, never raises on bad input.
        """
        try:
            command = self.sanitize_input(raw_command)
        except ValueError as exc:
            logger.warning("Rejected command at sanitization stage: %s", exc)
            return ShellResult(
                command=str(raw_command),
                success=False,
                stdout="",
                stderr="",
                blocked=True,
                block_reason=str(exc),
            )

        is_valid, reason = self.validate_command(command)
        if not is_valid:
            logger.warning("Blocked command: %s | reason: %s", command, reason)
            return ShellResult(
                command=command,
                success=False,
                stdout="",
                stderr="",
                blocked=True,
                block_reason=reason,
            )

        start_time = time.monotonic()
        try:
            tokens = shlex.split(command)
            completed = subprocess.run(
                tokens,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                cwd=self.cwd,
                shell=False,
            )
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            logger.info("Executed command '%s' in %sms (exit=%s)", command, elapsed_ms, completed.returncode)

            return ShellResult(
                command=command,
                success=completed.returncode == 0,
                stdout=completed.stdout or "",
                stderr=completed.stderr or "",
                blocked=False,
                exit_code=completed.returncode,
                execution_time_ms=elapsed_ms,
            )

        except subprocess.TimeoutExpired:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            logger.warning("Command '%s' timed out after %sms", command, elapsed_ms)
            return ShellResult(
                command=command,
                success=False,
                stdout="",
                stderr=f"Command timed out after {self.timeout_seconds}s",
                blocked=False,
                exit_code=-1,
                execution_time_ms=elapsed_ms,
            )
        except (OSError, FileNotFoundError) as exc:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            logger.error("Execution error for '%s': %s", command, exc)
            return ShellResult(
                command=command,
                success=False,
                stdout="",
                stderr=str(exc),
                blocked=False,
                exit_code=-1,
                execution_time_ms=elapsed_ms,
            )