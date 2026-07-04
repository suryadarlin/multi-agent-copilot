"""
file_reader.py

Safe, traversal-resistant file inspection utilities for the MCP Tool Layer.
Restricts all file access to a configured project root.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("mcp_tools.file_reader")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logger.addHandler(_handler)


class PathSecurityError(Exception):
    """Raised when a requested path attempts to escape the sandboxed project root."""


@dataclass
class FileReadResult:
    filename: str
    size: int
    extension: str
    content: Optional[str]
    exists: bool
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FileMetadata:
    filename: str
    size: int
    extension: str
    exists: bool
    is_directory: bool
    modified_time: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DirectoryListing:
    path: str
    entries: list[str] = field(default_factory=list)
    directories: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


class FileReader:
    """
    Provides sandboxed, read-only access to a project directory.
    All paths are resolved and validated against project_root to
    prevent directory traversal attacks (e.g. ../../../etc/passwd).
    """

    MAX_READ_BYTES = 5 * 1024 * 1024  # 5 MB cap on file reads

    def __init__(self, project_root: str) -> None:
        resolved_root = Path(project_root).resolve()
        if not resolved_root.exists() or not resolved_root.is_dir():
            raise ValueError(f"project_root does not exist or is not a directory: {resolved_root}")
        self.project_root = resolved_root

    def _resolve_safe_path(self, relative_path: str) -> Path:
        """
        Resolves a user-supplied relative path against project_root and
        verifies the result does not escape the sandbox.
        """
        if relative_path is None:
            raise PathSecurityError("Path cannot be None.")

        candidate = (self.project_root / relative_path).resolve()

        try:
            candidate.relative_to(self.project_root)
        except ValueError as exc:
            logger.warning("Blocked path traversal attempt: %s", relative_path)
            raise PathSecurityError(
                f"Path '{relative_path}' resolves outside the project root."
            ) from exc

        return candidate

    def file_exists(self, relative_path: str) -> bool:
        try:
            resolved = self._resolve_safe_path(relative_path)
            return resolved.exists()
        except PathSecurityError:
            return False

    def read_file(self, relative_path: str) -> FileReadResult:
        """Reads a text file's contents within the sandbox, with size capping."""
        try:
            resolved = self._resolve_safe_path(relative_path)
        except PathSecurityError as exc:
            return FileReadResult(
                filename=relative_path,
                size=0,
                extension="",
                content=None,
                exists=False,
                error=str(exc),
            )

        if not resolved.exists() or not resolved.is_file():
            return FileReadResult(
                filename=relative_path,
                size=0,
                extension=resolved.suffix,
                content=None,
                exists=False,
                error="File does not exist or is not a regular file.",
            )

        try:
            size = resolved.stat().st_size
            if size > self.MAX_READ_BYTES:
                return FileReadResult(
                    filename=relative_path,
                    size=size,
                    extension=resolved.suffix,
                    content=None,
                    exists=True,
                    error=f"File exceeds maximum readable size of {self.MAX_READ_BYTES} bytes.",
                )

            content = resolved.read_text(encoding="utf-8", errors="replace")
            return FileReadResult(
                filename=relative_path,
                size=size,
                extension=resolved.suffix,
                content=content,
                exists=True,
            )
        except OSError as exc:
            logger.error("Failed to read file %s: %s", resolved, exc)
            return FileReadResult(
                filename=relative_path,
                size=0,
                extension=resolved.suffix,
                content=None,
                exists=True,
                error=str(exc),
            )

    def list_directory(self, relative_path: str = ".") -> DirectoryListing:
        """Lists immediate contents of a directory within the sandbox."""
        try:
            resolved = self._resolve_safe_path(relative_path)
        except PathSecurityError as exc:
            return DirectoryListing(path=relative_path, error=str(exc))

        if not resolved.exists() or not resolved.is_dir():
            return DirectoryListing(path=relative_path, error="Path does not exist or is not a directory.")

        try:
            entries = sorted(resolved.iterdir(), key=lambda p: p.name.lower())
            files = [e.name for e in entries if e.is_file()]
            directories = [e.name for e in entries if e.is_dir()]
            return DirectoryListing(
                path=relative_path,
                entries=[e.name for e in entries],
                directories=directories,
                files=files,
            )
        except OSError as exc:
            logger.error("Failed to list directory %s: %s", resolved, exc)
            return DirectoryListing(path=relative_path, error=str(exc))

    def get_file_metadata(self, relative_path: str) -> FileMetadata:
        try:
            resolved = self._resolve_safe_path(relative_path)
        except PathSecurityError as exc:
            return FileMetadata(
                filename=relative_path,
                size=0,
                extension="",
                exists=False,
                is_directory=False,
            )

        if not resolved.exists():
            return FileMetadata(
                filename=relative_path,
                size=0,
                extension=resolved.suffix,
                exists=False,
                is_directory=False,
            )

        try:
            stat_result = resolved.stat()
            return FileMetadata(
                filename=relative_path,
                size=stat_result.st_size,
                extension=resolved.suffix,
                exists=True,
                is_directory=resolved.is_dir(),
                modified_time=stat_result.st_mtime,
            )
        except OSError as exc:
            logger.error("Failed to stat file %s: %s", resolved, exc)
            return FileMetadata(
                filename=relative_path,
                size=0,
                extension=resolved.suffix,
                exists=False,
                is_directory=False,
            )