"""
debug_agent.py

DebugAgent: performs root-cause analysis on execution failures captured from
mcp_tools.code_executor / shell_runner output (stderr + traceback text),
classifies the failure type, locates the responsible module, and proposes a
remediation strategy for the AutoFixAgent to act on.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger("ai_engineering_copilot.agents.debug_agent")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class ErrorType(str, Enum):
    IMPORT_ERROR = "ImportError"
    MODULE_NOT_FOUND = "ModuleNotFoundError"
    SYNTAX_ERROR = "SyntaxError"
    NAME_ERROR = "NameError"
    TYPE_ERROR = "TypeError"
    ATTRIBUTE_ERROR = "AttributeError"
    VALUE_ERROR = "ValueError"
    KEY_ERROR = "KeyError"
    INDEX_ERROR = "IndexError"
    RUNTIME_ERROR = "RuntimeError"
    DEPENDENCY_CONFLICT = "DependencyConflict"
    MISSING_PACKAGE = "MissingPackage"
    ASSERTION_ERROR = "AssertionError"
    TIMEOUT_ERROR = "TimeoutError"
    UNKNOWN = "UnknownError"


@dataclass
class DebugAnalysis:
    error_type: ErrorType
    faulty_module: str
    root_cause: str
    suggested_fix: str
    confidence: float
    raw_traceback: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_type": self.error_type.value,
            "faulty_module": self.faulty_module,
            "root_cause": self.root_cause,
            "suggested_fix": self.suggested_fix,
            "confidence": round(self.confidence, 2),
        }


class DebugAgent:
    """
    Classifies and explains execution failures so AutoFixAgent can generate
    a targeted patch instead of regenerating entire modules blindly.
    """

    _TRACEBACK_FRAME = re.compile(
        r'File "(?P<file>[^"]+)", line (?P<line>\d+), in (?P<func>\S+)'
    )
    _FINAL_EXCEPTION = re.compile(
        r"^(?P<exc_type>[A-Za-z_][A-Za-z0-9_]*)(Error)?:\s*(?P<msg>.*)$", re.MULTILINE
    )
    _MISSING_MODULE = re.compile(
        r"No module named ['\"](?P<module>[\w\.]+)['\"]"
    )
    _DEPENDENCY_VERSION_CONFLICT = re.compile(
        r"(?i)(requires|incompatible|version conflict|ResolutionImpossible)"
    )

    def analyze_traceback(self, traceback_text: str) -> DebugAnalysis:
        """
        Entry point: parses a raw stderr/traceback string and returns a full
        DebugAnalysis combining error classification, module localization,
        and a suggested fix.
        """
        if not traceback_text or not traceback_text.strip():
            logger.warning("Empty traceback supplied to analyze_traceback()")
            return DebugAnalysis(
                error_type=ErrorType.UNKNOWN,
                faulty_module="unknown",
                root_cause="No traceback content was provided.",
                suggested_fix="Re-run execution with stderr capture enabled.",
                confidence=0.0,
            )

        error_type = self.detect_error_type(traceback_text)
        faulty_module = self.locate_faulty_module(traceback_text)
        root_cause = self._extract_root_cause(traceback_text, error_type)
        suggested_fix = self.suggest_fix(error_type, root_cause, faulty_module)
        confidence = self._estimate_confidence(traceback_text, error_type)

        analysis = DebugAnalysis(
            error_type=error_type,
            faulty_module=faulty_module,
            root_cause=root_cause,
            suggested_fix=suggested_fix,
            confidence=confidence,
            raw_traceback=traceback_text,
        )
        logger.info(
            "Diagnosed failure: type=%s module=%s confidence=%.2f",
            error_type.value,
            faulty_module,
            confidence,
        )
        return analysis

    def detect_error_type(self, traceback_text: str) -> ErrorType:
        if self._MISSING_MODULE.search(traceback_text):
            return ErrorType.MODULE_NOT_FOUND
        if self._DEPENDENCY_VERSION_CONFLICT.search(traceback_text):
            return ErrorType.DEPENDENCY_CONFLICT
        if "ModuleNotFoundError" in traceback_text:
            return ErrorType.MODULE_NOT_FOUND
        if "ImportError" in traceback_text:
            return ErrorType.IMPORT_ERROR
        if "SyntaxError" in traceback_text or "IndentationError" in traceback_text:
            return ErrorType.SYNTAX_ERROR
        if "NameError" in traceback_text:
            return ErrorType.NAME_ERROR
        if "TypeError" in traceback_text:
            return ErrorType.TYPE_ERROR
        if "AttributeError" in traceback_text:
            return ErrorType.ATTRIBUTE_ERROR
        if "KeyError" in traceback_text:
            return ErrorType.KEY_ERROR
        if "IndexError" in traceback_text:
            return ErrorType.INDEX_ERROR
        if "ValueError" in traceback_text:
            return ErrorType.VALUE_ERROR
        if "AssertionError" in traceback_text:
            return ErrorType.ASSERTION_ERROR
        if "TimeoutError" in traceback_text or "timed out" in traceback_text.lower():
            return ErrorType.TIMEOUT_ERROR
        if "RuntimeError" in traceback_text:
            return ErrorType.RUNTIME_ERROR
        if "pip" in traceback_text.lower() and (
            "not found" in traceback_text.lower() or "no matching distribution" in traceback_text.lower()
        ):
            return ErrorType.MISSING_PACKAGE
        return ErrorType.UNKNOWN

    def locate_faulty_module(self, traceback_text: str) -> str:
        """
        Walks all 'File "...", line N, in func' frames and returns the
        deepest frame that points into project source (i.e. not site-packages
        or the standard library), since that is almost always the agent's
        own generated code.
        """
        frames = self._TRACEBACK_FRAME.findall(traceback_text)
        if not frames:
            return "unknown"

        project_frames = [
            f for f in frames
            if "site-packages" not in f[0] and "lib/python" not in f[0].replace("\\", "/")
        ]
        target = project_frames[-1] if project_frames else frames[-1]
        file_path = target[0].replace("\\", "/")
        return file_path.rsplit("/", 1)[-1]

    def suggest_fix(self, error_type: ErrorType, root_cause: str, faulty_module: str) -> str:
        suggestions = {
            ErrorType.MODULE_NOT_FOUND: (
                f"Add the missing dependency to requirements.txt and install it, "
                f"or correct the import path in {faulty_module}."
            ),
            ErrorType.IMPORT_ERROR: (
                f"Verify the imported symbol exists and is exported from its source module; "
                f"check for circular imports involving {faulty_module}."
            ),
            ErrorType.SYNTAX_ERROR: (
                f"Re-generate {faulty_module} with corrected syntax around the reported line; "
                f"run an AST parse check before re-execution."
            ),
            ErrorType.NAME_ERROR: (
                f"Define or import the missing identifier referenced in {faulty_module} "
                f"before its first use."
            ),
            ErrorType.TYPE_ERROR: (
                f"Inspect the function signature and call site in {faulty_module}; "
                f"argument types or counts likely mismatch."
            ),
            ErrorType.ATTRIBUTE_ERROR: (
                f"Confirm the object in {faulty_module} is the expected type/instance "
                f"and that the attribute/method exists on it."
            ),
            ErrorType.KEY_ERROR: (
                f"Add a guarded .get()/default or validate dictionary keys before access in {faulty_module}."
            ),
            ErrorType.INDEX_ERROR: (
                f"Add bounds checking before list/sequence indexing in {faulty_module}."
            ),
            ErrorType.VALUE_ERROR: (
                f"Validate input values before the failing operation in {faulty_module}."
            ),
            ErrorType.ASSERTION_ERROR: (
                f"Review the failing assertion in {faulty_module}; "
                f"either the implementation or the test expectation is incorrect."
            ),
            ErrorType.TIMEOUT_ERROR: (
                f"Optimize or bound the long-running operation in {faulty_module}, "
                f"or increase the execution timeout if behavior is expected."
            ),
            ErrorType.DEPENDENCY_CONFLICT: (
                "Pin compatible package versions in requirements.txt and reinstall "
                "in a clean virtual environment."
            ),
            ErrorType.MISSING_PACKAGE: (
                "Add the missing package to requirements.txt and run pip install."
            ),
            ErrorType.RUNTIME_ERROR: (
                f"Inspect runtime state and control flow in {faulty_module}; "
                f"root cause: {root_cause}"
            ),
            ErrorType.UNKNOWN: (
                "Insufficient signal to auto-diagnose; escalate for manual review "
                "or request additional logging."
            ),
        }
        return suggestions.get(error_type, suggestions[ErrorType.UNKNOWN])

    def _extract_root_cause(self, traceback_text: str, error_type: ErrorType) -> str:
        if error_type == ErrorType.MODULE_NOT_FOUND:
            match = self._MISSING_MODULE.search(traceback_text)
            if match:
                return f"Missing module: '{match.group('module')}'"

        matches = list(self._FINAL_EXCEPTION.finditer(traceback_text))
        if matches:
            last = matches[-1]
            msg = last.group("msg").strip()
            return msg if msg else f"{error_type.value} raised with no message"

        last_line = traceback_text.strip().splitlines()[-1] if traceback_text.strip() else ""
        return last_line or "Unable to extract a specific root cause from traceback."

    def _estimate_confidence(self, traceback_text: str, error_type: ErrorType) -> float:
        if error_type == ErrorType.UNKNOWN:
            return 0.2
        frame_count = len(self._TRACEBACK_FRAME.findall(traceback_text))
        if frame_count == 0:
            return 0.4
        if frame_count >= 2:
            return 0.9
        return 0.65