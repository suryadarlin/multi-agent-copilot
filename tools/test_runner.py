"""
test_runner.py

Automated pytest execution and result parsing for the MCP Tool Layer.
Runs generated test suites in a subprocess with JSON-based result
collection and a regex fallback for terminal output parsing.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("mcp_tools.test_runner")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logger.addHandler(_handler)


@dataclass
class TestFailure:
    __test__ = False

    test_name: str
    message: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TestSummary:
    __test__ = False

    tests_run: int
    passed: int
    failed: int
    errors: int
    skipped: int
    coverage: Optional[float]
    failure_summary: list[TestFailure] = field(default_factory=list)
    raw_stdout: str = ""
    raw_stderr: str = ""
    timed_out: bool = False
    execution_time_ms: int = 0

    def to_dict(self) -> dict:
        result = asdict(self)
        result["failure_summary"] = [f.to_dict() for f in self.failure_summary]
        return result


class TestRunner:
    """
    Executes a pytest suite against a target path, parses results via
    the pytest-json-report plugin when available, and falls back to
    terminal-output regex parsing otherwise.
    """

    __test__ = False

    def __init__(
        self,
        target_path: str,
        timeout_seconds: int = 60,
        python_executable: str = "python",
    ) -> None:
        self.target_path = Path(target_path).resolve()
        if not self.target_path.exists():
            raise ValueError(f"target_path does not exist: {self.target_path}")
        self.timeout_seconds = timeout_seconds
        self.python_executable = python_executable

    def run_pytest(self) -> TestSummary:
        """Runs pytest with coverage and JSON reporting, returning a structured summary."""
        start_time = time.monotonic()

        with tempfile.TemporaryDirectory(prefix="mcp_test_report_") as tmp_dir:
            report_path = Path(tmp_dir) / "report.json"

            command = [
                self.python_executable,
                "-m",
                "pytest",
                str(self.target_path),
                "--json-report",
                f"--json-report-file={report_path}",
                "--cov",
                "--cov-report=term-missing",
                "-q",
            ]

            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                )
                elapsed_ms = int((time.monotonic() - start_time) * 1000)

                if report_path.exists():
                    summary = self.collect_results(report_path, completed.stdout, completed.stderr)
                else:
                    logger.warning("JSON report not produced; falling back to text parsing.")
                    summary = self._parse_text_output(completed.stdout, completed.stderr)

                summary.execution_time_ms = elapsed_ms
                return summary

            except subprocess.TimeoutExpired:
                elapsed_ms = int((time.monotonic() - start_time) * 1000)
                logger.warning("pytest run timed out after %sms", elapsed_ms)
                return TestSummary(
                    tests_run=0,
                    passed=0,
                    failed=0,
                    errors=0,
                    skipped=0,
                    coverage=None,
                    failure_summary=[],
                    raw_stdout="",
                    raw_stderr=f"Test run exceeded timeout of {self.timeout_seconds}s",
                    timed_out=True,
                    execution_time_ms=elapsed_ms,
                )
            except (OSError, FileNotFoundError) as exc:
                elapsed_ms = int((time.monotonic() - start_time) * 1000)
                logger.error("Failed to launch pytest: %s", exc)
                return TestSummary(
                    tests_run=0,
                    passed=0,
                    failed=0,
                    errors=0,
                    skipped=0,
                    coverage=None,
                    failure_summary=[],
                    raw_stdout="",
                    raw_stderr=str(exc),
                    timed_out=False,
                    execution_time_ms=elapsed_ms,
                )

    def collect_results(self, report_path: Path, stdout: str, stderr: str) -> TestSummary:
        """Parses the pytest-json-report output into a structured TestSummary."""
        try:
            raw = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("Failed to parse JSON report: %s", exc)
            return self._parse_text_output(stdout, stderr)

        summary_block = raw.get("summary", {})
        tests_run = summary_block.get("total", 0)
        passed = summary_block.get("passed", 0)
        failed = summary_block.get("failed", 0)
        errors = summary_block.get("error", 0)
        skipped = summary_block.get("skipped", 0)

        failures = self.parse_failures(raw)
        coverage = self.calculate_coverage(stdout)

        return TestSummary(
            tests_run=tests_run,
            passed=passed,
            failed=failed,
            errors=errors,
            skipped=skipped,
            coverage=coverage,
            failure_summary=failures,
            raw_stdout=stdout,
            raw_stderr=stderr,
        )

    def parse_failures(self, raw_report: dict) -> list[TestFailure]:
        """Extracts individual failing test names and messages from a JSON report."""
        failures: list[TestFailure] = []
        for test in raw_report.get("tests", []):
            outcome = test.get("outcome")
            if outcome in ("failed", "error"):
                test_name = test.get("nodeid", "unknown_test")
                call_block = test.get("call", {}) or test.get("setup", {})
                message = call_block.get("longrepr", "No failure detail available.")
                if not isinstance(message, str):
                    message = str(message)
                failures.append(TestFailure(test_name=test_name, message=message[:2000]))
        return failures

    def calculate_coverage(self, stdout: str) -> Optional[float]:
        """Extracts the total coverage percentage from pytest-cov terminal output."""
        match = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+(?:\.\d+)?)%", stdout)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        return None

    def _parse_text_output(self, stdout: str, stderr: str) -> TestSummary:
        """Regex-based fallback parser for plain pytest terminal summaries."""
        passed = failed = errors = skipped = 0

        passed_match = re.search(r"(\d+)\s+passed", stdout)
        failed_match = re.search(r"(\d+)\s+failed", stdout)
        error_match = re.search(r"(\d+)\s+error", stdout)
        skipped_match = re.search(r"(\d+)\s+skipped", stdout)

        if passed_match:
            passed = int(passed_match.group(1))
        if failed_match:
            failed = int(failed_match.group(1))
        if error_match:
            errors = int(error_match.group(1))
        if skipped_match:
            skipped = int(skipped_match.group(1))

        tests_run = passed + failed + errors + skipped

        failure_blocks = re.findall(
            r"FAILED\s+([\w/\.\-:]+)(?:\s+-\s+(.*))?", stdout
        )
        failures = [
            TestFailure(test_name=name, message=(msg or "No detail captured.").strip()[:2000])
            for name, msg in failure_blocks
        ]

        coverage = self.calculate_coverage(stdout)

        return TestSummary(
            tests_run=tests_run,
            passed=passed,
            failed=failed,
            errors=errors,
            skipped=skipped,
            coverage=coverage,
            failure_summary=failures,
            raw_stdout=stdout,
            raw_stderr=stderr,
        )
