"""
security_agent.py

SecurityAgent: performs static security analysis on AI-generated code,
mimicking checks performed by tools such as bandit/semgrep and basic
OWASP guidance. Pattern-based detection is intentionally conservative
(prefers false positives over missed criticals) since output feeds an
autonomous repair loop.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger("ai_engineering_copilot.agents.security_agent")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class SecurityIssue:
    rule_id: str
    severity: Severity
    file: str
    line: int
    message: str
    snippet: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "severity": self.severity.value,
            "file": self.file,
            "line": self.line,
            "message": self.message,
            "snippet": self.snippet,
        }


@dataclass
class _Rule:
    rule_id: str
    pattern: re.Pattern
    severity: Severity
    message: str


class SecurityAgent:
    """
    Scans a project directory (or in-memory source map) for common
    vulnerability classes: hardcoded secrets, injection risk, insecure
    auth/JWT handling, and unsafe shell/eval usage.
    """

    SECRET_RULES: list[_Rule] = [
        _Rule(
            "SEC-SECRET-001",
            re.compile(r"""(?i)(api[_-]?key|secret[_-]?key|access[_-]?token)\s*=\s*["'][A-Za-z0-9_\-/+=]{8,}["']"""),
            Severity.CRITICAL,
            "Hardcoded API key / secret / access token detected.",
        ),
        _Rule(
            "SEC-SECRET-002",
            re.compile(r"""(?i)password\s*=\s*["'][^"'\s]{4,}["']"""),
            Severity.CRITICAL,
            "Hardcoded password literal detected.",
        ),
        _Rule(
            "SEC-SECRET-003",
            re.compile(r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----"),
            Severity.CRITICAL,
            "Embedded private key material detected.",
        ),
    ]

    INJECTION_RULES: list[_Rule] = [
        _Rule(
            "SEC-INJ-001",
            re.compile(r"""(?i)(execute|cursor\.execute)\s*\(\s*f?["'].*(%s|\{.*\}|\+)"""),
            Severity.HIGH,
            "Potential SQL injection via string formatting/concatenation in query execution.",
        ),
        _Rule(
            "SEC-INJ-002",
            re.compile(r"""(?i)\.format\(.*\)\s*\)?\s*#?\s*$"""),
            Severity.LOW,
            "String .format() used near query construction; verify parameterization.",
        ),
        _Rule(
            "SEC-INJ-003",
            re.compile(r"""os\.system\(|subprocess\.(Popen|call|run)\([^)]*shell\s*=\s*True"""),
            Severity.HIGH,
            "Shell execution with shell=True or os.system; risk of command injection.",
        ),
        _Rule(
            "SEC-INJ-004",
            re.compile(r"""\beval\(|\bexec\("""),
            Severity.CRITICAL,
            "Use of eval()/exec() on potentially untrusted input.",
        ),
    ]

    AUTH_RULES: list[_Rule] = [
        _Rule(
            "SEC-AUTH-001",
            re.compile(r"""(?i)jwt\.(encode|decode)\([^)]*algorithm\s*=\s*["']none["']"""),
            Severity.CRITICAL,
            "JWT configured with 'none' algorithm — signature verification bypass.",
        ),
        _Rule(
            "SEC-AUTH-002",
            re.compile(r"""(?i)jwt\.decode\([^)]*verify\s*=\s*False"""),
            Severity.CRITICAL,
            "JWT decoding with verify=False disables signature validation.",
        ),
        _Rule(
            "SEC-AUTH-003",
            re.compile(r"""(?i)@app\.route\(.*\)\s*\n\s*def\s+\w+\([^)]*\):(?!\s*\n\s*#\s*auth)"""),
            Severity.MEDIUM,
            "Route handler defined without visible authentication decorator/check nearby.",
        ),
        _Rule(
            "SEC-AUTH-004",
            re.compile(r"""(?i)md5\(|hashlib\.md5"""),
            Severity.MEDIUM,
            "MD5 used for hashing — cryptographically broken for security-sensitive use.",
        ),
    ]

    def __init__(self, project_root: str | Path | None = None) -> None:
        self.project_root = Path(project_root).resolve() if project_root else None
        self._issues: list[SecurityIssue] = []
        logger.info("SecurityAgent initialized (project_root=%s)", self.project_root)

    def scan_project(self, files: dict[str, str] | None = None) -> dict[str, Any]:
        """
        Scans either an explicit {relative_path: source_code} mapping (preferred
        for in-memory generated code) or, if not provided, walks project_root
        for *.py files on disk.
        """
        self._issues = []
        source_map = files if files is not None else self._load_files_from_disk()

        if not source_map:
            logger.warning("No source files provided to scan_project(); returning empty report.")
            return self.generate_security_report()

        for filepath, content in source_map.items():
            self._issues.extend(self.detect_hardcoded_secrets(filepath, content))
            self._issues.extend(self.detect_injection_risk(filepath, content))
            self._issues.extend(self.detect_insecure_auth(filepath, content))

        logger.info("Security scan complete: %d issue(s) found across %d file(s)", len(self._issues), len(source_map))
        return self.generate_security_report()

    def detect_hardcoded_secrets(self, filepath: str, content: str) -> list[SecurityIssue]:
        return self._apply_rules(filepath, content, self.SECRET_RULES)

    def detect_injection_risk(self, filepath: str, content: str) -> list[SecurityIssue]:
        return self._apply_rules(filepath, content, self.INJECTION_RULES)

    def detect_insecure_auth(self, filepath: str, content: str) -> list[SecurityIssue]:
        return self._apply_rules(filepath, content, self.AUTH_RULES)

    def generate_security_report(self) -> dict[str, Any]:
        counts = {sev: 0 for sev in Severity}
        for issue in self._issues:
            counts[issue.severity] += 1

        return {
            "critical": counts[Severity.CRITICAL],
            "high": counts[Severity.HIGH],
            "medium": counts[Severity.MEDIUM],
            "low": counts[Severity.LOW],
            "issues": [issue.to_dict() for issue in self._issues],
        }

    def _apply_rules(self, filepath: str, content: str, rules: Iterable[_Rule]) -> list[SecurityIssue]:
        found: list[SecurityIssue] = []
        lines = content.splitlines()
        for rule in rules:
            for match in rule.pattern.finditer(content):
                line_no = content[: match.start()].count("\n") + 1
                snippet = lines[line_no - 1].strip() if 0 < line_no <= len(lines) else ""
                found.append(
                    SecurityIssue(
                        rule_id=rule.rule_id,
                        severity=rule.severity,
                        file=filepath,
                        line=line_no,
                        message=rule.message,
                        snippet=snippet[:200],
                    )
                )
        return found

    def _load_files_from_disk(self) -> dict[str, str]:
        if self.project_root is None or not self.project_root.exists():
            return {}
        source_map: dict[str, str] = {}
        for py_file in self.project_root.rglob("*.py"):
            if any(part in {".venv", "venv", "__pycache__", "node_modules"} for part in py_file.parts):
                continue
            try:
                source_map[str(py_file.relative_to(self.project_root))] = py_file.read_text(
                    encoding="utf-8", errors="replace"
                )
            except OSError as exc:
                logger.warning("Could not read %s: %s", py_file, exc)
        return source_map