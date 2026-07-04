"""
critic_agent.py

CriticAgent inspects code artifacts produced by CodeAgent and performs a
rule-based architecture + security review, surfacing concrete, actionable
issues rather than generic advice.

The rule engine is deliberately pluggable: each check is a small, named
function registered in `self._rules`, returning zero or more `Issue`
instances. New checks can be added without touching the review loop.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from agents.code_agent import CodeGenerationResult

logger = logging.getLogger("copilot.critic_agent")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


SEVERITY_ORDER = ("LOW", "MEDIUM", "HIGH", "CRITICAL")


class CriticReviewError(Exception):
    """Raised when the critic cannot complete a review."""


@dataclass
class Issue:
    """A single finding from the review engine."""

    file: str
    rule: str
    message: str
    severity: str  # one of SEVERITY_ORDER

    def __post_init__(self) -> None:
        if self.severity not in SEVERITY_ORDER:
            raise ValueError(f"Invalid severity: {self.severity}")


@dataclass
class CriticReview:
    """Typed output of CriticAgent.review()."""

    issues_found: list[dict[str, str]]
    severity: str
    improvements: list[str] = field(default_factory=list)


RuleFn = Callable[[str, str], list[Issue]]


class CriticAgent:
    """
    Reviews generated code artifacts for architecture and security issues.

    Usage:
        critic = CriticAgent()
        review = await critic.review(code_result)
    """

    def __init__(self) -> None:
        self._rules: list[RuleFn] = [
            self._check_error_handling,
            self._check_jwt_security,
            self._check_password_hashing,
            self._check_input_validation,
            self._check_hardcoded_secrets,
            self._check_cors_wildcard,
            self._check_bare_except,
        ]
        logger.info("CriticAgent initialized with %d rules", len(self._rules))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    async def review(self, code_result: "CodeGenerationResult") -> CriticReview:
        if not code_result or not code_result.files:
            raise CriticReviewError("No code artifacts supplied for review")

        all_issues: list[Issue] = []
        for filename, content in code_result.files.items():
            for rule in self._rules:
                try:
                    found = rule(filename, content)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Rule %s raised on %s: %s", rule.__name__, filename, exc)
                    continue
                all_issues.extend(found)

        logger.info("Review complete: %d issue(s) found", len(all_issues))

        overall_severity = self._aggregate_severity(all_issues)
        improvements = self._derive_improvements(all_issues)

        return CriticReview(
            issues_found=[
                {
                    "file": issue.file,
                    "rule": issue.rule,
                    "message": issue.message,
                    "severity": issue.severity,
                }
                for issue in all_issues
            ],
            severity=overall_severity,
            improvements=improvements,
        )

    # ------------------------------------------------------------------
    # Rule implementations
    # ------------------------------------------------------------------
    def _check_error_handling(self, filename: str, content: str) -> list[Issue]:
        issues: list[Issue] = []
        if filename in ("routes.py", "auth.py") and "HTTPException" not in content:
            issues.append(
                Issue(
                    file=filename,
                    rule="error_handling",
                    message="No HTTPException usage found; endpoints may leak unhandled errors",
                    severity="MEDIUM",
                )
            )
        return issues

    def _check_jwt_security(self, filename: str, content: str) -> list[Issue]:
        issues: list[Issue] = []
        if filename != "auth.py" or "jwt" not in content.lower():
            return issues

        if "refresh_token" not in content and "create_refresh_token" not in content:
            issues.append(
                Issue(
                    file=filename,
                    rule="jwt_refresh_missing",
                    message="Refresh token implementation missing; access-token-only design forces frequent re-login or long-lived access tokens",
                    severity="HIGH",
                )
            )
        if re.search(r'ALGORITHM\s*=\s*["\']none["\']', content, re.IGNORECASE):
            issues.append(
                Issue(
                    file=filename,
                    rule="jwt_alg_none",
                    message="JWT algorithm is set to 'none', which disables signature verification entirely",
                    severity="CRITICAL",
                )
            )
        if "ACCESS_TOKEN_EXPIRE" not in content:
            issues.append(
                Issue(
                    file=filename,
                    rule="jwt_no_expiry",
                    message="No explicit access token expiry constant found; tokens may not expire",
                    severity="HIGH",
                )
            )
        return issues

    def _check_password_hashing(self, filename: str, content: str) -> list[Issue]:
        issues: list[Issue] = []
        if filename != "auth.py":
            return issues

        if "password" in content.lower() and "CryptContext" not in content and "bcrypt" not in content.lower():
            issues.append(
                Issue(
                    file=filename,
                    rule="weak_password_hashing",
                    message="Password handling found without a recognized hashing library (e.g. passlib/bcrypt); passwords may be stored insecurely",
                    severity="CRITICAL",
                )
            )
        if re.search(r"md5|sha1\(", content, re.IGNORECASE):
            issues.append(
                Issue(
                    file=filename,
                    rule="insecure_hash_algorithm",
                    message="MD5/SHA1 detected; these are unsuitable for password hashing",
                    severity="CRITICAL",
                )
            )
        return issues

    def _check_input_validation(self, filename: str, content: str) -> list[Issue]:
        issues: list[Issue] = []
        if filename == "models.py" and "BaseModel" not in content:
            issues.append(
                Issue(
                    file=filename,
                    rule="missing_validation_schema",
                    message="No pydantic BaseModel found; request/response payloads are not validated",
                    severity="HIGH",
                )
            )
        return issues

    def _check_hardcoded_secrets(self, filename: str, content: str) -> list[Issue]:
        issues: list[Issue] = []
        pattern = re.compile(r'(SECRET_KEY|PASSWORD|API_KEY)\s*=\s*["\'][^"\']{3,}["\']')
        for match in pattern.finditer(content):
            issues.append(
                Issue(
                    file=filename,
                    rule="hardcoded_secret",
                    message=f"Possible hardcoded secret literal for {match.group(1)}; load from environment/secret manager instead",
                    severity="CRITICAL",
                )
            )
        return issues

    def _check_cors_wildcard(self, filename: str, content: str) -> list[Issue]:
        issues: list[Issue] = []
        if filename == "main.py" and re.search(r'allow_origins\s*=\s*\[\s*["\']\*["\']', content):
            issues.append(
                Issue(
                    file=filename,
                    rule="cors_wildcard",
                    message="CORS allow_origins is wildcarded ('*'); restrict to known origins in production",
                    severity="MEDIUM",
                )
            )
        return issues

    def _check_bare_except(self, filename: str, content: str) -> list[Issue]:
        issues: list[Issue] = []
        if re.search(r"except\s*:\s*\n", content):
            issues.append(
                Issue(
                    file=filename,
                    rule="bare_except",
                    message="Bare 'except:' clause found; catch specific exceptions to avoid masking bugs",
                    severity="LOW",
                )
            )
        return issues

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------
    def _aggregate_severity(self, issues: list[Issue]) -> str:
        if not issues:
            return "NONE"
        highest_index = max(SEVERITY_ORDER.index(issue.severity) for issue in issues)
        return SEVERITY_ORDER[highest_index]

    def _derive_improvements(self, issues: list[Issue]) -> list[str]:
        seen: set[str] = set()
        improvements: list[str] = []
        for issue in issues:
            if issue.rule in seen:
                continue
            seen.add(issue.rule)
            improvements.append(f"[{issue.severity}] {issue.file}: {issue.message}")
        return improvements