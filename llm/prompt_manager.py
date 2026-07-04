"""
prompt_manager.py

PromptManager: centralized prompt engineering for every agent in the
pipeline. Keeps prompt structure consistent, versioned, and optimized for
autonomous coding agents (explicit output-format contracts, no chit-chat,
deterministic structure the rest of the system can parse reliably).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from textwrap import dedent

logger = logging.getLogger("ai_engineering_copilot.llm.prompt_manager")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

_PROMPT_VERSION = "1.0.0"


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    version: str
    body: str


class PromptManager:
    """
    Builds fully-formed prompt strings for each stage of the pipeline.
    All templates enforce: explicit role framing, explicit constraints,
    and an explicit, parseable output contract (raw code only, or JSON only).
    """

    def __init__(self, prompt_version: str = _PROMPT_VERSION) -> None:
        self.prompt_version = prompt_version

    # ------------------------------------------------------------------
    # Code generation
    # ------------------------------------------------------------------
    def build_code_prompt(
        self,
        task_description: str,
        target_file: str,
        language: str = "python",
        constraints: list[str] | None = None,
        context_files: dict[str, str] | None = None,
    ) -> str:
        constraints = constraints or []
        constraints_block = "\n".join(f"- {c}" for c in constraints) or "- None beyond the global rules below."
        context_block = self._render_context_files(context_files)

        return dedent(f"""
            ROLE: You are a senior {language} software engineer operating as an autonomous
            coding agent inside a multi-agent software engineering pipeline.

            TASK:
            Generate the complete, production-ready contents of the file `{target_file}`.

            TASK DESCRIPTION:
            {task_description}

            ADDITIONAL CONSTRAINTS:
            {constraints_block}

            {context_block}

            GLOBAL RULES:
            - Do not include placeholders, TODO comments, or pseudocode.
            - Do not simplify logic for brevity; implement it fully.
            - Include type hints, docstrings, and structured exception handling.
            - The code must be immediately runnable with no missing pieces.
            - Do not include explanations, markdown headers, or commentary.

            OUTPUT CONTRACT:
            Return ONLY the raw source code for `{target_file}`.
            Do not wrap it in markdown code fences. Do not add any text before or after the code.
        """).strip()

    # ------------------------------------------------------------------
    # Code review / critic
    # ------------------------------------------------------------------
    def build_review_prompt(
        self,
        target_file: str,
        source_code: str,
        review_focus: list[str] | None = None,
    ) -> str:
        review_focus = review_focus or [
            "correctness",
            "edge case handling",
            "readability",
            "architectural consistency",
            "performance",
        ]
        focus_block = "\n".join(f"- {f}" for f in review_focus)

        return dedent(f"""
            ROLE: You are a principal engineer performing a rigorous code review as part of
            an autonomous software engineering pipeline.

            FILE UNDER REVIEW: {target_file}

            SOURCE CODE:
            ```
            {source_code}
            ```

            REVIEW FOCUS AREAS:
            {focus_block}

            OUTPUT CONTRACT:
            Return ONLY a JSON object with this exact schema, no commentary outside the JSON:
            {{
              "approved": boolean,
              "issues": [
                {{"severity": "critical|major|minor", "line": integer, "description": string, "recommendation": string}}
              ],
              "summary": string
            }}
        """).strip()

    # ------------------------------------------------------------------
    # Debugging
    # ------------------------------------------------------------------
    def build_debug_prompt(
        self,
        target_file: str,
        source_code: str,
        error_type: str,
        traceback_text: str,
    ) -> str:
        return dedent(f"""
            ROLE: You are an expert debugging agent specializing in root-cause analysis
            for autonomously generated software.

            FAILING FILE: {target_file}
            DETECTED ERROR TYPE: {error_type}

            SOURCE CODE:
            ```
            {source_code}
            ```

            CAPTURED TRACEBACK / STDERR:
            ```
            {traceback_text}
            ```

            TASK:
            Identify the precise root cause of this failure and describe what must change
            to resolve it. Do not rewrite the file yet; this is a diagnostic step only.

            OUTPUT CONTRACT:
            Return ONLY a JSON object with this exact schema:
            {{
              "root_cause": string,
              "faulty_lines": [integer],
              "explanation": string,
              "fix_strategy": string
            }}
        """).strip()

    # ------------------------------------------------------------------
    # Security audit
    # ------------------------------------------------------------------
    def build_security_prompt(
        self,
        target_file: str,
        source_code: str,
        static_findings: list[dict] | None = None,
    ) -> str:
        static_findings = static_findings or []
        findings_block = (
            "\n".join(
                f"- [{f.get('severity', 'unknown')}] {f.get('message', '')} (line {f.get('line', '?')})"
                for f in static_findings
            )
            or "- No static findings were pre-detected; perform a full manual review."
        )

        return dedent(f"""
            ROLE: You are an application security engineer performing a manual audit,
            equivalent in rigor to bandit/semgrep plus OWASP Top 10 review, for an
            autonomous code-generation pipeline.

            FILE UNDER AUDIT: {target_file}

            SOURCE CODE:
            ```
            {source_code}
            ```

            STATIC ANALYZER FINDINGS (verify, refine, and add to these):
            {findings_block}

            CHECK SPECIFICALLY FOR:
            - Hardcoded secrets, API keys, passwords, or private keys
            - SQL / command / code injection vectors
            - Insecure authentication or JWT handling (e.g. alg=none, verify=False)
            - Unsafe shell execution (os.system, shell=True, eval/exec on untrusted input)
            - Missing authorization checks on sensitive endpoints
            - Use of broken cryptographic primitives (e.g. MD5, SHA1 for security purposes)

            OUTPUT CONTRACT:
            Return ONLY a JSON object with this exact schema:
            {{
              "issues": [
                {{"severity": "critical|high|medium|low", "line": integer, "category": string, "description": string, "remediation": string}}
              ],
              "overall_risk": "critical|high|medium|low|none"
            }}
        """).strip()

    # ------------------------------------------------------------------
    # Auto-fix
    # ------------------------------------------------------------------
    def build_fix_prompt(
        self,
        file_name: str,
        original_source: str,
        error_type: str,
        root_cause: str,
        suggested_fix: str,
    ) -> str:
        return dedent(f"""
            ROLE: You are an autonomous auto-fix agent. You repair broken or vulnerable
            source files with minimal, surgical, correct changes while preserving all
            unrelated functionality.

            FILE TO FIX: {file_name}

            CURRENT SOURCE:
            ```
            {original_source if original_source.strip() else "# (file currently empty or not yet generated)"}
            ```

            DIAGNOSED ERROR TYPE: {error_type}
            ROOT CAUSE: {root_cause}
            SUGGESTED FIX STRATEGY: {suggested_fix}

            TASK:
            Produce the complete corrected contents of `{file_name}`. The result must:
            - Resolve the diagnosed root cause completely.
            - Preserve all unrelated existing functionality and public interfaces.
            - Contain no placeholders, TODOs, or pseudocode.
            - Be fully runnable Python 3.11+ with type hints and exception handling intact.

            OUTPUT CONTRACT:
            Return ONLY the complete raw source code for `{file_name}`.
            Do not wrap it in markdown code fences. Do not add any text before or after the code.
        """).strip()

    # ------------------------------------------------------------------
    # Architecture review
    # ------------------------------------------------------------------
    def build_architecture_prompt(
        self,
        project_summary: str,
        file_manifest: list[str],
        concerns: list[str] | None = None,
    ) -> str:
        concerns = concerns or []
        manifest_block = "\n".join(f"- {f}" for f in file_manifest)
        concerns_block = "\n".join(f"- {c}" for c in concerns) or "- General architectural soundness."

        return dedent(f"""
            ROLE: You are a principal software architect reviewing the design of an
            autonomous multi-agent software engineering system.

            PROJECT SUMMARY:
            {project_summary}

            CURRENT FILE MANIFEST:
            {manifest_block}

            SPECIFIC CONCERNS TO ADDRESS:
            {concerns_block}

            OUTPUT CONTRACT:
            Return ONLY a JSON object with this exact schema:
            {{
              "architecture_score": number,
              "strengths": [string],
              "risks": [string],
              "recommendations": [string]
            }}
        """).strip()

    @staticmethod
    def _render_context_files(context_files: dict[str, str] | None) -> str:
        if not context_files:
            return ""
        rendered = "\n\n".join(
            f"--- {name} ---\n```\n{content}\n```" for name, content in context_files.items()
        )
        return f"RELEVANT EXISTING PROJECT FILES (for context only, do not regenerate these):\n\n{rendered}\n"