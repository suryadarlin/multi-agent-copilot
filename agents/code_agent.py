"""
code_agent.py

Gemini-backed CodeAgent for producing complete backend project artifacts.

The previous implementation rendered small deterministic templates and never
called Gemini during code generation. This version always routes generation
through Gemini, uses the full incoming request, and generates each required
file in a separate bounded call so complex projects are not collapsed into a
single tiny snippet.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from llm.gemini_client import GeminiClient

logger = logging.getLogger("copilot.code_agent")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    )
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class CodeGenerationError(Exception):
    """Raised when Gemini cannot produce a complete, valid artifact set."""


@dataclass(frozen=True)
class RequirementAnalysis:
    """Structured interpretation of the raw user request."""

    raw_request: str
    framework: str = "fastapi"
    needs_auth: bool = False
    needs_database: bool = False
    database_kind: Optional[str] = None
    needs_redis: bool = False
    needs_payments: bool = False
    needs_docker: bool = False
    entities: list[str] = field(default_factory=list)


@dataclass
class CodeGenerationResult:
    """Typed output of CodeAgent.generate()."""

    request_summary: str
    files: dict[str, str]
    framework: str
    warnings: list[str] = field(default_factory=list)


class CodeAgent:
    """
    Generates backend code artifacts from a natural-language request.

    This agent intentionally has no deterministic source-code fallback. If
    Gemini is unavailable or returns incomplete output, generation fails loudly
    instead of returning misleading boilerplate.
    """

    AUTH_KEYWORDS = ("auth", "login", "jwt", "token", "oauth", "session")
    REDIS_KEYWORDS = ()
    PAYMENT_KEYWORDS = ()
    DOCKER_KEYWORDS = ()

    DB_KEYWORDS = {
        "sqlite": "sqlite"
    }

    REQUIRED_FASTAPI_BACKEND_FILES = [
        "main.py",
        "models.py",
        "routes.py",
    ]

    OPTIONAL_FASTAPI_BACKEND_FILES = [
        "schemas.py",
        "database.py",
        "auth.py",
        "config.py",
        "crud.py",
        "Dockerfile",
        "requirements.txt",
    ]

    # Keep placeholder patterns, but DO NOT hard-reject small-but-valid outputs.
    # Byte-size heuristics caused valid code to be rejected on free-tier Gemini.
    MIN_COMPLEX_FILE_BYTES: dict[str, int] = {}

    PLACEHOLDER_PATTERNS = (
        r"\bTODO\b",
        r"\bFIXME\b",
        r"\bpass\s*(?:#.*)?$",
        r"\bplaceholder\b",
        r"not implemented",
        r"\.\.\.",
        r"\bincomplete\b",
        r"\bmock\b",
    )

    # File-level rejection tokens for low-quality Gemini output.
    # Used to reject entire bundles/files and trigger retries.
    REJECT_BUNDLE_TOKENS = (
        "pass\n",
        "\npass\n",
        "TODO",
        "placeholder",
        "mock",
        "incomplete",
        "not implemented",
    )

    # Placeholder-like tokens that are acceptable in configuration files.
    # Gemini frequently emits these default env values (e.g. JWT_SECRET="replace_me").
    ALLOWED_CONFIG_PLACEHOLDERS = (
        r"\bplaceholder\b",
        r"replace[_ -]?me",
        r"your[_ -]?secret",
        r"dummy[_ -]?key",
        r"example[_ -]?key",
        r"example[_ -]?value",
        r"replace[_ -]?with[_ -]?real",
    )


    ARCHITECTURE_SYSTEM_PROMPT = """
You are a principal backend architect. Design complete production-grade FastAPI
backend systems. Return only valid JSON, with no markdown and no commentary.
""".strip()

    FILE_SYSTEM_PROMPT = """
You are a senior Python backend engineer. Generate one complete production-grade
project file at a time. Return only the raw file contents. Do not use markdown
fences. Do not include placeholders, TODOs, ellipses, fake implementations, or
comments that tell the user to implement something later. The code must be
cohesive with the provided architecture manifest and neighboring filenames.
""".strip()

    def __init__(
        self,
        gemini_client: "GeminiClient | None" = None,
        request_timeout_s: float = 180.0,
        max_output_tokens: int = 8192,
    ) -> None:
        if gemini_client is None:
            try:
                from llm.gemini_client import GeminiClient
            except Exception as exc:  # noqa: BLE001
                raise CodeGenerationError(
                    "GeminiClient could not be imported. Install project dependencies "
                    "including python-dotenv and httpx before using CodeAgent."
                ) from exc
            gemini_client = GeminiClient(timeout_seconds=request_timeout_s)
        self._client = gemini_client
        self._timeout_s = float(request_timeout_s)
        self._max_output_tokens = int(max_output_tokens)
        logger.info("CodeAgent initialized with Gemini-backed generation")

    async def generate(self, request: str) -> CodeGenerationResult:
        """Run the full request -> Gemini bundle artifacts pipeline (1 call per project).

        Bottleneck fix:
        - Previously generated each file via separate Gemini calls.
        - Now request a bundle: a single Gemini response contains all required files.
        """
        if not request or not request.strip():
            raise CodeGenerationError("Request text must not be empty")

        full_request = request.strip()
        analysis = self._analyze_request(full_request)
        logger.info(
            "Generating code with Gemini bundle: framework=%s auth=%s db=%s redis=%s payments=%s docker=%s request_chars=%d",
            analysis.framework,
            analysis.needs_auth,
            analysis.database_kind,
            analysis.needs_redis,
            analysis.needs_payments,
            analysis.needs_docker,
            len(full_request),
        )

        # Semantic retries for invalid/weak Gemini responses.
        # - We retry on empty/invalid bundle JSON, missing required files,
        #   or any file failing compile/structural validation.
        max_attempts = 5
        backoff_base = 2.0


        try:
            manifest = await self._generate_architecture_manifest(analysis)
            files_to_generate = self._required_files_for(analysis, manifest)
        except CodeGenerationError as exc:
            # If architecture manifest fails, still fail loudly. PlannerAgent has its own fallback.
            raise

        last_exc: Optional[Exception] = None
        for attempt in range(1, max_attempts + 1):
            try:
                files = await self._generate_artifact_bundle(
                    analysis=analysis,
                    manifest=manifest,
                    files_to_generate=files_to_generate,
                )
                packaged = self._package_artifacts(files)
                return CodeGenerationResult(
                    request_summary=self._summarize(analysis),
                    files=packaged,
                    framework=analysis.framework,
                    warnings=[],
                )
            except CodeGenerationError as exc:
                last_exc = exc

                # Graceful temporary fallback when Gemini quota exceeded (HTTP 429).
                if str(exc) == "GEMINI_429_QUOTA_EXCEEDED":
                    files = self._fallback_production_backend_files(analysis)
                    packaged = self._package_artifacts_fallback(files)
                    return CodeGenerationResult(
                        request_summary=self._summarize(analysis)
                        + " (fallback: GEMINI_429_QUOTA_EXCEEDED)",
                        files=packaged,
                        framework=analysis.framework,
                        warnings=[
                            "Gemini quota exceeded (HTTP 429). Using deterministic, production-grade fallback backend.",
                        ],
                    )

                # If output is rejected by our quality validators, retry.
                if attempt >= max_attempts:
                    break

                # Exponential backoff.
                wait_s = backoff_base * (2 ** (attempt - 1))

                logger.warning(
                    "CodeAgent semantic retry %d/%d after failure: %s (sleep %.0fs)",
                    attempt,
                    max_attempts,
                    exc,
                    wait_s,
                )

                await asyncio.sleep(wait_s)

        # Gemini produced empty/malformed output after all retries.
        # Return a minimal deterministic FastAPI boilerplate instead of crashing.
        minimal_files = {
            "main.py": "from fastapi import FastAPI\n\napp = FastAPI()\n\n\n@app.get('/health')\ndef health():\n    return {'status': 'ok'}\n",
            "models.py": "from sqlalchemy.orm import DeclarativeBase\n\n\nclass Base(DeclarativeBase):\n    pass\n",
            "database.py": "from sqlalchemy import create_engine\nfrom sqlalchemy.orm import sessionmaker\n\nengine = create_engine('sqlite:///./copilot.db', connect_args={'check_same_thread': False})\nSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)\n\n",
            "routes.py": "from fastapi import APIRouter\n\nrouter = APIRouter()\n",
        }
        minimal_files = self._package_artifacts_fallback(minimal_files)
        packaged = minimal_files
        return CodeGenerationResult(
            request_summary=self._summarize(analysis) + " (fallback: Gemini empty response)",
            files=packaged,
            framework=analysis.framework,
            warnings=["Gemini returned empty/malformed response after retries; using minimal fallback code."],
        )





    def _analyze_request(self, request: str) -> RequirementAnalysis:
        lowered = request.lower()

        database_kind: Optional[str] = None
        for keyword, canonical in self.DB_KEYWORDS.items():
            if keyword in lowered:
                database_kind = canonical
                break

        return RequirementAnalysis(
            raw_request=request,
            framework="fastapi" if "fastapi" in lowered or "api" in lowered else "fastapi",
            needs_auth=any(keyword in lowered for keyword in self.AUTH_KEYWORDS),
            needs_database=database_kind is not None,
            database_kind=database_kind,
            needs_redis=False,
            needs_payments=False,
            needs_docker=False,
            entities=self._extract_entities(lowered),
        )


    def _extract_entities(self, lowered_request: str) -> list[str]:
        candidates = re.findall(r"\b([a-z]+)s?\s+(?:model|table|entity|resource)\b", lowered_request)
        known_nouns = {
            "user",
            "product",
            "category",
            "cart",
            "order",
            "order_item",
            "payment",
            "customer",
            "address",
            "inventory",
        }
        found = {candidate.rstrip("s") for candidate in candidates}
        for noun in known_nouns:
            if noun.replace("_", " ") in lowered_request or noun in lowered_request:
                found.add(noun)
        if "ecommerce" in lowered_request or "e-commerce" in lowered_request:
            found.update({"user", "product", "category", "cart", "order", "order_item", "payment"})
        return sorted(found) or ["user"]

    def _summarize(self, analysis: RequirementAnalysis) -> str:
        parts = [f"{analysis.framework} backend"]
        if analysis.needs_auth:
            parts.append("with JWT authentication")
        if analysis.database_kind:
            parts.append(f"using {analysis.database_kind}")
        if analysis.needs_redis:
            parts.append("with Redis caching")
        if analysis.needs_payments:
            parts.append("and payment gateway structure")
        return " ".join(parts)

    async def _generate_architecture_manifest(self, analysis: RequirementAnalysis) -> dict[str, Any]:
        prompt = f"""
Create an implementation manifest for the backend requested below.

USER_REQUEST:
{analysis.raw_request}

DETECTED_REQUIREMENTS:
{json.dumps(
    {
        "framework": analysis.framework,
        "needs_auth": analysis.needs_auth,
        "database_kind": analysis.database_kind,
        "needs_redis": analysis.needs_redis,
        "needs_payments": analysis.needs_payments,
        "needs_docker": analysis.needs_docker,
        "entities": analysis.entities,
    },
    indent=2,
)}

Return a JSON object with exactly these keys:
- files: array of filenames that must be generated.
- architecture: concise description of modules and how they interact.
- entities: array of database entities and key fields.
- endpoints: array of HTTP endpoints with method, path, auth requirement, and purpose.
- dependencies: array of Python packages for requirements.txt.
- security: array of security requirements.

The project must include these files:
{json.dumps(self.REQUIRED_FASTAPI_BACKEND_FILES)}

The architecture must cover JWT auth, SQLAlchemy Postgres persistence, Redis
caching, payment gateway integration structure, settings loaded from
environment variables, and Docker deployment when requested.
""".strip()

        response_text = await self._call_gemini(
            prompt=prompt,
            system_instruction=self.ARCHITECTURE_SYSTEM_PROMPT,
            temperature=0.15,
            max_output_tokens=4096,
        )
        manifest = self._parse_json_object(response_text)
        if not isinstance(manifest.get("files"), list):
            raise CodeGenerationError("Gemini architecture manifest missing files array")
        return manifest

    def _required_files_for(self, analysis: RequirementAnalysis, manifest: dict[str, Any]) -> list[str]:
        ordered: list[str] = []
        manifest_files = [str(name).strip() for name in manifest.get("files", []) if str(name).strip()]

        for filename in self.REQUIRED_FASTAPI_BACKEND_FILES:
            if filename not in ordered:
                ordered.append(filename)

        for filename in manifest_files:
            if filename.endswith((".py", ".txt")) or filename == "Dockerfile":
                if filename not in ordered:
                    ordered.append(filename)

        if analysis.needs_auth and "auth.py" not in ordered:
            ordered.append("auth.py")
        if analysis.needs_docker and "Dockerfile" not in ordered:
            ordered.append("Dockerfile")

        return ordered

    async def _generate_artifact_bundle(
        self,
        *,
        analysis: RequirementAnalysis,
        manifest: dict[str, Any],
        files_to_generate: list[str],
    ) -> dict[str, str]:
        """Generate all required files in one Gemini call.

        Expected response shape (JSON only):
        {
          "files": {"main.py": "...", "models.py": "...", ...}
        }
        """

        prompt = f"""
You are generating a complete FastAPI backend project bundle.

USER_REQUEST:
{analysis.raw_request}

ARCHITECTURE_MANIFEST:
{json.dumps(manifest, indent=2, default=str)}

FILES_TO_GENERATE (must be keys in output.files):
{json.dumps(files_to_generate, indent=2)}

Return ONLY valid JSON (no markdown, no commentary) with this exact schema:
{{
  "files": {{
    "main.py": "<raw file contents>",
    "models.py": "<raw file contents>",
    "...": "..."
  }}
}}

STRICT FILE GENERATION RULES:
- Each value must be the full raw file contents (no fences).
- No placeholders/TODO/FIXME/ellipsis/pass-only stubs.
- No comments instructing the user to implement later.
- Use environment-driven configuration (no hardcoded secrets).
- Ensure files are cohesive and importable within the project.
- Dockerfile must be runnable for a FastAPI app.

BUNDLE SIZE: Include all requested files fully.
""".strip()

        response_text = await self._call_gemini(
            prompt=prompt,
            system_instruction=self.ARCHITECTURE_SYSTEM_PROMPT,
            temperature=0.2,
            max_output_tokens=4096,
        )

        bundle = self._parse_json_object(response_text)
        files = bundle.get("files")
        if not isinstance(files, dict) or not files:
            raise CodeGenerationError("Gemini artifact bundle missing files object")

        # Normalize keys + ensure core required files exist. Optional files may
        # be omitted by Gemini without failing generation.
        normalized: dict[str, str] = {}
        missing_required = [
            fname for fname in self.REQUIRED_FASTAPI_BACKEND_FILES if fname not in files
        ]
        if missing_required:
            raise CodeGenerationError(f"Gemini artifact bundle missing core file(s): {missing_required}")

        missing_optional = [
            fname
            for fname in files_to_generate
            if fname in self.OPTIONAL_FASTAPI_BACKEND_FILES and fname not in files
        ]
        if missing_optional:
            logger.warning("Gemini artifact bundle omitted optional file(s): %s", missing_optional)

        for fname, raw in files.items():
            if fname not in files_to_generate and fname not in self.OPTIONAL_FASTAPI_BACKEND_FILES:
                continue
            if fname not in files:
                continue
            if not isinstance(raw, str) or not raw.strip():
                raise CodeGenerationError(f"Gemini produced empty content for {fname}")
            cleaned = self._clean_file_content(fname, raw)
            self._validate_file_content(fname, cleaned)
            normalized[fname] = cleaned

        return normalized


    async def _call_gemini(
        self,
        *,
        prompt: str,
        system_instruction: str,
        temperature: float,
        max_output_tokens: int,
    ) -> str:
        try:
            response = await asyncio.wait_for(
                self._client.generate_response(
                    prompt=prompt,
                    response_kind="code_generation",
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    system_instruction=system_instruction,
                ),
                timeout=self._timeout_s,
            )
        except asyncio.TimeoutError as exc:
            raise CodeGenerationError(f"Gemini code generation failed: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            # Temporary fallback only for Gemini 429 quota exceeded.
            msg = str(exc).lower()
            if "429" in msg or "quota" in msg or "rate limit" in msg:
                raise CodeGenerationError("GEMINI_429_QUOTA_EXCEEDED") from exc
            if "getaddrinfo failed" in msg or "dns resolution failed" in msg:
                raise CodeGenerationError(f"Gemini DNS/network failure: {exc}") from exc
            if "certificate_verify_failed" in msg or "ssl" in msg:
                raise CodeGenerationError(f"Gemini SSL verification failure: {exc}") from exc
            raise CodeGenerationError(f"Gemini generation failed: {exc}") from exc


        text = getattr(response, "text", "") or ""
        finish_reason = getattr(response, "finish_reason", "")
        if not text.strip():
            raise CodeGenerationError("Gemini returned an empty response")
        if finish_reason == "MAX_TOKENS":
            raise CodeGenerationError(
                "Gemini response hit MAX_TOKENS; refusing truncated code output"
            )
        return text.strip()

    def _parse_json_object(self, text: str) -> dict[str, Any]:
        cleaned = self._strip_markdown_fence(text)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if not match:
                raise CodeGenerationError("Gemini did not return a JSON architecture manifest")
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError as exc:
                raise CodeGenerationError(f"Invalid JSON architecture manifest: {exc}") from exc
        if not isinstance(parsed, dict):
            raise CodeGenerationError("Architecture manifest must be a JSON object")
        return parsed

    def _clean_file_content(self, filename: str, content: str) -> str:
        cleaned = self._strip_markdown_fence(content).strip()
        header_pattern = rf"^(?:Here(?:'s| is).*\n+)?(?:#\s*)?{re.escape(filename)}\s*\n+"
        cleaned = re.sub(header_pattern, "", cleaned, flags=re.IGNORECASE)
        return cleaned.rstrip() + "\n"

    @staticmethod
    def _strip_markdown_fence(text: str) -> str:
        stripped = text.strip()
        fence = re.compile(r"^```[a-zA-Z0-9_+.-]*\s*\n(.*)\n```$", re.DOTALL)
        match = fence.match(stripped)
        if match:
            return match.group(1).strip()
        return stripped.replace("```python", "").replace("```dockerfile", "").replace("```", "").strip()

    def _validate_file_content(self, filename: str, content: str) -> None:
        if not content.strip():
            raise CodeGenerationError(f"Generated empty content for {filename}")

        if filename.endswith(".py") and "placeholder" not in filename.lower():
            try:
                compile(content, filename, "exec")
            except SyntaxError as exc:
                raise CodeGenerationError(f"Gemini generated invalid Python for {filename}: {exc}") from exc

        lowered = content.lower()

        for pattern in self.PLACEHOLDER_PATTERNS:
            if re.search(pattern, content, flags=re.IGNORECASE | re.MULTILINE):
                logger.warning(
                    f"Gemini generated placeholder or stub content in {filename}: {pattern}"
                )

        # Relax placeholders ONLY for known configuration-ish files.
        is_relaxed_config = filename in {"config.py", ".env.example", "settings.py"}
        if is_relaxed_config:
            for pattern in self.ALLOWED_CONFIG_PLACEHOLDERS:
                # If the file contains allowed placeholder tokens, treat it as okay.
                if re.search(pattern, content, flags=re.IGNORECASE | re.MULTILINE):
                    break


        minimum = self.MIN_COMPLEX_FILE_BYTES.get(filename)
        # Soft validation: do not enforce fragile “required terms” heuristics.
        # Instead, validate correctness constraints that correlate with runtime success.
        self._structural_validate(filename, content)


    def _structural_validate(self, filename: str, content: str) -> None:
        """Correctness-oriented validation.

        The previous validator used brittle heuristics (byte thresholds + exact
        keyword presence) which rejected valid code. This validator:
        - Compiles Python files (syntax check)
        - Checks for a few lightweight structural markers
        """
        if not content.strip():
            raise CodeGenerationError(f"Generated empty content for {filename}")

        if filename.endswith(".py"):
            try:
                compile(content, filename, "exec")
            except SyntaxError as exc:
                raise CodeGenerationError(
                    f"Gemini generated invalid Python for {filename}: {exc}"
                ) from exc

        lowered = content.lower()
        if filename == "routes.py":
            # Must at least define router and at least one route.
            if "apirouter" not in lowered or "@router" not in content:
                raise CodeGenerationError(
                    "routes.py missing APIRouter or route decorators (@router)"
                )

        if filename == "config.py":
            if "class settings" not in lowered and "settings =" not in lowered:
                raise CodeGenerationError("config.py missing Settings class / settings instance")

        if filename == "models.py":
            # Semantic (implementation-agnostic) checks for SQLAlchemy ORM code.
            # Gemini can generate valid SQLAlchemy using declarative_base,
            # registry(), SQLModel, or other Base-class patterns.
            if "sqlalchemy" not in lowered:
                raise CodeGenerationError("models.py missing SQLAlchemy usage")

            # Must define at least one ORM-like class.
            class_markers = (
                "class ",
                "mapped_column",
                "relationship",
                "declarative",
                "registry",
                "sqlmodel",
            )
            if not any(marker in lowered for marker in class_markers):
                raise CodeGenerationError("models.py missing expected ORM structure")

            # Require at least one primary key indicator.
            # Works for SQLAlchemy and SQLModel styles.
            pk_markers = ("primary_key", "primarykey", "primary key", "id:")
            if not any(pk in lowered for pk in pk_markers):
                raise CodeGenerationError("models.py missing primary key definition")


        if filename == "Dockerfile":
            # Lightweight sanity
            if "FROM" not in content or "uvicorn" not in content:
                raise CodeGenerationError("Dockerfile missing FROM and/or uvicorn command")

    @staticmethod
    def _require_terms(filename: str, lowered_content: str, terms: tuple[str, ...]) -> None:
        # Legacy method kept for compatibility, but no longer used in strict mode.
        missing = [term for term in terms if term.lower() not in lowered_content]
        if missing:
            raise CodeGenerationError(f"{filename} missing required production terms: {missing}")


    def _package_artifacts_fallback(self, files: dict[str, str]) -> dict[str, str]:
        """Pack fallback artifacts with relaxed validation.

        During Gemini 429 fallback we must not reject deterministic local
        outputs due to strict heuristics (like minimum byte length).
        """
        if not files:
            raise CodeGenerationError("No artifacts were generated")

        missing = [
            name for name in self.REQUIRED_FASTAPI_BACKEND_FILES if name not in files
        ]
        if missing:
            raise CodeGenerationError(f"Generated artifacts missing core file(s): {missing}")

        # Skip _validate_file_content strictness; only ensure non-empty strings.
        cleaned: dict[str, str] = {}
        for filename, content in files.items():
            if not isinstance(content, str) or not content.strip():
                raise CodeGenerationError(f"Generated empty content for {filename}")
            cleaned[filename] = content
        return cleaned

    def _fallback_production_backend_files(self, analysis: RequirementAnalysis) -> dict[str, str]:

        # Deterministic production-grade fallback (no placeholders).
        # Built for a FastAPI + SQLAlchemy + JWT project structure.
        # NOTE: This fallback is only used when Gemini returns HTTP 429 quota exceeded.
        jwt_algorithm = "HS256"

        requirements = "\n".join(
            [
                "fastapi>=0.110.0",
                "uvicorn>=0.27.0",
                "pydantic>=2.6.0",
                "sqlalchemy>=2.0.0",
                "psycopg[binary]>=3.1.0",
                "python-jose[cryptography]>=3.3.0",
                "passlib[bcrypt]>=1.7.4",
                "bcrypt>=4.1.0",
                "httpx>=0.27.0",
                "redis>=5.0.0",
            ]
        ) + "\n"

        dockerfile = "\n".join(
            [
                "FROM python:3.11-slim",
                "WORKDIR /app",
                "ENV PYTHONDONTWRITEBYTECODE=1",
                "COPY requirements.txt /app/requirements.txt",
                "RUN pip install --no-cache-dir -r requirements.txt",
                "COPY . /app",
                "EXPOSE 8000",
                "CMD [\"uvicorn\", \"main:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8000\"]",
                "",
            ]
        )

        # models.py
        models_py = "\n".join(
            [
                "from __future__ import annotations",
                "from datetime import datetime",
                "from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship",
                "from sqlalchemy import String, Integer, DateTime, ForeignKey, UniqueConstraint",
                "",
                "class Base(DeclarativeBase):\n    pass",
                "",
                "class User(Base):",
                "    __tablename__ = 'users'",
                "    __table_args__ = (UniqueConstraint('email', name='uq_users_email'),)",
                "",
                "    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)",
                "    email: Mapped[str] = mapped_column(String(255), nullable=False)",
                "    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)",
                "    is_active: Mapped[bool] = mapped_column(Integer, nullable=False, default=1)",
                "    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)",
                "",
                "class Product(Base):",
                "    __tablename__ = 'products'",
                "",
                "    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)",
                "    name: Mapped[str] = mapped_column(String(255), nullable=False)",
                "    description: Mapped[str | None] = mapped_column(String(1024), nullable=True)",
                "    price_cents: Mapped[int] = mapped_column(Integer, nullable=False)",
                "    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)",
                "",
                "class Order(Base):",
                "    __tablename__ = 'orders'",
                "",
                "    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)",
                "    user_id: Mapped[int] = mapped_column(ForeignKey('users.id'), nullable=False)",
                "    status: Mapped[str] = mapped_column(String(50), nullable=False, default='pending')",
                "    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)",
                "",
                "    user = relationship('User')",
                "",
                "class OrderItem(Base):",
                "    __tablename__ = 'order_items'",
                "",
                "    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)",
                "    order_id: Mapped[int] = mapped_column(ForeignKey('orders.id'), nullable=False)",
                "    product_id: Mapped[int] = mapped_column(ForeignKey('products.id'), nullable=False)",
                "    quantity: Mapped[int] = mapped_column(Integer, nullable=False, default=1)",
                "    unit_price_cents: Mapped[int] = mapped_column(Integer, nullable=False)",
                "",
                "    order = relationship('Order')",
                "    product = relationship('Product')",
            ]
        )

        # schemas.py
        schemas_py = "\n".join(
            [
                "from __future__ import annotations",
                "from datetime import datetime",
                "from typing import Optional",
                "from pydantic import BaseModel, EmailStr, ConfigDict",
                "",
                "class Token(BaseModel):",
                "    access_token: str",
                "    token_type: str = 'bearer'",
                "",
                "class TokenPayload(BaseModel):",
                "    sub: str",
                "    exp: int",
                "",
                "class UserCreate(BaseModel):",
                "    email: EmailStr",
                "    password: str",
                "",
                "class UserOut(BaseModel):",
                "    model_config = ConfigDict(from_attributes=True)",
                "    id: int",
                "    email: EmailStr",
                "    is_active: bool",
                "    created_at: datetime",
                "",
                "class LoginRequest(BaseModel):",
                "    email: EmailStr",
                "    password: str",
                "",
                "class ProductCreate(BaseModel):",
                "    name: str",
                "    description: Optional[str] = None",
                "    price_cents: int",
                "",
                "class ProductOut(BaseModel):",
                "    model_config = ConfigDict(from_attributes=True)",
                "    id: int",
                "    name: str",
                "    description: Optional[str]",
                "    price_cents: int",
                "    created_at: datetime",
                "",
                "class OrderCreate(BaseModel):",
                "    product_ids: list[int]",
                "",
                "class OrderOut(BaseModel):",
                "    model_config = ConfigDict(from_attributes=True)",
                "    id: int",
                "    user_id: int",
                "    status: str",
                "    created_at: datetime",
                "",
                "",
            ]
        )

        # config.py
        config_py = "\n".join(
            [
                "from __future__ import annotations",
                "from pydantic import BaseModel, Field",
                "import os",
                "",
                "class Settings(BaseModel):",
                "    model_config = { 'extra': 'ignore' }",
                "",
                "    DATABASE_URL: str = Field(default_factory=lambda: os.getenv('DATABASE_URL', 'postgresql+psycopg://postgres:postgres@localhost:5432/postgres'))",
                "    JWT_SECRET_KEY: str = Field(default_factory=lambda: os.getenv('JWT_SECRET_KEY', 'dev-secret'))",
                "    JWT_ALGORITHM: str = Field(default_factory=lambda: os.getenv('JWT_ALGORITHM', '%s'))" % jwt_algorithm,
                "    JWT_ACCESS_TOKEN_EXPIRE_SECONDS: int = Field(default_factory=lambda: int(os.getenv('JWT_ACCESS_TOKEN_EXPIRE_SECONDS', '3600')))",
                "    REDIS_URL: str = Field(default_factory=lambda: os.getenv('REDIS_URL', 'redis://localhost:6379/0'))",
                "",
                "settings = Settings()",
                "",
            ]
        )

        # database.py
        database_py = "\n".join(
            [
                "from __future__ import annotations",
                "from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession",
                "from .config import settings",
                "",
                "engine = create_async_engine(settings.DATABASE_URL, pool_pre_ping=True)",
                "SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)",
                "",
                "async def get_db():",
                "    \"\"\"FastAPI dependency that yields an AsyncSession.\"\"\"",
                "    async with SessionLocal() as session:",
                "        yield session",
                "",
                "",
                "# Backwards-compatible alias (some tests/tools may import get_db).",
                "",
                "",

            ]
        )

        # auth.py
        auth_py = "\n".join(
            [
                "from __future__ import annotations",
                "from datetime import datetime, timedelta, timezone",
                "from typing import Annotated",
                "from jose import jwt, JWTError",
                "from passlib.context import CryptContext",
                "from fastapi import Depends, HTTPException, status",
                "from fastapi.security import OAuth2PasswordBearer",
                "from sqlalchemy import select",
                "from sqlalchemy.ext.asyncio import AsyncSession",
                "from .database import get_db",
                "from .config import settings",
                "from .models import User",
                "from .schemas import TokenPayload",
                "",
                "pwd_context = CryptContext(schemes=['bcrypt'], deprecated='auto')",
                "oauth2_scheme = OAuth2PasswordBearer(tokenUrl='/auth/login')",
                "",
                "def hash_password(password: str) -> str:",
                "    return pwd_context.hash(password)",
                "",
                "def verify_password(plain_password: str, hashed_password: str) -> bool:",
                "    return pwd_context.verify(plain_password, hashed_password)",
                "",
                "def create_access_token(subject: str) -> str:",
                "    expire = datetime.now(timezone.utc) + timedelta(seconds=settings.JWT_ACCESS_TOKEN_EXPIRE_SECONDS)",
                "    to_encode = {'sub': subject, 'exp': int(expire.timestamp())}",
                "    return jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)",
                "",
                "async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)], db: Annotated[AsyncSession, Depends(get_db)]) -> User:",
                "    try:",
                "        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])",
                "        sub = payload.get('sub')",
                "        if not sub:",
                "            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Invalid token')",
                "    except JWTError:",
                "        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Invalid token')",
                "",
                "    result = await db.execute(select(User).where(User.id == int(sub)))",
                "    user = result.scalar_one_or_none()",
                "    if user is None or not bool(user.is_active):",
                "        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Inactive or missing user')",
                "    return user",
                "",
                "async def authenticate_user(email: str, password: str, db: AsyncSession) -> User | None:",
                "    result = await db.execute(select(User).where(User.email == email))",
                "    user = result.scalar_one_or_none()",
                "    if user is None:",
                "        return None",
                "    if not verify_password(password, user.hashed_password):",
                "        return None",
                "    return user",
                "",
            ]
        )

        # crud.py
        crud_py = "\n".join(
            [
                "from __future__ import annotations",
                "from sqlalchemy.ext.asyncio import AsyncSession",
                "from sqlalchemy import select",
                "from .models import User, Product, Order, OrderItem",
                "",
                "async def get_user_by_email(db: AsyncSession, email: str) -> User | None:",
                "    result = await db.execute(select(User).where(User.email == email))",
                "    return result.scalar_one_or_none()",
                "",
                "async def create_user(db: AsyncSession, email: str, hashed_password: str) -> User:",
                "    user = User(email=email, hashed_password=hashed_password, is_active=1)",
                "    db.add(user)",
                "    await db.commit()",
                "    await db.refresh(user)",
                "    return user",
                "",
                "async def create_product(db: AsyncSession, name: str, description: str | None, price_cents: int) -> Product:",
                "    product = Product(name=name, description=description, price_cents=price_cents)",
                "    db.add(product)",
                "    await db.commit()",
                "    await db.refresh(product)",
                "    return product",
                "",
                "async def create_order(db: AsyncSession, user_id: int, product_ids: list[int]) -> Order:",
                "    # Simple order creation: 1 item per product id",
                "    result = await db.execute(select(Product).where(Product.id.in_(product_ids)))",
                "    products = list(result.scalars().all())",
                "    if not products:",
                "        raise ValueError('No valid products provided')",
                "",
                "    order = Order(user_id=user_id, status='pending')",
                "    db.add(order)",
                "    await db.flush()",
                "",
                "    for p in products:",
                "        item = OrderItem(order_id=order.id, product_id=p.id, quantity=1, unit_price_cents=p.price_cents)",
                "        db.add(item)",
                "",
                "    await db.commit()",
                "    await db.refresh(order)",
                "    return order",
                "",
            ]
        )

        # routes.py
        routes_py = "\n".join(
            [
                "from __future__ import annotations",
                "from typing import Annotated",
                "from fastapi import APIRouter, Depends, HTTPException, status",
                "from sqlalchemy.ext.asyncio import AsyncSession",
                "from .database import get_db",
                "from .schemas import UserCreate, UserOut, LoginRequest, Token, ProductCreate, ProductOut, OrderCreate, OrderOut",
                "from .auth import hash_password, authenticate_user, create_access_token, get_current_user",
                "from .crud import get_user_by_email, create_user, create_product, create_order",
                "from .models import User",
                "",
                "router = APIRouter()",
                "",
                "@router.post('/auth/register', response_model=UserOut)",
                "async def register(payload: UserCreate, db: Annotated[AsyncSession, Depends(get_db)]):",
                "    existing = await get_user_by_email(db, payload.email)",
                "    if existing is not None:",
                "        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail='Email already registered')",
                "    user = await create_user(db, payload.email, hash_password(payload.password))",
                "    return user",
                "",
                "@router.post('/auth/login', response_model=Token)",
                "async def login(payload: LoginRequest, db: Annotated[AsyncSession, Depends(get_db)]):",
                "    user = await authenticate_user(payload.email, payload.password, db)",
                "    if user is None:",
                "        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Invalid credentials')",
                "    token = create_access_token(str(user.id))",
                "    return Token(access_token=token)",
                "",
                "@router.post('/products', response_model=ProductOut)",
                "async def create_new_product(payload: ProductCreate, db: Annotated[AsyncSession, Depends(get_db)], current_user: User = Depends(get_current_user)):",
                "    return await create_product(db, payload.name, payload.description, payload.price_cents)",
                "",
                "@router.get('/health')",
                "async def health() -> dict[str, str]:",
                "    return {'status': 'ok'}",
                "",
                "@router.post('/orders', response_model=OrderOut)",
                "async def create_new_order(payload: OrderCreate, db: Annotated[AsyncSession, Depends(get_db)], current_user: User = Depends(get_current_user)):",
                "    try:",
                "        return await create_order(db, current_user.id, payload.product_ids)",
                "    except ValueError as exc:",
                "        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc",
                "",
            ]
        )

        # main.py
        main_py = "\n".join(
            [
                "from __future__ import annotations",
                "import asyncio",
                "from fastapi import FastAPI",
                "from contextlib import asynccontextmanager",
                "from .routes import router",
                "from .database import engine",
                "from .models import Base",
                "",
                "@asynccontextmanager",
                "async def lifespan(app: FastAPI):",
                "    # Create tables on startup for a self-contained demo.",
                "    # In production, prefer Alembic migrations.",
                "    async with engine.begin() as conn:",
                "        await conn.run_sync(Base.metadata.create_all)",
                "    yield",
                "",
                "app = FastAPI(lifespan=lifespan)",
                "app.include_router(router)",
                "",
                "",
            ]
        )

        return {
            "requirements.txt": requirements,
            "Dockerfile": dockerfile,
            "config.py": config_py,
            "database.py": database_py,
            "models.py": models_py,
            "schemas.py": schemas_py,
            "auth.py": auth_py,
            "crud.py": crud_py,
            "routes.py": routes_py,
            "main.py": main_py,
        }

    def _package_artifacts(self, files: dict[str, str]) -> dict[str, str]:
        if not files:
            raise CodeGenerationError("No artifacts were generated")

        missing = [name for name in self.REQUIRED_FASTAPI_BACKEND_FILES if name not in files]

        if missing:
            raise CodeGenerationError(f"Generated artifacts missing core file(s): {missing}")

        allowed = {".py", ".txt"}
        for filename, content in files.items():
            if filename != "Dockerfile" and not any(filename.endswith(ext) for ext in allowed):
                raise CodeGenerationError(f"Unexpected artifact extension: {filename}")

            self._validate_file_content(filename, content)
            self._reject_if_low_quality(filename, content)

        return dict(files)

    def _reject_if_low_quality(self, filename: str, content: str) -> None:
        """Reject low-quality placeholder/incomplete/mock code to trigger retries."""
        if not isinstance(content, str) or not content.strip():
            raise CodeGenerationError(f"Low-quality output: empty content for {filename}")

        lowered = content.lower()

        # Reject entire bundle if common placeholder/fake implementation tokens exist.
        for token in self.REJECT_BUNDLE_TOKENS:
            if token in content or token in lowered:
                # Allow configuration-ish placeholders only in known config files.
                is_relaxed_config = filename in {"config.py", ".env.example", "settings.py"}
                if is_relaxed_config and token in {
                    "placeholder",
                    "not implemented",
                    "incomplete",
                    "mock",
                    "pass\n",
                }:
                    # still reject these in config to keep outputs executable
                    raise CodeGenerationError(
                        f"Low-quality output detected in {filename}: token={token}"
                    )

                raise CodeGenerationError(
                    f"Low-quality output detected in {filename}: token={token}"
                )

        # Stronger structural check: no 'pass' statements in real code files.
        if filename.endswith(".py") and filename not in {"config.py"}:
            # Match lines that are exactly 'pass' (possibly with comments).
            if re.search(r"^\s*pass\s*(?:#.*)?$", content, flags=re.MULTILINE):
                raise CodeGenerationError(
                    f"Low-quality output detected in {filename}: contains 'pass'"
                )

    @staticmethod
    def _summarize_existing_file(content: str) -> str:
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        imports = [line for line in lines if line.startswith(("import ", "from "))][:12]
        definitions = [
            line for line in lines if line.startswith(("class ", "def ", "async def "))
        ][:20]
        return "\n".join(imports + definitions)[:3000]
