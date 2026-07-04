"""
gemini_client.py

GeminiClient: thin, production-grade async wrapper around the Google AI
Studio Gemini API (https://ai.google.dev/gemini-api/docs) used as the
reasoning engine for code generation, debugging, security, and architecture
review prompts across the agent pipeline.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import socket
import ssl
from dataclasses import dataclass
from dotenv import load_dotenv

# Load .env eagerly, but also refresh env at call-time to avoid cases where
# GEMINI_API_KEY is set after import or in a different working context.
load_dotenv()
from enum import Enum
from typing import Any
from urllib.parse import urlparse

try:
    import httpx
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "httpx is required for GeminiClient. Install with `pip install httpx`."
    ) from exc

logger = logging.getLogger("ai_engineering_copilot.llm.gemini_client")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

_DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
_DEFAULT_MODEL = "gemini-3.1-flash-lite"


class GeminiClientError(Exception):
    """Raised for unrecoverable Gemini API failures."""


class ResponseKind(str, Enum):
    CODE_GENERATION = "code_generation"
    CODE_FIX = "code_fix"
    DEBUGGING = "debugging"
    SECURITY = "security"
    ARCHITECTURE_REVIEW = "architecture_review"
    GENERIC = "generic"


@dataclass
class GeminiResponse:
    text: str
    finish_reason: str
    model: str
    raw: dict[str, Any]


class GeminiClient:
    """
    Async client for Gemini's generateContent endpoint, with retry/backoff,
    response validation, and lightweight structured-output extraction
    (stripping markdown code fences, parsing JSON when expected).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = _DEFAULT_MODEL,
        base_url: str = _DEFAULT_BASE_URL,
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
        backoff_base_seconds: float = 1.5,
        ssl_verify: ssl.SSLContext | str | bool | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            logger.warning(
                "GEMINI_API_KEY is not set at GeminiClient init. Will refresh env at call-time."
            )
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_base_seconds = backoff_base_seconds
        # On Windows, httpx defaults to certifi. Antivirus or corporate HTTPS
        # inspection roots often live in the Windows/Python trust store but not
        # in certifi, causing CERTIFICATE_VERIFY_FAILED despite valid local TLS.
        self.ssl_verify = ssl_verify if ssl_verify is not None else ssl.create_default_context()

    async def generate_response(
        self,
        prompt: str,
        response_kind: str | ResponseKind = ResponseKind.GENERIC,
        temperature: float = 0.2,
        max_output_tokens: int = 4096,
        system_instruction: str | None = None,
    ) -> GeminiResponse:
        """
        High-level entry point: sends a prompt and returns a parsed
        GeminiResponse, retrying transient failures automatically.
        """
        # Refresh env once at call-time to avoid import-time ordering issues.
        if not self.api_key:
            load_dotenv()
            self.api_key = os.environ.get("GEMINI_API_KEY")

        if not self.api_key:
            raise GeminiClientError("GEMINI_API_KEY is not configured (missing in env / .env).")

        kind = ResponseKind(response_kind) if isinstance(response_kind, str) else response_kind
        logger.info("Generating response (kind=%s, model=%s)", kind.value, self.model)

        payload = self._build_payload(
            prompt=prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            system_instruction=system_instruction,
        )

        raw = await self.retry_on_failure(self.send_prompt, payload)
        return self._parse_response(raw)

    async def send_prompt(self, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Performs a single HTTP call to the Gemini generateContent endpoint.
        Raises GeminiClientError on non-2xx responses.
        """
        url = f"{self.base_url}/models/{self.model}:generateContent"
        headers = {"Content-Type": "application/json", "x-goog-api-key": self.api_key or ""}

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds, verify=self.ssl_verify) as client:
                response = await client.post(url, headers=headers, json=payload)
        except httpx.ConnectError as exc:
            message = str(exc)
            if "getaddrinfo failed" in message.lower():
                raise GeminiClientError(self._format_dns_failure(url, exc)) from exc
            raise

        if response.status_code >= 400:
            logger.error("Gemini API error %s: %s", response.status_code, response.text[:500])
            raise GeminiClientError(
                f"Gemini API request failed with status {response.status_code}: {response.text[:500]}"
            )

        try:
            return response.json()
        except json.JSONDecodeError as exc:
            raise GeminiClientError(f"Gemini API returned non-JSON response: {exc}") from exc

    async def retry_on_failure(self, func, *args: Any, **kwargs: Any) -> Any:
        """
        Generic async retry wrapper with exponential backoff. Retries on
        GeminiClientError and transient network exceptions; does not retry
        on configuration errors (missing API key).
        """
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except (GeminiClientError, httpx.TransportError, httpx.TimeoutException) as exc:
                last_exc = exc
                wait_time = self.backoff_base_seconds * (2 ** (attempt - 1))
                logger.warning(
                    "Gemini call failed (attempt %d/%d): %s. Retrying in %.1fs",
                    attempt,
                    self.max_retries,
                    exc,
                    wait_time,
                )
                if attempt < self.max_retries:
                    await asyncio.sleep(wait_time)
        assert last_exc is not None
        raise GeminiClientError(f"Gemini API call failed after {self.max_retries} attempts: {last_exc}") from last_exc

    def _format_dns_failure(self, url: str, exc: Exception) -> str:
        host = urlparse(url).hostname or self.base_url
        try:
            socket.getaddrinfo(host, 443)
        except OSError as dns_exc:
            dns_state = (
                f"Windows/Python DNS still cannot resolve {host}: {dns_exc}. "
                "Check DNS, VPN, proxy, firewall, or antivirus web-shield settings, then retry."
            )
        else:
            dns_state = (
                f"{host} resolves now, so the getaddrinfo failure was transient "
                "or caused by a temporary network/DNS state during the request."
            )

        return (
            f"DNS resolution failed for Gemini endpoint {host}: {exc}. "
            f"{dns_state}"
        )

    def validate_response(self, response: GeminiResponse, expect: str = "text") -> str:
        """
        Validates and post-processes a GeminiResponse based on the expected
        output shape:
          - expect="text": returns trimmed raw text
          - expect="code": strips markdown fences, returns raw source
          - expect="json": parses and re-serializes to confirm validity, returns raw JSON text
        """
        if not response.text or not response.text.strip():
            raise GeminiClientError("Gemini response was empty.")

        if response.finish_reason not in ("STOP", "MAX_TOKENS", ""):
            logger.warning("Unusual finish_reason from Gemini: %s", response.finish_reason)

        text = response.text.strip()

        if expect == "code":
            return self._strip_code_fences(text)

        if expect == "json":
            cleaned = self._strip_code_fences(text)
            try:
                json.loads(cleaned)
            except json.JSONDecodeError as exc:
                raise GeminiClientError(f"Expected valid JSON but parsing failed: {exc}") from exc
            return cleaned

        return text

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        fence_pattern = re.compile(r"^```[a-zA-Z0-9_+-]*\n(.*)\n```$", re.DOTALL)
        match = fence_pattern.match(text.strip())
        if match:
            return match.group(1)
        return text.replace("```python", "").replace("```", "").strip()

    def _build_payload(
        self,
        prompt: str,
        temperature: float,
        max_output_tokens: int,
        system_instruction: str | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens,
            },
        }
        if system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}
        return payload

    def _parse_response(self, raw: dict[str, Any]) -> GeminiResponse:
        try:
            candidates = raw.get("candidates", [])
            if not candidates:
                raise GeminiClientError(f"No candidates returned by Gemini API: {raw}")

            candidate = candidates[0]
            parts = candidate.get("content", {}).get("parts", [])
            text = "".join(part.get("text", "") for part in parts)
            finish_reason = candidate.get("finishReason", "")

            return GeminiResponse(text=text, finish_reason=finish_reason, model=self.model, raw=raw)
        except (KeyError, IndexError, TypeError) as exc:
            raise GeminiClientError(f"Malformed Gemini API response: {exc}") from exc
