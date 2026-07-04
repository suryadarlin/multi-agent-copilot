"""
context_manager.py

ContextManager: Prevents context rot in long-running multi-agent workflows
by maintaining a bounded, relevance-ranked context window, summarizing or
compressing stale entries, and producing scoped context payloads for
individual agents (progressive disclosure).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger("ai_engineering_copilot.context_manager")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logger.addHandler(_handler)


class ContextPriority(str, Enum):
    CRITICAL = "critical"
    NORMAL = "normal"
    LOW = "low"


@dataclass
class ContextEntry:
    agent_name: str
    content: str
    priority: ContextPriority
    timestamp: float = field(default_factory=time.time)
    token_estimate: int = 0
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.token_estimate == 0:
            self.token_estimate = max(1, len(self.content) // 4)


class ContextManager:
    """
    Maintains a rolling context buffer across the agent pipeline and exposes
    scoped, compressed views to individual agents to avoid context rot.
    """

    def __init__(
        self,
        max_total_tokens: int = 8000,
        compression_trigger_ratio: float = 0.85,
        recent_entry_window: int = 6,
    ) -> None:
        self._entries: list[ContextEntry] = []
        self._max_total_tokens = max_total_tokens
        self._compression_trigger_ratio = compression_trigger_ratio
        self._recent_entry_window = recent_entry_window

    def add_context(
        self,
        agent_name: str,
        content: str,
        priority: ContextPriority = ContextPriority.NORMAL,
        tags: Optional[list[str]] = None,
    ) -> None:
        if not content or not content.strip():
            logger.warning("Refusing to add empty context entry from %s", agent_name)
            return

        entry = ContextEntry(
            agent_name=agent_name,
            content=content.strip(),
            priority=priority,
            tags=tags or [],
        )
        self._entries.append(entry)
        logger.info(
            "Added context entry from=%s priority=%s tokens=%d",
            agent_name,
            priority.value,
            entry.token_estimate,
        )

        if self._current_token_usage() >= self._max_total_tokens * self._compression_trigger_ratio:
            self.compress_context()

    def _current_token_usage(self) -> int:
        return sum(entry.token_estimate for entry in self._entries)

    def summarize_old_context(self, keep_recent: Optional[int] = None) -> str:
        """
        Produces a condensed text summary of entries outside the recent
        window. Uses extractive truncation rather than calling an LLM, to
        keep this module synchronous and dependency-free.
        """
        keep_recent = keep_recent if keep_recent is not None else self._recent_entry_window
        if len(self._entries) <= keep_recent:
            return ""

        stale_entries = self._entries[:-keep_recent] if keep_recent > 0 else self._entries
        summary_lines = []
        for entry in stale_entries:
            condensed = entry.content[:160].rstrip()
            summary_lines.append(f"[{entry.agent_name}/{entry.priority.value}] {condensed}")

        summary = "\n".join(summary_lines)
        logger.info("Summarized %d stale entries into %d chars", len(stale_entries), len(summary))
        return summary

    def trim_irrelevant_context(self) -> int:
        """
        Removes LOW priority entries older than the recent window. Returns
        the number of entries removed.
        """
        if len(self._entries) <= self._recent_entry_window:
            return 0

        protected = set(id(e) for e in self._entries[-self._recent_entry_window:])
        before_count = len(self._entries)

        self._entries = [
            entry
            for entry in self._entries
            if id(entry) in protected or entry.priority != ContextPriority.LOW
        ]

        removed = before_count - len(self._entries)
        if removed:
            logger.info("Trimmed %d low-priority stale context entries", removed)
        return removed

    def compress_context(self) -> None:
        """
        Compresses old entries into a single synthetic summary entry,
        preserving CRITICAL entries verbatim and replacing NORMAL/LOW
        stale entries with a condensed digest. This is the primary
        anti-context-rot mechanism.
        """
        if len(self._entries) <= self._recent_entry_window:
            return

        recent = self._entries[-self._recent_entry_window:]
        stale = self._entries[: -self._recent_entry_window]

        critical_stale = [e for e in stale if e.priority == ContextPriority.CRITICAL]
        compressible_stale = [e for e in stale if e.priority != ContextPriority.CRITICAL]

        if compressible_stale:
            digest_text = self.summarize_old_context(keep_recent=0) if not critical_stale else "\n".join(
                f"[{e.agent_name}] {e.content[:160].rstrip()}" for e in compressible_stale
            )
            digest_entry = ContextEntry(
                agent_name="context_manager",
                content=f"[COMPRESSED SUMMARY of {len(compressible_stale)} entries]\n{digest_text}",
                priority=ContextPriority.NORMAL,
                tags=["compressed"],
            )
            self._entries = critical_stale + [digest_entry] + recent
        else:
            self._entries = critical_stale + recent

        new_usage = self._current_token_usage()
        logger.info(
            "Compressed context: now %d entries, ~%d tokens (limit %d)",
            len(self._entries),
            new_usage,
            self._max_total_tokens,
        )

    def build_agent_context(
        self,
        agent_name: str,
        relevant_tags: Optional[list[str]] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Builds a scoped context payload for a specific agent, implementing
        progressive disclosure: only entries tagged relevant to this agent
        (plus all CRITICAL entries and recent history) are included.
        """
        budget = max_tokens if max_tokens is not None else self._max_total_tokens
        relevant_tags_set = set(relevant_tags or [])

        candidates = [
            entry
            for entry in self._entries
            if entry.priority == ContextPriority.CRITICAL
            or not relevant_tags_set
            or relevant_tags_set.intersection(entry.tags)
            or entry in self._entries[-self._recent_entry_window:]
        ]

        selected: list[ContextEntry] = []
        running_tokens = 0
        for entry in reversed(candidates):
            if running_tokens + entry.token_estimate > budget:
                continue
            selected.append(entry)
            running_tokens += entry.token_estimate

        selected.reverse()

        payload_lines = [
            f"[{entry.agent_name} | {entry.priority.value}] {entry.content}"
            for entry in selected
        ]
        payload = "\n\n".join(payload_lines)

        logger.info(
            "Built scoped context for agent=%s: %d entries, ~%d tokens",
            agent_name,
            len(selected),
            running_tokens,
        )
        return payload

    def get_token_usage(self) -> dict[str, Any]:
        usage = self._current_token_usage()
        return {
            "current_tokens": usage,
            "max_tokens": self._max_total_tokens,
            "utilization_pct": round((usage / self._max_total_tokens) * 100, 2),
            "entry_count": len(self._entries),
        }