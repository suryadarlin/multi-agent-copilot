"""
memory_store.py

MemoryStore: Persistent cross-session memory for the agent pipeline,
backed by SQLite. Tracks execution history, debugging fixes, recurring
errors, and reusable solutions so the system avoids repeating past
mistakes.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

logger = logging.getLogger("ai_engineering_copilot.memory_store")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logger.addHandler(_handler)


class MemoryStoreError(Exception):
    """Raised on unrecoverable persistence failures."""


@dataclass
class FixRecord:
    error_signature: str
    error_message: str
    fix_description: str
    fix_code_diff: str
    success_count: int
    last_used_ts: float


class MemoryStore:
    """
    SQLite-backed persistent memory store shared across agent runs.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS executions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        agent_name TEXT NOT NULL,
        task_summary TEXT NOT NULL,
        result_json TEXT NOT NULL,
        success INTEGER NOT NULL,
        created_ts REAL NOT NULL
    );

    CREATE TABLE IF NOT EXISTS error_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        error_signature TEXT NOT NULL,
        error_message TEXT NOT NULL,
        occurrence_count INTEGER NOT NULL DEFAULT 1,
        first_seen_ts REAL NOT NULL,
        last_seen_ts REAL NOT NULL
    );

    CREATE TABLE IF NOT EXISTS fix_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        error_signature TEXT NOT NULL UNIQUE,
        error_message TEXT NOT NULL,
        fix_description TEXT NOT NULL,
        fix_code_diff TEXT NOT NULL,
        success_count INTEGER NOT NULL DEFAULT 1,
        last_used_ts REAL NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_error_signature ON error_history(error_signature);
    CREATE INDEX IF NOT EXISTS idx_fix_signature ON fix_history(error_signature);
    """

    def __init__(self, db_path: str = "memory_store.db") -> None:
        self._db_path = Path(db_path)
        self._init_schema()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self._db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        try:
            with self._connection() as conn:
                conn.executescript(self.SCHEMA)
            logger.info("Initialized memory store schema at %s", self._db_path)
        except sqlite3.Error as exc:
            logger.error("Failed to initialize schema: %s", exc)
            raise MemoryStoreError(f"Schema initialization failed: {exc}") from exc

    def save_execution(
        self, agent_name: str, task_summary: str, result: dict[str, Any], success: bool
    ) -> int:
        if not agent_name or not task_summary:
            raise MemoryStoreError("agent_name and task_summary are required")

        try:
            with self._connection() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO executions (agent_name, task_summary, result_json, success, created_ts)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (agent_name, task_summary, json.dumps(result), int(success), time.time()),
                )
                execution_id = cursor.lastrowid
            logger.info("Saved execution id=%d agent=%s success=%s", execution_id, agent_name, success)
            return execution_id
        except sqlite3.Error as exc:
            logger.error("Failed to save execution: %s", exc)
            raise MemoryStoreError(f"save_execution failed: {exc}") from exc

    def load_previous_errors(self, error_signature: str, limit: int = 10) -> list[dict[str, Any]]:
        try:
            with self._connection() as conn:
                rows = conn.execute(
                    """
                    SELECT * FROM error_history
                    WHERE error_signature = ?
                    ORDER BY last_seen_ts DESC
                    LIMIT ?
                    """,
                    (error_signature, limit),
                ).fetchall()
            results = [dict(row) for row in rows]
            logger.info(
                "Loaded %d prior occurrences for signature=%s", len(results), error_signature
            )
            return results
        except sqlite3.Error as exc:
            logger.error("Failed to load previous errors: %s", exc)
            raise MemoryStoreError(f"load_previous_errors failed: {exc}") from exc

    def record_error_occurrence(self, error_signature: str, error_message: str) -> int:
        """
        Increments the occurrence counter for a recurring error signature,
        inserting a new row if this is the first occurrence.
        """
        now = time.time()
        try:
            with self._connection() as conn:
                existing = conn.execute(
                    "SELECT id, occurrence_count FROM error_history WHERE error_signature = ?",
                    (error_signature,),
                ).fetchone()

                if existing:
                    new_count = existing["occurrence_count"] + 1
                    conn.execute(
                        "UPDATE error_history SET occurrence_count = ?, last_seen_ts = ? WHERE id = ?",
                        (new_count, now, existing["id"]),
                    )
                    logger.info(
                        "Error signature=%s now seen %d times", error_signature, new_count
                    )
                    return new_count

                conn.execute(
                    """
                    INSERT INTO error_history
                    (error_signature, error_message, occurrence_count, first_seen_ts, last_seen_ts)
                    VALUES (?, ?, 1, ?, ?)
                    """,
                    (error_signature, error_message, now, now),
                )
                return 1
        except sqlite3.Error as exc:
            logger.error("Failed to record error occurrence: %s", exc)
            raise MemoryStoreError(f"record_error_occurrence failed: {exc}") from exc

    def store_fix_history(
        self,
        error_signature: str,
        error_message: str,
        fix_description: str,
        fix_code_diff: str,
    ) -> None:
        now = time.time()
        try:
            with self._connection() as conn:
                existing = conn.execute(
                    "SELECT id, success_count FROM fix_history WHERE error_signature = ?",
                    (error_signature,),
                ).fetchone()

                if existing:
                    conn.execute(
                        """
                        UPDATE fix_history
                        SET success_count = ?, last_used_ts = ?, fix_description = ?, fix_code_diff = ?
                        WHERE id = ?
                        """,
                        (
                            existing["success_count"] + 1,
                            now,
                            fix_description,
                            fix_code_diff,
                            existing["id"],
                        ),
                    )
                else:
                    conn.execute(
                        """
                        INSERT INTO fix_history
                        (error_signature, error_message, fix_description, fix_code_diff, success_count, last_used_ts)
                        VALUES (?, ?, ?, ?, 1, ?)
                        """,
                        (error_signature, error_message, fix_description, fix_code_diff, now),
                    )
            logger.info("Stored fix history for signature=%s", error_signature)
        except sqlite3.Error as exc:
            logger.error("Failed to store fix history: %s", exc)
            raise MemoryStoreError(f"store_fix_history failed: {exc}") from exc

    def retrieve_previous_solution(self, error_signature: str) -> Optional[FixRecord]:
        try:
            with self._connection() as conn:
                row = conn.execute(
                    "SELECT * FROM fix_history WHERE error_signature = ?",
                    (error_signature,),
                ).fetchone()

            if row is None:
                logger.info("No previous solution found for signature=%s", error_signature)
                return None

            record = FixRecord(
                error_signature=row["error_signature"],
                error_message=row["error_message"],
                fix_description=row["fix_description"],
                fix_code_diff=row["fix_code_diff"],
                success_count=row["success_count"],
                last_used_ts=row["last_used_ts"],
            )
            logger.info(
                "Found reusable solution for signature=%s (used %d times)",
                error_signature,
                record.success_count,
            )
            return record
        except sqlite3.Error as exc:
            logger.error("Failed to retrieve previous solution: %s", exc)
            raise MemoryStoreError(f"retrieve_previous_solution failed: {exc}") from exc

    def clear_old_memory(self, older_than_days: int = 30) -> dict[str, int]:
        cutoff_ts = time.time() - (older_than_days * 86400)
        try:
            with self._connection() as conn:
                exec_cursor = conn.execute(
                    "DELETE FROM executions WHERE created_ts < ?", (cutoff_ts,)
                )
                error_cursor = conn.execute(
                    "DELETE FROM error_history WHERE last_seen_ts < ?", (cutoff_ts,)
                )
            result = {
                "executions_deleted": exec_cursor.rowcount,
                "error_records_deleted": error_cursor.rowcount,
            }
            logger.info("Cleared old memory: %s", result)
            return result
        except sqlite3.Error as exc:
            logger.error("Failed to clear old memory: %s", exc)
            raise MemoryStoreError(f"clear_old_memory failed: {exc}") from exc