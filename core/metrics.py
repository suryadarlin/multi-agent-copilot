"""
monitoring/metrics.py
=======================

Lightweight, dependency-free metrics tracker for the AI Software
Engineering Copilot. Persists counters/aggregates to a JSON snapshot file
so the Streamlit dashboard (or any external monitor) can read live stats
without needing a separate metrics backend (Prometheus, etc. can be added
later by swapping `MetricsStore`'s persistence layer).

Tracks:
    - execution_time      (per-workflow and per-agent)
    - retry_count         (auto-fix loop iterations)
    - test_success_rate   (passed / total across all test runs)
    - bug_frequency       (failed tests / critic findings per run)
    - security_issue_count
    - auto_fix_success_rate
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

METRICS_DIR = Path("/app/logs") if Path("/app").exists() else Path("./logs")
METRICS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_SNAPSHOT_PATH = METRICS_DIR / "metrics_snapshot.json"

_lock = threading.Lock()


@dataclass
class RunMetrics:
    run_id: str
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    agent_durations_ms: dict[str, float] = field(default_factory=dict)
    retry_count: int = 0
    tests_run: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    bugs_found: int = 0
    security_issues: int = 0
    auto_fix_attempts: int = 0
    auto_fix_successes: int = 0
    success: bool = False

    @property
    def total_duration_ms(self) -> float:
        end = self.finished_at or time.time()
        return (end - self.started_at) * 1000


@dataclass
class AggregateMetrics:
    total_runs: int = 0
    total_successful_runs: int = 0
    total_execution_time_ms: float = 0.0
    total_retries: int = 0
    total_tests_run: int = 0
    total_tests_passed: int = 0
    total_bugs_found: int = 0
    total_security_issues: int = 0
    total_auto_fix_attempts: int = 0
    total_auto_fix_successes: int = 0

    @property
    def avg_execution_time_ms(self) -> float:
        return self.total_execution_time_ms / self.total_runs if self.total_runs else 0.0

    @property
    def test_success_rate(self) -> float:
        return (self.total_tests_passed / self.total_tests_run * 100) if self.total_tests_run else 0.0

    @property
    def bug_frequency_per_run(self) -> float:
        return self.total_bugs_found / self.total_runs if self.total_runs else 0.0

    @property
    def auto_fix_success_rate(self) -> float:
        return (
            self.total_auto_fix_successes / self.total_auto_fix_attempts * 100
            if self.total_auto_fix_attempts
            else 0.0
        )

    @property
    def workflow_success_rate(self) -> float:
        return (self.total_successful_runs / self.total_runs * 100) if self.total_runs else 0.0


class MetricsStore:
    """
    In-memory + disk-persisted metrics tracker. Thread-safe for use from a
    single Streamlit process (Streamlit reruns happen on the same process,
    multiple sessions may share this store).
    """

    def __init__(self, snapshot_path: Path = METRICS_SNAPSHOT_PATH) -> None:
        self._snapshot_path = snapshot_path
        self._runs: dict[str, RunMetrics] = {}
        self._aggregate = AggregateMetrics()
        self._load_snapshot()

    # -- run lifecycle ----------------------------------------------------

    def start_run(self, run_id: str) -> RunMetrics:
        with _lock:
            run = RunMetrics(run_id=run_id)
            self._runs[run_id] = run
            return run

    def record_agent_duration(self, run_id: str, agent_name: str, duration_ms: float) -> None:
        with _lock:
            run = self._runs.get(run_id)
            if run:
                run.agent_durations_ms[agent_name] = duration_ms

    def record_retry(self, run_id: str) -> None:
        with _lock:
            run = self._runs.get(run_id)
            if run:
                run.retry_count += 1

    def record_test_results(self, run_id: str, tests_run: int, tests_passed: int, tests_failed: int) -> None:
        with _lock:
            run = self._runs.get(run_id)
            if run:
                run.tests_run = tests_run
                run.tests_passed = tests_passed
                run.tests_failed = tests_failed

    def record_bug_found(self, run_id: str, count: int = 1) -> None:
        with _lock:
            run = self._runs.get(run_id)
            if run:
                run.bugs_found += count

    def record_security_issues(self, run_id: str, count: int) -> None:
        with _lock:
            run = self._runs.get(run_id)
            if run:
                run.security_issues += count

    def record_auto_fix_attempt(self, run_id: str, success: bool) -> None:
        with _lock:
            run = self._runs.get(run_id)
            if run:
                run.auto_fix_attempts += 1
                if success:
                    run.auto_fix_successes += 1

    def finish_run(self, run_id: str, success: bool) -> Optional[RunMetrics]:
        with _lock:
            run = self._runs.get(run_id)
            if not run:
                return None
            run.finished_at = time.time()
            run.success = success
            self._fold_into_aggregate(run)
            self._save_snapshot()
            return run

    # -- aggregation --------------------------------------------------------

    def _fold_into_aggregate(self, run: RunMetrics) -> None:
        agg = self._aggregate
        agg.total_runs += 1
        agg.total_successful_runs += 1 if run.success else 0
        agg.total_execution_time_ms += run.total_duration_ms
        agg.total_retries += run.retry_count
        agg.total_tests_run += run.tests_run
        agg.total_tests_passed += run.tests_passed
        agg.total_bugs_found += run.bugs_found
        agg.total_security_issues += run.security_issues
        agg.total_auto_fix_attempts += run.auto_fix_attempts
        agg.total_auto_fix_successes += run.auto_fix_successes

    def get_aggregate(self) -> AggregateMetrics:
        with _lock:
            return self._aggregate

    def get_run(self, run_id: str) -> Optional[RunMetrics]:
        with _lock:
            return self._runs.get(run_id)

    def snapshot(self) -> dict[str, Any]:
        with _lock:
            agg = self._aggregate
            return {
                "aggregate": asdict(agg),
                "derived": {
                    "avg_execution_time_ms": agg.avg_execution_time_ms,
                    "test_success_rate": agg.test_success_rate,
                    "bug_frequency_per_run": agg.bug_frequency_per_run,
                    "auto_fix_success_rate": agg.auto_fix_success_rate,
                    "workflow_success_rate": agg.workflow_success_rate,
                },
                "recent_runs": [asdict(r) for r in list(self._runs.values())[-20:]],
            }

    # -- persistence ----------------------------------------------------------

    def _save_snapshot(self) -> None:
        try:
            self._snapshot_path.write_text(json.dumps(self.snapshot(), indent=2, default=str))
        except OSError:
            pass  # metrics persistence is best-effort, never fatal to the workflow

    def _load_snapshot(self) -> None:
        if not self._snapshot_path.exists():
            return
        try:
            data = json.loads(self._snapshot_path.read_text())
            agg_data = data.get("aggregate", {})
            self._aggregate = AggregateMetrics(**{k: v for k, v in agg_data.items() if k in AggregateMetrics.__dataclass_fields__})
        except (json.JSONDecodeError, OSError, TypeError):
            self._aggregate = AggregateMetrics()


# Module-level singleton for convenient import across agents/dashboard
metrics_store = MetricsStore()