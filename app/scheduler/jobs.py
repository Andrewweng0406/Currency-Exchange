from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import structlog
from apscheduler.schedulers.background import BackgroundScheduler

from app.config import settings
from app.logging import configure_logging
from app.scheduler.state import PipelineBusyError, PipelineRecord, pipeline_lease, lease_fds

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PipelineStep:
    name: str
    script: str
    args: tuple[str, ...] = ()


DAILY_PIPELINE: tuple[PipelineStep, ...] = (
    PipelineStep("ingestion", "ingest_phase1.py"),
    PipelineStep("features", "build_features.py"),
    PipelineStep("models", "train_models.py", ("--horizon", "all")),
    PipelineStep("forward_validation", "run_forward_validation.py"),
    PipelineStep("model_health", "evaluate_model_health.py"),
    PipelineStep("line_report", "send_line_report.py", ("--send",)),
)

_scheduler: BackgroundScheduler | None = None


def run_script(script: str, *args: str) -> int:
    log = structlog.get_logger()
    command = [sys.executable, str(ROOT / "scripts" / script), *args]
    log.info("scheduled_script_started", script=script, args=list(args))
    try:
        completed = subprocess.run(command, cwd=ROOT, check=False, timeout=1800, pass_fds=lease_fds())
    except subprocess.TimeoutExpired:
        log.error("scheduled_script_timeout", script=script, timeout_seconds=1800)
        return 124
    log.info("scheduled_script_finished", script=script, returncode=completed.returncode)
    return completed.returncode


def run_daily_pipeline(steps: Iterable[PipelineStep] | None = None, continue_on_failure: bool = False,
                       *, trigger: str = "manual") -> dict:
    configure_logging()
    log = structlog.get_logger()
    results = []
    if steps is None:
        steps = [step for step in DAILY_PIPELINE if step.name != "line_report" or _line_report_enabled()]
    steps = list(steps)
    if not steps:
        raise ValueError("A pipeline must contain at least one step")
    try:
        with pipeline_lease():
            record = PipelineRecord(trigger)
            log.info("daily_pipeline_started", run_id=record.payload["run_id"], trigger=trigger)
            try:
                for step in steps:
                    record.step_started(step.name)
                    returncode = run_script(step.script, *step.args)
                    result = {"name": step.name, "script": step.script, "returncode": returncode}
                    results.append(result)
                    record.step_finished(result)
                    if returncode != 0 and not continue_on_failure:
                        break
            except BaseException as exc:
                record.finish("FAILED" if isinstance(exc, Exception) else "INTERRUPTED", type(exc).__name__)
                raise
            record.finish("SUCCEEDED" if all(item["returncode"] == 0 for item in results) else "FAILED")
    except PipelineBusyError:
        log.warning("daily_pipeline_already_running")
        return {"ok": False, "status": "ALREADY_RUNNING", "results": []}
    ok = all(item["returncode"] == 0 for item in results)
    log.info("daily_pipeline_finished", ok=ok, results=results)
    return {"ok": ok, "results": results}


def start_background_scheduler() -> BackgroundScheduler | None:
    global _scheduler
    if _scheduler and _scheduler.running:
        return _scheduler
    if not _scheduler_enabled():
        return None

    configure_logging()
    scheduler = BackgroundScheduler(timezone="UTC")
    # Run dependencies sequentially; wall-clock offsets do not ensure completion.
    scheduler.add_job(
        lambda: run_daily_pipeline(continue_on_failure=False, trigger="scheduled"),
        "cron", hour=22, minute=30, id="daily_pipeline",
        coalesce=True, max_instances=1, misfire_grace_time=3600,
    )
    scheduler.start()
    _scheduler = scheduler
    structlog.get_logger().info("background_scheduler_started", jobs=[job.id for job in scheduler.get_jobs()])
    return scheduler


def shutdown_background_scheduler() -> None:
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
    _scheduler = None


def scheduler_status() -> dict:
    if not _scheduler_enabled():
        return {"enabled": False, "running": False, "jobs": []}
    if _scheduler is None:
        return {"enabled": True, "running": False, "jobs": []}
    return {
        "enabled": True,
        "running": _scheduler.running,
        "jobs": [
            {
                "id": job.id,
                "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None,
            }
            for job in _scheduler.get_jobs()
        ],
    }


def _scheduler_enabled() -> bool:
    cfg = settings().get("scheduler", {})
    value = os.getenv("SCHEDULER_ENABLED", str(cfg.get("enabled", "false")))
    return str(value).lower() in {"1", "true", "yes", "on"}


def _line_report_enabled() -> bool:
    return os.getenv("LINE_REPORT_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
