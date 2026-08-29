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
    PipelineStep("model_health", "evaluate_model_health.py"),
    PipelineStep("line_report", "send_line_report.py", ("--send",)),
)

_scheduler: BackgroundScheduler | None = None


def run_script(script: str, *args: str) -> int:
    log = structlog.get_logger()
    command = [sys.executable, str(ROOT / "scripts" / script), *args]
    log.info("scheduled_script_started", script=script, args=list(args))
    completed = subprocess.run(command, cwd=ROOT, check=False)
    log.info("scheduled_script_finished", script=script, returncode=completed.returncode)
    return completed.returncode


def run_daily_pipeline(steps: Iterable[PipelineStep] = DAILY_PIPELINE, continue_on_failure: bool = True) -> dict:
    configure_logging()
    log = structlog.get_logger()
    results = []
    log.info("daily_pipeline_started")
    for step in steps:
        returncode = run_script(step.script, *step.args)
        results.append({"name": step.name, "script": step.script, "returncode": returncode})
        if returncode != 0 and not continue_on_failure:
            break
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
    scheduler.add_job(lambda: run_script("ingest_phase1.py"), "cron", hour=22, minute=30, id="daily_ingestion", coalesce=True, max_instances=1)
    scheduler.add_job(lambda: run_script("build_features.py"), "cron", hour=22, minute=45, id="daily_features", coalesce=True, max_instances=1)
    scheduler.add_job(
        lambda: run_script("train_models.py", "--horizon", "all"),
        "cron",
        hour=23,
        minute=0,
        id="daily_models",
        coalesce=True,
        max_instances=1,
    )
    scheduler.add_job(lambda: run_script("evaluate_model_health.py"), "cron", hour=23, minute=20, id="model_health", coalesce=True, max_instances=1)
    scheduler.add_job(lambda: run_script("send_line_report.py", "--send"), "cron", hour=23, minute=30, id="line_report", coalesce=True, max_instances=1)
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
