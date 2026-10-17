"""Durable pipeline execution records on the model evidence volume."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
import fcntl
import json
import os
from pathlib import Path
import tempfile
from uuid import uuid4

from app.config import ROOT

STATE_DIR = ROOT / "models" / "artifacts" / "operations"
COMPLETION_GRACE = timedelta(hours=4)
_lease_fd = ContextVar("pipeline_lease_fd", default=None)


def lease_fds():
    descriptor = _lease_fd.get()
    return () if descriptor is None else (descriptor,)


def utc_now():
    return datetime.now(timezone.utc)


def _write(path: Path, payload: dict):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class PipelineBusyError(RuntimeError):
    pass


@contextmanager
def pipeline_lease():
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with (STATE_DIR / "daily_pipeline.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PipelineBusyError("A pipeline is already running") from exc
        token = _lease_fd.set(handle.fileno())
        try:
            yield
        finally:
            _lease_fd.reset(token)
            fcntl.flock(handle, fcntl.LOCK_UN)


class PipelineRecord:
    def __init__(self, trigger: str):
        self.payload = {"run_id": uuid4().hex, "trigger": trigger, "status": "RUNNING",
                        "started_at_utc": utc_now().isoformat(), "finished_at_utc": None,
                        "current_step": None, "results": []}
        self.save()

    def save(self):
        _write(STATE_DIR / f"run_{self.payload['run_id']}.json", self.payload)
        _write(STATE_DIR / "latest.json", self.payload)

    def step_started(self, name: str):
        self.payload["current_step"] = name
        self.save()

    def step_finished(self, result: dict):
        self.payload["results"].append(dict(result))
        self.save()

    def finish(self, status: str, error_type: str | None = None):
        if status != "SUCCEEDED":
            self.payload["failed_step"] = self.payload["current_step"]
        self.payload.update(status=status, finished_at_utc=utc_now().isoformat(), current_step=None)
        if error_type:
            self.payload["error_type"] = error_type
        self.save()
        if status == "SUCCEEDED":
            _write(STATE_DIR / "last_success.json", self.payload)


def _load(name: str):
    path = STATE_DIR / name
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or value.get("status") not in {"RUNNING", "SUCCEEDED", "FAILED", "INTERRUPTED"}:
        raise ValueError("Invalid execution record")
    if datetime.fromisoformat(value["started_at_utc"]).tzinfo is None:
        raise ValueError("Execution timestamp must include timezone")
    if not isinstance(value["run_id"], str) or not isinstance(value["results"], list):
        raise ValueError("Invalid execution record fields")
    if value["status"] != "RUNNING":
        if datetime.fromisoformat(value["finished_at_utc"]).tzinfo is None:
            raise ValueError("Completion timestamp must include timezone")
    if name == "last_success.json" and value["status"] != "SUCCEEDED":
        raise ValueError("Invalid success checkpoint")
    return value


def _active():
    try:
        with (STATE_DIR / "daily_pipeline.lock").open("r") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            fcntl.flock(handle, fcntl.LOCK_UN)
    except FileNotFoundError:
        pass
    return False


def pipeline_status(now: datetime | None = None):
    """Read-only; checking an uninitialized service creates no files."""
    now = now or utc_now()
    if now.tzinfo is None:
        raise ValueError("Current timestamp must include timezone")
    now = now.astimezone(timezone.utc)
    due = now.replace(hour=22, minute=30, second=0, microsecond=0)
    if due > now:
        due -= timedelta(days=1)
    try:
        latest = _load("latest.json")
        success = _load("last_success.json")
        if latest is not None and latest["status"] == "SUCCEEDED" and (
            success is None or datetime.fromisoformat(latest["started_at_utc"]) > datetime.fromisoformat(success["started_at_utc"])
        ):
            success = latest
        if latest is None:
            status = "NO_RECORD"
        elif latest["status"] == "RUNNING":
            status = "RUNNING" if _active() else "INTERRUPTED"
        else:
            status = latest["status"]
        success_current = success is not None and datetime.fromisoformat(success["started_at_utc"]) >= due
        overdue = now >= due + COMPLETION_GRACE and not success_current
        if status == "SUCCEEDED" and overdue:
            status = "STALE"
        return {"status": status, "overdue": overdue, "scheduled_at_utc": due.isoformat(),
                "completion_deadline_utc": (due + COMPLETION_GRACE).isoformat(),
                "latest_run": latest, "last_success": success,
                "notifications_enabled": os.getenv("LINE_REPORT_ENABLED", "false").lower() in {"1", "true", "yes", "on"}}
    except (OSError, ValueError, KeyError, TypeError):
        return {"status": "STATE_ERROR", "latest_run": None, "last_success": None}
