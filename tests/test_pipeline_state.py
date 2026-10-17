from datetime import datetime, timedelta, timezone
import json
import os
import subprocess
import sys
import time

import pytest

from app.scheduler import state
from app.scheduler.jobs import PipelineStep, run_daily_pipeline


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    monkeypatch.setattr(state, "STATE_DIR", tmp_path / "operations")
    monkeypatch.setenv("LINE_REPORT_ENABLED", "false")


def test_read_before_first_run_does_not_create_any_files():
    result = state.pipeline_status()
    assert result["status"] == "NO_RECORD"
    assert not state.STATE_DIR.exists()


def test_empty_pipeline_cannot_create_a_false_success_record():
    with pytest.raises(ValueError, match="at least one step"):
        run_daily_pipeline([])
    assert not state.STATE_DIR.exists()


def test_step_progress_and_trigger_are_visible_while_work_is_running(monkeypatch):
    def run(script, *args):
        status = state.pipeline_status()
        assert status["status"] == "RUNNING"
        assert status["latest_run"]["trigger"] == "scheduled"
        assert status["latest_run"]["current_step"] == ("first" if script == "one.py" else "second")
        assert len(status["latest_run"]["results"]) == (0 if script == "one.py" else 1)
        return 0
    monkeypatch.setattr("app.scheduler.jobs.run_script", run)
    assert run_daily_pipeline([PipelineStep("first", "one.py"), PipelineStep("second", "two.py")], trigger="scheduled")["ok"]
    assert state.pipeline_status()["latest_run"]["current_step"] is None


def test_failure_preserves_last_success_and_stops_dependencies(monkeypatch):
    now = datetime(2026, 10, 17, 23, tzinfo=timezone.utc)
    monkeypatch.setattr(state, "utc_now", lambda: now)
    monkeypatch.setattr("app.scheduler.jobs.run_script", lambda *a: 0)
    assert run_daily_pipeline([PipelineStep("first", "one.py")], trigger="scheduled")["ok"]
    success = state.pipeline_status(now)["last_success"]
    calls = []
    def fail(script, *args):
        calls.append(script)
        return 2
    monkeypatch.setattr("app.scheduler.jobs.run_script", fail)
    result = run_daily_pipeline([PipelineStep("first", "one.py"), PipelineStep("second", "two.py")])
    assert not result["ok"] and calls == ["one.py"]
    status = state.pipeline_status(now)
    assert status["status"] == "FAILED"
    assert status["last_success"] == success
    assert status["latest_run"]["results"][0]["returncode"] == 2
    assert status["latest_run"]["failed_step"] == "first"
    assert len(list(state.STATE_DIR.glob("run_*.json"))) == 2


def test_real_other_process_lock_blocks_duplicate_and_detects_process_death(monkeypatch):
    state.STATE_DIR.mkdir(parents=True)
    lock = state.STATE_DIR / "daily_pipeline.lock"
    code = "import fcntl,time,sys; f=open(sys.argv[1],'a'); fcntl.flock(f,fcntl.LOCK_EX); print('locked',flush=True); time.sleep(30)"
    child = subprocess.Popen([sys.executable, "-c", code, str(lock)], stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "locked"
        record = state.PipelineRecord("scheduled")
        before = (state.STATE_DIR / "latest.json").read_bytes()
        calls = []
        monkeypatch.setattr("app.scheduler.jobs.run_script", lambda *a: calls.append(a) or 0)
        assert state.pipeline_status()["status"] == "RUNNING"
        assert run_daily_pipeline([PipelineStep("one", "one.py")])["status"] == "ALREADY_RUNNING"
        assert not calls
        assert (state.STATE_DIR / "latest.json").read_bytes() == before
    finally:
        child.terminate()
        child.wait(timeout=5)
    assert state.pipeline_status()["status"] == "INTERRUPTED"
    assert json.loads((state.STATE_DIR / "latest.json").read_text())["status"] == "RUNNING"
    with state.pipeline_lease():
        recovered = state.PipelineRecord("manual")
        recovered.finish("SUCCEEDED")
    assert state.pipeline_status()["status"] == "SUCCEEDED"


def test_success_becomes_stale_after_next_schedule_deadline(monkeypatch):
    start = datetime(2026, 10, 17, 22, 31, tzinfo=timezone.utc)
    monkeypatch.setattr(state, "utc_now", lambda: start)
    with state.pipeline_lease():
        state.PipelineRecord("scheduled").finish("SUCCEEDED")
    assert state.pipeline_status(start + timedelta(days=1, hours=1))["status"] == "SUCCEEDED"
    late = state.pipeline_status(start + timedelta(days=1, hours=4))
    assert late["status"] == "STALE" and late["overdue"]


def test_child_step_retains_lock_when_pipeline_parent_is_killed():
    script_dir = state.STATE_DIR.parent / "scripts"
    script_dir.mkdir()
    pid_file = state.STATE_DIR.parent / "child.pid"
    (script_dir / "keepalive.py").write_text(
        "import os,time\nfrom pathlib import Path\n"
        + f"Path({str(pid_file)!r}).write_text(str(os.getpid()))\n"
        + "time.sleep(30)\n")
    code = """import sys
from pathlib import Path
from app.scheduler import state,jobs
state.STATE_DIR=Path(sys.argv[1])
jobs.ROOT=state.STATE_DIR.parent
with state.pipeline_lease():
    state.PipelineRecord('manual')
    jobs.run_script('keepalive.py')
"""
    parent = subprocess.Popen([sys.executable, "-c", code, str(state.STATE_DIR)], stdout=subprocess.DEVNULL)
    child_pid = None
    try:
        deadline = time.monotonic() + 5
        while not pid_file.exists() and time.monotonic() < deadline:
            assert parent.poll() is None
            time.sleep(0.02)
        child_pid = int(pid_file.read_text())
        parent.kill()
        parent.wait(timeout=5)
        assert state.pipeline_status()["status"] == "RUNNING"
        with pytest.raises(state.PipelineBusyError):
            with state.pipeline_lease():
                pass
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=5)
        if child_pid is not None:
            os.kill(child_pid, 15)
    for _ in range(50):
        if not state._active():
            break
        time.sleep(0.02)
    assert state.pipeline_status()["status"] == "INTERRUPTED"


def test_corrupt_state_is_explicit_and_not_rewritten():
    state.STATE_DIR.mkdir()
    path = state.STATE_DIR / "latest.json"
    path.write_text("broken JSON")
    assert state.pipeline_status()["status"] == "STATE_ERROR"
    assert path.read_text() == "broken JSON"


def test_unexpected_script_error_is_recorded_without_exception_message(monkeypatch):
    def fail(*args):
        raise OSError("sensitive details must not be persisted")
    monkeypatch.setattr("app.scheduler.jobs.run_script", fail)
    with pytest.raises(OSError):
        run_daily_pipeline([PipelineStep("one", "one.py")])
    assert state.pipeline_status()["status"] == "FAILED"
    assert state.pipeline_status()["latest_run"]["failed_step"] == "one"
    content = (state.STATE_DIR / "latest.json").read_text()
    assert "OSError" in content
    assert "sensitive details" not in content
