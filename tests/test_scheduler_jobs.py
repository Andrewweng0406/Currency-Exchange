from app.scheduler.jobs import DAILY_PIPELINE, PipelineStep, run_daily_pipeline, scheduler_status
import pytest


@pytest.fixture(autouse=True)
def isolated_execution_records(tmp_path, monkeypatch):
    monkeypatch.setattr("app.scheduler.state.STATE_DIR", tmp_path / "operations")


def test_daily_pipeline_runs_steps_in_order(monkeypatch):
    calls = []

    def fake_run_script(script, *args):
        calls.append((script, args))
        return 0

    monkeypatch.setattr("app.scheduler.jobs.run_script", fake_run_script)
    result = run_daily_pipeline(
        steps=[
            PipelineStep("first", "one.py"),
            PipelineStep("second", "two.py", ("--flag",)),
        ]
    )

    assert result["ok"]
    assert calls == [("one.py", ()), ("two.py", ("--flag",))]


def test_daily_pipeline_records_failures_but_continues(monkeypatch):
    returncodes = iter([2, 0])

    def fake_run_script(script, *args):
        return next(returncodes)

    monkeypatch.setattr("app.scheduler.jobs.run_script", fake_run_script)
    result = run_daily_pipeline(
        continue_on_failure=True,
        steps=[
            PipelineStep("ingestion", "ingest_phase1.py"),
            PipelineStep("line_report", "send_line_report.py", ("--send",)),
        ]
    )

    assert not result["ok"]
    assert [item["returncode"] for item in result["results"]] == [2, 0]


def test_default_daily_pipeline_ends_with_line_report():
    assert [step.name for step in DAILY_PIPELINE] == ["ingestion", "features", "models", "forward_validation", "model_health", "line_report"]
    assert DAILY_PIPELINE[-1].args == ("--send",)


def test_scheduler_status_disabled(monkeypatch):
    monkeypatch.setenv("SCHEDULER_ENABLED", "false")
    assert scheduler_status() == {"enabled": False, "running": False, "jobs": []}


def test_pipeline_stops_dependent_steps_after_failure(monkeypatch):
    calls = []
    def fail(script, *args):
        calls.append(script)
        return 2
    monkeypatch.setattr("app.scheduler.jobs.run_script", fail)
    result = run_daily_pipeline(continue_on_failure=False)
    assert not result["ok"]
    assert calls == ["ingest_phase1.py"]


def test_notifications_require_explicit_enablement(monkeypatch):
    calls = []
    monkeypatch.delenv("LINE_REPORT_ENABLED", raising=False)
    monkeypatch.setattr("app.scheduler.jobs.run_script", lambda script, *args: calls.append(script) or 0)
    assert run_daily_pipeline()["ok"]
    assert "run_forward_validation.py" in calls
    assert "send_line_report.py" not in calls
    calls.clear()
    monkeypatch.setenv("LINE_REPORT_ENABLED", "true")
    assert run_daily_pipeline()["ok"]
    assert calls[-1] == "send_line_report.py"


def test_timeout_returns_failure_instead_of_hanging_scheduler(monkeypatch):
    import subprocess
    from app.scheduler.jobs import run_script
    def expire(*args, **kwargs):
        assert kwargs["timeout"] == 1800
        raise subprocess.TimeoutExpired(args[0], 1800)
    monkeypatch.setattr("app.scheduler.jobs.subprocess.run", expire)
    assert run_script("ingest_phase1.py") == 124
