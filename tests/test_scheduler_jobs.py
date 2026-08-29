from app.scheduler.jobs import DAILY_PIPELINE, PipelineStep, run_daily_pipeline, scheduler_status


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
        steps=[
            PipelineStep("ingestion", "ingest_phase1.py"),
            PipelineStep("line_report", "send_line_report.py", ("--send",)),
        ]
    )

    assert not result["ok"]
    assert [item["returncode"] for item in result["results"]] == [2, 0]


def test_default_daily_pipeline_ends_with_line_report():
    assert [step.name for step in DAILY_PIPELINE] == ["ingestion", "features", "models", "model_health", "line_report"]
    assert DAILY_PIPELINE[-1].args == ("--send",)


def test_scheduler_status_disabled(monkeypatch):
    monkeypatch.setenv("SCHEDULER_ENABLED", "false")
    assert scheduler_status() == {"enabled": False, "running": False, "jobs": []}
