from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"


def run_script(script: str, *args: str) -> None:
    subprocess.run([str(PYTHON), str(ROOT / "scripts" / script), *args], cwd=ROOT, check=False)


def main() -> None:
    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(lambda: run_script("ingest_phase1.py"), "cron", hour=22, minute=30, id="daily_ingestion")
    scheduler.add_job(lambda: run_script("build_features.py"), "cron", hour=22, minute=45, id="daily_features")
    scheduler.add_job(lambda: run_script("train_models.py", "--horizon", "all"), "cron", hour=23, minute=0, id="daily_models")
    scheduler.add_job(lambda: run_script("evaluate_model_health.py"), "cron", hour=23, minute=20, id="model_health")
    scheduler.add_job(lambda: run_script("send_line_report.py", "--send"), "cron", hour=23, minute=30, id="line_report")
    scheduler.start()


if __name__ == "__main__":
    main()
