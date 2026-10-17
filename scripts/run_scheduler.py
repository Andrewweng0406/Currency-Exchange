from __future__ import annotations

import sys
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.scheduler.jobs import run_daily_pipeline


def main() -> None:
    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(run_daily_pipeline, "cron", hour=22, minute=30, id="daily_pipeline",
                      coalesce=True, max_instances=1, misfire_grace_time=3600,
                      kwargs={"trigger": "scheduled"})
    scheduler.start()


if __name__ == "__main__":
    main()
