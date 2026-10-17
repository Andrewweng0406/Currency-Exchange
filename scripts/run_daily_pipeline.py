from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.scheduler.jobs import run_daily_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the sequential daily pipeline; notifications require LINE_REPORT_ENABLED=true.")
    parser.parse_args()
    result = run_daily_pipeline()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
