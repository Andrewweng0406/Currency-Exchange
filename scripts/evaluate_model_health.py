from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.config import settings
from app.database.session import make_session
from app.monitoring.model_health import evaluate_matured_predictions, latest_model_health_summary


def main() -> None:
    session = make_session(settings()["database"]["url"])
    rows = evaluate_matured_predictions(session)
    summary = latest_model_health_summary(session)
    print(json.dumps({"rows_written": rows, "summary": asdict(summary)}, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
