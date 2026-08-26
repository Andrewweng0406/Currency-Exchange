from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.config import settings
from app.database.session import make_session
from app.exchange.planner import default_inputs, persist_exchange_plan, recommend_exchange
from app.risk.scoring import latest_risk_snapshot


def main() -> None:
    session = make_session(settings()["database"]["url"])
    risk = latest_risk_snapshot(session)
    inputs = default_inputs()
    recommendation = recommend_exchange(inputs, risk.twd_risk_score, risk.opportunity_score)
    persist_exchange_plan(session, inputs, recommendation)
    print(json.dumps({"risk": asdict(risk), "exchange": asdict(recommendation)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
