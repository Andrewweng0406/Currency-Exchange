from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

from sqlalchemy import select

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.alerts.engine import generate_alerts, persist_alert, should_send_alert
from app.backtesting.dataset import load_feature_frame
from app.config import settings
from app.database.schema import BankRate, Prediction
from app.database.session import make_session
from app.exchange.planner import default_inputs, recommend_exchange
from app.economic_events.importer import upcoming_event_risk
from app.line.client import LineMessagingClient
from app.line.formatter import daily_report
from app.risk.scoring import latest_risk_snapshot


def main() -> None:
    dry_run = "--send" not in sys.argv
    session = make_session(settings()["database"]["url"])
    features = load_feature_frame(session)
    latest = features.iloc[-1]
    risk = latest_risk_snapshot(session)
    exchange = recommend_exchange(default_inputs(), risk.twd_risk_score, risk.opportunity_score)
    predictions = _predictions(session)
    bank = session.execute(select(BankRate.spot_selling).order_by(BankRate.observed_at_utc.desc()).limit(1)).scalar_one_or_none()
    message = daily_report(
        current_usdtwd=float(latest["USDTWD_CLOSE"]),
        bank_spot_selling=bank,
        changes={"1d": latest.get("USDTWD_RETURN_1D"), "5d": latest.get("USDTWD_RETURN_5D"), "20d": latest.get("USDTWD_RETURN_20D")},
        predictions=predictions,
        risk=risk,
        exchange=exchange,
        upcoming_events=upcoming_event_risk(session),
    )
    alerts = [candidate for candidate in generate_alerts(risk, predictions) if should_send_alert(session, candidate)]
    if dry_run:
        print(message)
        print("\n--- ALERTS ---")
        print(json.dumps([asdict(a) for a in alerts], ensure_ascii=False, indent=2))
        return
    client = LineMessagingClient()
    print(client.send_text(message))
    for alert in alerts:
        saved = persist_alert(session, alert)
        print(client.send_text(saved.message))


def _predictions(session) -> dict[str, dict[str, float | None]]:
    rows = session.execute(
        select(Prediction).where(Prediction.model_version == "phase4_ensemble_v1").order_by(Prediction.observed_at_utc.desc())
    ).scalars().all()
    out = {}
    for row in rows:
        snapshot = json.loads(row.input_snapshot) if row.input_snapshot else {}
        out.setdefault(
            row.horizon,
            {
                "prob_up": row.prob_up,
                "prob_down": row.prob_down,
                "confidence": row.confidence,
                "prediction_interval_80": snapshot.get("prediction_interval_80"),
            },
        )
    return out


if __name__ == "__main__":
    main()
