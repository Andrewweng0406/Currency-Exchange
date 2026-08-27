from __future__ import annotations

import json
import sys
from pathlib import Path

from sqlalchemy import select

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.ai.openai_interpreter import interpret_risk_context, persist_ai_interpretation
from app.backtesting.dataset import load_feature_frame
from app.config import settings
from app.database.schema import BankRate, Prediction
from app.database.session import make_session
from app.economic_events.importer import upcoming_event_risk
from app.risk.scoring import latest_risk_snapshot


def main() -> None:
    save = "--save" in sys.argv
    session = make_session(settings()["database"]["url"])
    features = load_feature_frame(session)
    latest = features.iloc[-1]
    risk = latest_risk_snapshot(session)
    predictions = _predictions(session)
    bank = session.execute(select(BankRate.spot_selling).order_by(BankRate.observed_at_utc.desc()).limit(1)).scalar_one_or_none()
    result = interpret_risk_context(
        risk,
        predictions,
        {
            "usdtwd": float(latest["USDTWD_CLOSE"]),
            "bank_spot_selling": bank,
            "change_1d": latest.get("USDTWD_RETURN_1D"),
            "change_5d": latest.get("USDTWD_RETURN_5D"),
            "change_20d": latest.get("USDTWD_RETURN_20D"),
        },
        upcoming_event_risk(session),
    )
    if save:
        persist_ai_interpretation(session, result)
    print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))


def _predictions(session) -> dict[str, dict[str, float | None]]:
    rows = session.execute(
        select(Prediction).where(Prediction.model_version == "phase4_ensemble_v1").order_by(Prediction.observed_at_utc.desc())
    ).scalars().all()
    out = {}
    for row in rows:
        out.setdefault(
            row.horizon,
            {
                "prob_up": row.prob_up,
                "prob_down": row.prob_down,
                "expected_return": row.expected_return,
                "confidence": row.confidence,
                "risk_score": row.risk_score,
            },
        )
    return out


if __name__ == "__main__":
    main()
