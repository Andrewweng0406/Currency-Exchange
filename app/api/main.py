from __future__ import annotations

import json

from fastapi import FastAPI, Header, Request
from sqlalchemy import select

from app.backtesting.dataset import load_feature_frame
from app.config import settings
from app.database.schema import BankRate, Prediction
from app.database.session import make_session
from app.economic_events.importer import upcoming_event_risk
from app.exchange.planner import default_inputs, recommend_exchange
from app.line.client import verify_line_signature
from app.risk.scoring import latest_risk_snapshot

app = FastAPI(title="TWD FX Monitor API", version="0.1.0")


def _session():
    return make_session(settings()["database"]["url"])


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/overview")
def overview():
    session = _session()
    features = load_feature_frame(session)
    latest = features.iloc[-1]
    risk = latest_risk_snapshot(session)
    exchange = recommend_exchange(default_inputs(), risk.twd_risk_score, risk.opportunity_score)
    bank = session.execute(select(BankRate).order_by(BankRate.observed_at_utc.desc()).limit(1)).scalar_one_or_none()
    predictions = _predictions(session)
    return {
        "current": {
            "date": str(latest["date"]),
            "usdtwd": latest.get("USDTWD_CLOSE"),
            "bank_spot_selling": bank.spot_selling if bank else None,
            "bank_name": bank.bank_name if bank else None,
            "change_1d": latest.get("USDTWD_RETURN_1D"),
            "change_5d": latest.get("USDTWD_RETURN_5D"),
            "change_20d": latest.get("USDTWD_RETURN_20D"),
        },
        "risk": _dataclass_safe(risk),
        "exchange": exchange.__dict__,
        "predictions": predictions,
        "upcoming_events": upcoming_event_risk(session),
    }


@app.get("/features/latest")
def latest_features():
    session = _session()
    features = load_feature_frame(session)
    row = features.iloc[-1].to_dict()
    return {key: _json_safe(value) for key, value in row.items()}


@app.get("/charts/usdtwd")
def usdtwd_chart(limit: int = 365):
    session = _session()
    features = load_feature_frame(session).tail(limit)
    return [
        {
            "date": str(row["date"]),
            "usdtwd": _json_safe(row.get("USDTWD_CLOSE")),
            "dxy": _json_safe(row.get("DXY_CLOSE")),
            "us2y": _json_safe(row.get("US2Y_CLOSE")),
            "foreign_flow_5d": _json_safe(row.get("FOREIGN_FLOW_5D")),
            "prob_5d": None,
        }
        for _, row in features.iterrows()
    ]


@app.post("/line/webhook")
async def line_webhook(request: Request, x_line_signature: str | None = Header(default=None)):
    body = await request.body()
    verified = verify_line_signature(body, x_line_signature)
    payload = await request.json()
    user_ids = []
    for event in payload.get("events", []):
        source = event.get("source", {})
        if source.get("userId"):
            user_ids.append(source["userId"])
    return {"ok": True, "signature_verified": verified, "user_ids": user_ids}


def _predictions(session):
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
                "input_snapshot": json.loads(row.input_snapshot) if row.input_snapshot else {},
            },
        )
        if out[row.horizon]["input_snapshot"].get("prediction_interval_80"):
            out[row.horizon]["prediction_interval_80"] = out[row.horizon]["input_snapshot"]["prediction_interval_80"]
    return out


def _json_safe(value):
    try:
        if value != value:
            return None
    except TypeError:
        pass
    if hasattr(value, "item"):
        return value.item()
    return value


def _dataclass_safe(value):
    if hasattr(value, "__dict__"):
        return {key: _dataclass_safe(item) for key, item in value.__dict__.items()}
    if isinstance(value, dict):
        return {key: _dataclass_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_dataclass_safe(item) for item in value]
    return _json_safe(value)
