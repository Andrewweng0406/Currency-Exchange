from __future__ import annotations

import json
import math
import pandas as pd
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from sqlalchemy import select

from app.ai.openai_interpreter import latest_ai_interpretation
from app.backtesting.dataset import load_feature_frame
from app.backtesting.strategy_compare import build_strategy_report, compare_exchange_strategies, walk_forward_tune_strategy
from app.config import settings
from app.database.schema import Prediction
from app.database.session import make_session
from app.economic_events.importer import upcoming_event_risk
from app.exchange.planner import default_inputs, recommend_exchange
from app.line.client import verify_line_signature
from app.line.family_reasons import family_reasons
from app.monitoring.model_health import latest_model_health_summary
from app.monitoring.forward_validation import forward_validation_summary
from app.ops.advice import guard_model_advice
from app.ops.bank_quotes import latest_bank_quote
from app.ops.data_quality import data_coverage_report, summarize_data_quality
from app.risk.scoring import latest_risk_snapshot
from app.scheduler.jobs import scheduler_status, shutdown_background_scheduler, start_background_scheduler
from app.scheduler.state import pipeline_status


@asynccontextmanager
async def lifespan(app_: FastAPI):
    start_background_scheduler()
    yield
    shutdown_background_scheduler()


app = FastAPI(title="TWD FX Monitor API", version="0.1.0", lifespan=lifespan)


def _session():
    with make_session(settings()["database"]["url"]) as session:
        yield session


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/overview")
def overview(session=Depends(_session)):
    features = load_feature_frame(session)
    if features.empty:
        raise HTTPException(status_code=503, detail="No features available; run ingestion and feature build first")
    latest = features.iloc[-1]
    risk = latest_risk_snapshot(session)
    quality = guard_model_advice(session, summarize_data_quality(data_coverage_report(session)))
    exchange = None if quality.blocks_model_advice else recommend_exchange(
        default_inputs(), risk.twd_risk_score, risk.opportunity_score
    )
    bank = latest_bank_quote(session)
    predictions = _predictions(session)
    return _dataclass_safe({
        "current": {
            "date": str(latest["date"]),
            "usdtwd": latest.get("USDTWD_CLOSE"),
            "bank_spot_selling": bank.spot_selling if bank else None,
            "bank_name": bank.bank_name if bank else None,
            "bank_observed_at_utc": pd.to_datetime(bank.observed_at_utc, utc=True).isoformat() if bank else None,
            "bank_source": bank.source if bank else None,
            "bank_is_fallback": bank.bank_name != settings()["providers"]["bank"]["name"] if bank else None,
            "fx_source": latest.get("source"),
            "feature_set": "daily_v3",
            "change_1d": latest.get("USDTWD_RETURN_1D"),
            "change_5d": latest.get("USDTWD_RETURN_5D"),
            "change_20d": latest.get("USDTWD_RETURN_20D"),
        },
        "risk": _dataclass_safe(risk),
        "exchange": exchange.__dict__ if exchange else None,
        "data_quality": _dataclass_safe(quality),
        "advice_available": not quality.blocks_model_advice,
        "predictions": predictions,
        "family_reasons": _dataclass_safe(family_reasons(latest.to_dict())),
        "upcoming_events": upcoming_event_risk(session),
        "ai_interpretation": _dataclass_safe(latest_ai_interpretation(session)),
    })


@app.get("/features/latest")
def latest_features(session=Depends(_session)):
    features = load_feature_frame(session)
    if features.empty:
        raise HTTPException(status_code=503, detail="No features available")
    row = features.iloc[-1].to_dict()
    return {key: _json_safe(value) for key, value in row.items()}


@app.get("/data-quality")
def data_quality(session=Depends(_session)):
    report = data_coverage_report(session)
    report["summary"] = _dataclass_safe(summarize_data_quality(report))
    return report


@app.get("/model-health")
def model_health(session=Depends(_session)):
    return _dataclass_safe(latest_model_health_summary(session))


@app.get("/scheduler/status")
def scheduler_status_endpoint():
    return scheduler_status()


@app.get("/pipeline/status")
def pipeline_status_endpoint():
    return pipeline_status()


@app.get("/forward-validation")
def forward_validation(session=Depends(_session)):
    return forward_validation_summary(session)


@app.get("/backtests/strategy")
def strategy_backtest(
    start_year: int = Query(default=2023, ge=1900, le=2100),
    target_usd: float = Query(default=10_000, gt=0, le=1_000_000_000, allow_inf_nan=False),
    tune: bool = False,
    session=Depends(_session),
):
    if tune:
        return _dataclass_safe(walk_forward_tune_strategy(session, target_usd=target_usd, start_year=start_year))
    comparisons = compare_exchange_strategies(session, target_usd=target_usd, start_year=start_year)
    return _dataclass_safe(build_strategy_report(comparisons, target_usd=target_usd))


@app.get("/ai/latest")
def ai_latest(session=Depends(_session)):
    latest = latest_ai_interpretation(session)
    if latest is None:
        return {"enabled": False, "provider": "openai", "model": None}
    return _dataclass_safe(latest)


@app.get("/charts/usdtwd")
def usdtwd_chart(limit: int = Query(default=365, ge=1, le=5000), session=Depends(_session)):
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
    if not verified:
        raise HTTPException(status_code=401, detail="Invalid LINE signature")
    try:
        payload = await request.json()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON body") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("events", []), list):
        raise HTTPException(status_code=400, detail="Invalid webhook payload")
    user_ids = []
    for event in payload.get("events", []):
        if not isinstance(event, dict):
            raise HTTPException(status_code=400, detail="Invalid event")
        source = event.get("source") or {}
        if not isinstance(source, dict):
            raise HTTPException(status_code=400, detail="Invalid event source")
        if source.get("userId"):
            user_ids.append(source["userId"])
    return {"ok": True, "signature_verified": verified, "user_ids": user_ids}


def _predictions(session):
    rows = session.execute(
        select(Prediction).where(Prediction.model_version == "phase4_ensemble_v3").order_by(Prediction.observed_at_utc.desc())
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
    if isinstance(value, float) and not math.isfinite(value):
        return None
    try:
        if value != value:
            return None
    except TypeError:
        pass
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return value


def _dataclass_safe(value):
    if hasattr(value, "__dict__"):
        return {key: _dataclass_safe(item) for key, item in value.__dict__.items()}
    if isinstance(value, dict):
        return {key: _dataclass_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_dataclass_safe(item) for item in value]
    return _json_safe(value)
