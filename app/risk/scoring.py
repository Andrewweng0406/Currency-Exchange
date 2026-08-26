from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from sqlalchemy import select

from app.backtesting.dataset import load_feature_frame
from app.config import risk_policy
from app.database.schema import Prediction
from app.database.upsert import upsert_rows


@dataclass(frozen=True)
class RiskSnapshot:
    timestamp_utc: str
    twd_risk_score: int
    opportunity_score: int
    regime: list[str]
    confidence: float
    contributors: list[dict[str, Any]]


def latest_risk_snapshot(session) -> RiskSnapshot:
    features = load_feature_frame(session)
    if features.empty:
        raise ValueError("No features available")
    latest = features.iloc[-1]
    predictions = _latest_predictions(session)
    policy = risk_policy()
    components = _risk_components(latest, predictions)
    weights = policy["risk_score"]["weights"]
    score = 0.0
    contributors = []
    for name, value in components.items():
        weight = float(weights.get(name, 0))
        contribution = value * weight
        score += contribution
        contributors.append({"name": name, "value": round(value, 4), "weight": weight, "contribution": round(contribution, 4)})
    score_int = int(round(max(0, min(100, score))))
    confidence = _confidence(latest, predictions)
    opportunity = opportunity_score(features, policy)
    snapshot = RiskSnapshot(
        timestamp_utc=pd.Timestamp(latest["date"]).isoformat(),
        twd_risk_score=score_int,
        opportunity_score=opportunity,
        regime=detect_regime(latest),
        confidence=confidence,
        contributors=sorted(contributors, key=lambda item: abs(item["contribution"]), reverse=True)[:5],
    )
    _apply_risk_to_latest_predictions(session, predictions, score_int)
    return snapshot


def opportunity_score(features: pd.DataFrame, policy: dict[str, Any] | None = None) -> int:
    policy = policy or risk_policy()
    close = pd.to_numeric(features["USDTWD_CLOSE"], errors="coerce")
    latest = close.iloc[-1]
    score = 0.0
    for name, weight in policy["opportunity_score"]["weights"].items():
        window = int(name.split("_")[1].removesuffix("d"))
        sample = close.tail(window).dropna()
        if sample.empty:
            percentile = 0.5
        else:
            percentile = float((sample <= latest).mean())
        favorable = 1 - percentile
        score += favorable * float(weight) * 100
    return int(round(max(0, min(100, score))))


def detect_regime(row: pd.Series) -> list[str]:
    regimes = []
    dxy_20d = float(row.get("DXY_RETURN_20D") or row.get("BROAD_USD_INDEX_RETURN_20D") or 0)
    usdtwd_vol = float(row.get("USDTWD_VOLATILITY_20D") or 0)
    vix_change = float(row.get("VIX_CHANGE_5D") or 0)
    sp500_return = float(row.get("SP500_RETURN_5D") or 0)
    usdtwd_20d = float(row.get("USDTWD_RETURN_20D") or 0)
    asia = _asia_pressure(row)
    if vix_change > 3 or sp500_return < -0.03:
        regimes.append("RISK_OFF")
    else:
        regimes.append("RISK_ON")
    regimes.append("USD_STRONG" if dxy_20d > 0 else "USD_WEAK")
    regimes.append("HIGH_VOL" if usdtwd_vol > 0.004 else "LOW_VOL")
    if abs(usdtwd_20d - asia / 1000) > 0.02:
        regimes.append("TWD_IDIOSYNCRATIC")
    return regimes


def _latest_predictions(session) -> dict[str, Prediction]:
    rows = session.execute(
        select(Prediction)
        .where(Prediction.model_version == "phase4_ensemble_v1")
        .order_by(Prediction.observed_at_utc.desc())
    ).scalars().all()
    out = {}
    for row in rows:
        out.setdefault(row.horizon, row)
    return out


def _risk_components(row: pd.Series, predictions: dict[str, Prediction]) -> dict[str, float]:
    prob_5d = predictions.get("5d").prob_up if predictions.get("5d") else 0.5
    prob_20d = predictions.get("20d").prob_up if predictions.get("20d") else 0.5
    return {
        "prediction_5d": float(prob_5d) * 100,
        "prediction_20d": float(prob_20d) * 100,
        "recent_usdtwd_momentum": _scaled_centered(row.get("USDTWD_RETURN_5D"), scale=0.015),
        "dxy_momentum": _scaled_centered(row.get("DXY_RETURN_5D") or row.get("BROAD_USD_INDEX_RETURN_5D"), scale=0.015),
        "rates_pressure": _scaled_centered(row.get("US2Y_CHANGE_5D"), scale=0.15),
        "asia_fx_pressure": _asia_pressure(row),
        "global_risk_off": _global_risk_off(row),
        "data_quality_penalty": 100 - float(row.get("DATA_COMPLETENESS", 0.5) or 0.5) * 100,
    }


def _scaled_centered(value: Any, scale: float) -> float:
    value = float(value or 0)
    return max(0, min(100, 50 + 50 * value / scale))


def _asia_pressure(row: pd.Series) -> float:
    values = [row.get("CNH_RETURN_5D"), row.get("KRW_RETURN_5D"), row.get("JPY_RETURN_5D")]
    clean = [float(v) for v in values if v is not None and not pd.isna(v)]
    if not clean:
        return 50.0
    return max(0, min(100, 50 + 50 * (sum(clean) / len(clean)) / 0.02))


def _global_risk_off(row: pd.Series) -> float:
    vix = float(row.get("VIX_CHANGE_5D") or 0)
    sp = float(row.get("SP500_RETURN_5D") or 0)
    nasdaq = float(row.get("NASDAQ_RETURN_5D") or 0)
    pressure = 50 + max(0, vix) * 3 + max(0, -sp) * 600 + max(0, -nasdaq) * 400
    return max(0, min(100, pressure))


def _confidence(row: pd.Series, predictions: dict[str, Prediction]) -> float:
    model_conf = [p.confidence for p in predictions.values() if p.confidence is not None]
    base = sum(model_conf) / len(model_conf) if model_conf else 0.5
    completeness = float(row.get("DATA_COMPLETENESS", 0.5) or 0.5)
    volatility_penalty = min(0.25, float(row.get("USDTWD_VOLATILITY_20D") or 0) * 20)
    return round(max(0, min(1, 0.60 * base + 0.40 * completeness - volatility_penalty)), 4)


def _apply_risk_to_latest_predictions(session, predictions: dict[str, Prediction], risk_score: int) -> None:
    rows = []
    for prediction in predictions.values():
        rows.append(
            {
                "observed_at_utc": prediction.observed_at_utc,
                "source": prediction.source,
                "model_version": prediction.model_version,
                "horizon": prediction.horizon,
                "prob_up": prediction.prob_up,
                "prob_down": prediction.prob_down,
                "expected_return": prediction.expected_return,
                "confidence": prediction.confidence,
                "risk_score": risk_score,
                "input_snapshot": prediction.input_snapshot,
                "recommendation": prediction.recommendation,
            }
        )
    if rows:
        upsert_rows(session, Prediction, rows, ("model_version", "horizon", "observed_at_utc", "source"))


def snapshot_json(snapshot: RiskSnapshot) -> str:
    return json.dumps(snapshot.__dict__, ensure_ascii=False, sort_keys=True)
