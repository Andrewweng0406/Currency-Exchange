from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select

from app.config import settings
from app.database.schema import AIInterpretation


@dataclass(frozen=True)
class AIInterpretationResult:
    enabled: bool
    provider: str
    model: str | None
    macro_sentiment: str | None = None
    fed_tone: str | None = None
    risk_off_level: str | None = None
    confidence_adjustment: float = 0.0
    summary_zh_tw: str | None = None
    raw_response: str | None = None
    error: str | None = None


def latest_ai_interpretation(session) -> AIInterpretationResult | None:
    row = session.execute(select(AIInterpretation).order_by(AIInterpretation.observed_at_utc.desc()).limit(1)).scalar_one_or_none()
    if row is None:
        return None
    return AIInterpretationResult(
        enabled=True,
        provider=row.provider,
        model=row.model,
        macro_sentiment=row.macro_sentiment,
        fed_tone=row.fed_tone,
        risk_off_level=row.risk_off_level,
        confidence_adjustment=float(row.confidence_adjustment or 0),
        summary_zh_tw=row.summary_zh_tw,
        raw_response=row.raw_response,
    )


def interpret_risk_context(
    risk: Any,
    predictions: dict[str, Any],
    current: dict[str, Any],
    upcoming_events: list[dict[str, Any]] | None = None,
) -> AIInterpretationResult:
    cfg = settings().get("ai", {}).get("openai", {})
    model = str(cfg.get("model") or "gpt-5.6")
    if not _truthy(cfg.get("enabled")):
        return AIInterpretationResult(enabled=False, provider="openai", model=model, error="OpenAI interpreter disabled")
    if not os.getenv("OPENAI_API_KEY"):
        return AIInterpretationResult(enabled=False, provider="openai", model=model, error="OPENAI_API_KEY missing")

    try:
        from openai import OpenAI
    except ImportError as exc:
        return AIInterpretationResult(enabled=False, provider="openai", model=model, error=f"openai package missing: {exc}")

    payload = _payload(risk, predictions, current, upcoming_events or [])
    prompt = _prompt(payload)
    try:
        client = OpenAI(timeout=float(cfg.get("timeout_seconds", 30)))
        response = client.responses.create(
            model=model,
            input=prompt,
            max_output_tokens=int(cfg.get("max_output_tokens", 700)),
        )
        raw = getattr(response, "output_text", "") or str(response)
        parsed = _parse_json(raw)
        return _result_from_json(model, parsed, raw)
    except Exception as exc:  # noqa: BLE001
        return AIInterpretationResult(enabled=True, provider="openai", model=model, error=str(exc))


def persist_ai_interpretation(session, result: AIInterpretationResult, observed_at: datetime | None = None) -> AIInterpretation | None:
    if not result.enabled or result.error:
        return None
    row = AIInterpretation(
        observed_at_utc=observed_at or datetime.now(timezone.utc),
        source="openai",
        provider=result.provider,
        model=result.model or "",
        macro_sentiment=result.macro_sentiment,
        fed_tone=result.fed_tone,
        risk_off_level=result.risk_off_level,
        confidence_adjustment=result.confidence_adjustment,
        summary_zh_tw=result.summary_zh_tw,
        raw_response=result.raw_response,
    )
    session.add(row)
    session.commit()
    return row


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _payload(risk: Any, predictions: dict[str, Any], current: dict[str, Any], upcoming_events: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "current": current,
        "risk": {
            "twd_risk_score": getattr(risk, "twd_risk_score", None),
            "opportunity_score": getattr(risk, "opportunity_score", None),
            "regime": getattr(risk, "regime", None),
            "confidence": getattr(risk, "confidence", None),
            "contributors": getattr(risk, "contributors", None),
            "tail_risk": getattr(getattr(risk, "tail_risk", None), "probabilities", None),
            "cbc_intervention_risk": getattr(getattr(risk, "cbc_intervention_risk", None), "level", None),
        },
        "predictions": predictions,
        "upcoming_events": upcoming_events[:5],
    }


def _prompt(payload: dict[str, Any]) -> str:
    return (
        "You are an FX risk interpretation layer for a Taiwanese student in the US. "
        "Do not make deterministic exchange-rate claims. Do not change model probabilities. "
        "Use only the supplied JSON. Return JSON only with exactly these keys: "
        "macro_sentiment, fed_tone, risk_off_level, confidence_adjustment, summary_zh_tw. "
        "Allowed macro_sentiment values: USD_BULLISH, USD_BEARISH, TWD_BULLISH, TWD_BEARISH, NEUTRAL, MIXED. "
        "Allowed fed_tone values: HAWKISH, DOVISH, MIXED, UNKNOWN. "
        "Allowed risk_off_level values: LOW, MEDIUM, HIGH. "
        "confidence_adjustment must be a number from -10 to 5 and should usually be 0 or negative when data is missing. "
        "summary_zh_tw must be Traditional Chinese, concise, and mention probability/risk rather than certainty.\n\n"
        f"Input JSON:\n{json.dumps(payload, ensure_ascii=False, default=str)}"
    )


def _parse_json(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").removeprefix("json").strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("OpenAI response did not contain JSON")
    return json.loads(raw[start : end + 1])


def _result_from_json(model: str, data: dict[str, Any], raw: str) -> AIInterpretationResult:
    adjustment = float(data.get("confidence_adjustment") or 0)
    adjustment = max(-10.0, min(5.0, adjustment))
    return AIInterpretationResult(
        enabled=True,
        provider="openai",
        model=model,
        macro_sentiment=_clean_choice(data.get("macro_sentiment")),
        fed_tone=_clean_choice(data.get("fed_tone")),
        risk_off_level=_clean_choice(data.get("risk_off_level")),
        confidence_adjustment=adjustment,
        summary_zh_tw=str(data.get("summary_zh_tw") or "").strip()[:1000],
        raw_response=raw,
    )


def _clean_choice(value: Any) -> str | None:
    if value is None:
        return None
    return str(value).strip().upper()[:40]
