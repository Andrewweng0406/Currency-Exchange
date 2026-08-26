from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import select, func

from app.database.schema import BankRate, Feature, FxPrice, MarketData, Prediction


@dataclass(frozen=True)
class ReadinessCheck:
    name: str
    ok: bool
    detail: str


def run_readiness_checks(session) -> list[ReadinessCheck]:
    checks = [
        _env_present("DATABASE_URL", required=False, default_ok=True),
        _env_present("LINE_CHANNEL_ACCESS_TOKEN", required=False, default_ok=False),
        _env_present("LINE_USER_ID", required=False, default_ok=False),
        _table_count(session, FxPrice, "fx_prices", minimum=1000),
        _table_count(session, MarketData, "market_data", minimum=1000),
        _table_count(session, Feature, "features", minimum=1000),
        _table_count(session, Prediction, "predictions", minimum=3),
        _table_count(session, BankRate, "bank_rates", minimum=1),
        _freshness(session, Feature, "features", max_age_days=5),
    ]
    return checks


def _env_present(name: str, required: bool, default_ok: bool) -> ReadinessCheck:
    value = os.getenv(name)
    ok = bool(value) or (not required and default_ok)
    if value:
        detail = "set"
    elif required:
        detail = "missing required env var"
    else:
        detail = "missing optional env var"
    return ReadinessCheck(name=f"env:{name}", ok=ok, detail=detail)


def _table_count(session, model, label: str, minimum: int) -> ReadinessCheck:
    count = session.scalar(select(func.count()).select_from(model)) or 0
    return ReadinessCheck(name=f"table:{label}", ok=count >= minimum, detail=f"{count} rows, minimum {minimum}")


def _freshness(session, model, label: str, max_age_days: int) -> ReadinessCheck:
    latest = session.scalar(select(func.max(model.observed_at_utc)))
    if latest is None:
        return ReadinessCheck(name=f"freshness:{label}", ok=False, detail="no rows")
    if latest.tzinfo is None:
        latest = latest.replace(tzinfo=timezone.utc)
    age_days = (datetime.now(timezone.utc) - latest).total_seconds() / 86400
    return ReadinessCheck(name=f"freshness:{label}", ok=age_days <= max_age_days, detail=f"latest {latest.isoformat()}, age {age_days:.1f} days")
