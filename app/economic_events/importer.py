from __future__ import annotations

from datetime import timezone

import pandas as pd

from app.database.schema import EconomicEvent
from app.database.upsert import upsert_rows


REQUIRED_COLUMNS = {"event_name", "release_time", "previous", "forecast", "actual"}


def import_events_csv(session, path: str, source: str = "manual_csv") -> int:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    rows = []
    for item in df.to_dict("records"):
        release = pd.Timestamp(item["release_time"])
        if release.tzinfo is None:
            release = release.tz_localize("America/New_York")
        release_utc = release.tz_convert("UTC").to_pydatetime()
        actual = _num(item.get("actual"))
        forecast = _num(item.get("forecast"))
        surprise = actual - forecast if actual is not None and forecast is not None else None
        rows.append(
            {
                "observed_at_utc": release_utc,
                "source": source,
                "event_name": item["event_name"],
                "release_time_utc": release_utc,
                "previous": _num(item.get("previous")),
                "forecast": forecast,
                "actual": actual,
                "surprise": surprise,
                "surprise_zscore": None,
            }
        )
    return upsert_rows(session, EconomicEvent, rows, ("event_name", "release_time_utc", "source"))


def upcoming_event_risk(session, now_utc=None) -> list[dict]:
    from datetime import datetime, timedelta
    from sqlalchemy import select

    now_utc = now_utc or datetime.now(timezone.utc)
    rows = session.execute(
        select(EconomicEvent)
        .where(EconomicEvent.release_time_utc >= now_utc)
        .where(EconomicEvent.release_time_utc <= now_utc + timedelta(hours=24))
        .order_by(EconomicEvent.release_time_utc)
    ).scalars().all()
    return [{"event_name": row.event_name, "release_time_utc": row.release_time_utc.isoformat(), "risk_flag": "HIGH_IMPACT"} for row in rows]


def _num(value):
    parsed = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(parsed) else float(parsed)
