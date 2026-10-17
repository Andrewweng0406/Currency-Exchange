"""Read-only timestamp audit; legacy database rows do not prove immutable forecasts."""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import settings
from app.database.schema import FxPrice, Prediction
from app.database.session import make_engine


def audit_rows(predictions, official_dates):
    dates = sorted(set(pd.to_datetime(official_dates, utc=True).normalize()))
    dates_set = set(dates)
    records = []
    for pred in predictions:
        origin = pd.to_datetime(pred["observed_at_utc"], utc=True).normalize()
        created = pd.to_datetime(pred.get("created_at_utc"), utc=True, errors="coerce")
        pos = bisect_right(dates, origin)
        next_close = dates[pos] + pd.Timedelta(hours=9) if pos < len(dates) else None
        if origin not in dates_set:
            status = "non_official_origin"
        elif pd.isna(created):
            status = "unknown_created_time"
        elif created < origin + pd.Timedelta(hours=9):
            status = "created_before_origin_close"
        elif next_close is not None and created >= next_close:
            status = "created_after_next_official_close"
        elif next_close is None and created >= origin + pd.offsets.BDay(1) + pd.Timedelta(hours=9):
            status = "created_after_conservative_deadline"
        else:
            status = "created_in_window_only"
        try:
            snapshot = json.loads(pred.get("input_snapshot") or "{}")
        except (ValueError, TypeError):
            snapshot = {}
        explicit = snapshot.get("issued_at_utc") if isinstance(snapshot, dict) else None
        records.append({"id": pred["id"], "model_version": pred["model_version"], "horizon": pred["horizon"],
                        "origin_utc": origin.isoformat(), "created_at_utc": None if pd.isna(created) else created.isoformat(),
                        "next_official_close_available_utc": next_close.isoformat() if next_close is not None else None,
                        "temporal_status": status,
                        "explicit_issued_at_present": not pd.isna(pd.to_datetime(explicit, utc=True, errors="coerce"))})
    groups = []
    for version, horizon in sorted({(r["model_version"], r["horizon"]) for r in records}):
        rows = [r for r in records if (r["model_version"], r["horizon"]) == (version, horizon)]
        groups.append({"model_version": version, "horizon": horizon, "rows": len(rows),
                       "explicit_issued_at_rows": sum(r["explicit_issued_at_present"] for r in rows),
                       "temporal_status_counts": dict(Counter(r["temporal_status"] for r in rows))})
    return {"groups": groups, "records": records,
            "limitation": "Created time is only a necessary timestamp check, not proof that probabilities were never overwritten. Official close availability uses the conservative 09:00 UTC proxy. No legacy performance is certified by this audit."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    engine = make_engine(settings()["database"]["url"])
    with engine.connect() as connection:
        if engine.dialect.name == "postgresql":
            connection.exec_driver_sql("SET TRANSACTION READ ONLY")
        elif engine.dialect.name == "sqlite":
            connection.exec_driver_sql("PRAGMA query_only = ON")
        with Session(bind=connection) as session:
            predictions = session.execute(select(Prediction.__table__).where(
                Prediction.source == "model_training_latest")).mappings().all()
            dates = session.scalars(select(FxPrice.observed_at_utc).where(
                FxPrice.pair == "USD/TWD", FxPrice.source == "cbc_taiwan", FxPrice.close > 0)).all()
            result = audit_rows(predictions, dates)
    result["audited_at_utc"] = datetime.now(timezone.utc).isoformat()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
