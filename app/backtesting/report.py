from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

import pandas as pd

from app.backtesting.dataset import add_forward_targets, load_feature_frame, modeling_columns
from app.backtesting.exchange_strategy import equal_tranche_exchange, fixed_day_exchange
from app.backtesting.splits import yearly_expanding_splits


def phase3_summary(session) -> dict[str, Any]:
    features = load_feature_frame(session)
    dataset = add_forward_targets(features)
    frame = dataset.frame
    splits = yearly_expanding_splits(frame.dropna(subset=["USDTWD_CLOSE"]), min_train_years=5)
    latest = frame.iloc[-1].to_dict() if not frame.empty else {}
    report: dict[str, Any] = {
        "feature_rows": len(frame),
        "feature_start": str(pd.Timestamp(frame["date"].min()).date()) if not frame.empty else None,
        "feature_end": str(pd.Timestamp(frame["date"].max()).date()) if not frame.empty else None,
        "modeling_columns": len(modeling_columns(frame)) if not frame.empty else 0,
        "walk_forward_splits": [
            {
                "train_start": str(split.train_start.date()),
                "train_end": str(split.train_end.date()),
                "test_start": str(split.test_start.date()),
                "test_end": str(split.test_end.date()),
                "train_rows": len(split.train_index),
                "test_rows": len(split.test_index),
            }
            for split in splits
        ],
        "latest_target_preview": _json_safe(
            {key: latest.get(key) for key in ["TARGET_RETURN_1D", "TARGET_RETURN_5D", "TARGET_RETURN_20D"] if key in latest}
        ),
    }
    completed = frame.dropna(subset=["TARGET_RETURN_20D"])
    if len(completed) >= 260:
        payment_date = pd.Timestamp(completed["date"].max(), tz="UTC") if pd.Timestamp(completed["date"].max()).tzinfo is None else pd.Timestamp(completed["date"].max()).tz_convert("UTC")
        window = completed.tail(90)
        fixed = fixed_day_exchange(window, target_usd=10_000, payment_date=payment_date)
        tranches = equal_tranche_exchange(window, target_usd=10_000, payment_date=payment_date)
        report["strategy_smoke_test"] = {
            "fixed_day_once": _json_safe(asdict(fixed)),
            "equal_tranches": _json_safe(asdict(tranches)),
            "note": "Smoke test only. Strategy C waits for real model predictions/risk scores in later phases.",
        }
    return report


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value
