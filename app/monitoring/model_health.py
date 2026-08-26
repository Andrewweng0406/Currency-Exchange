from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import select

from app.backtesting.dataset import add_forward_targets, load_feature_frame
from app.backtesting.metrics import classification_metrics, regression_metrics
from app.database.schema import ModelPerformance, Prediction
from app.database.upsert import upsert_rows


def evaluate_matured_predictions(session, windows: tuple[int, ...] = (7, 30, 90)) -> int:
    features = add_forward_targets(load_feature_frame(session), horizons=(1, 5, 20)).frame
    predictions = session.execute(select(Prediction).where(Prediction.model_version == "phase4_ensemble_v1")).scalars().all()
    if features.empty or not predictions:
        return 0
    by_date = features.set_index(pd.to_datetime(features["date"], utc=True))
    now = datetime.now(timezone.utc)
    rows = []
    for horizon in ["1d", "5d", "20d"]:
        days = int(horizon.removesuffix("d"))
        preds = [p for p in predictions if p.horizon == horizon and p.prob_up is not None]
        records = []
        for pred in preds:
            date = pd.Timestamp(pred.observed_at_utc, tz="UTC") if pd.Timestamp(pred.observed_at_utc).tzinfo is None else pd.Timestamp(pred.observed_at_utc).tz_convert("UTC")
            if date not in by_date.index:
                continue
            target_return = by_date.loc[date].get(f"TARGET_RETURN_{days}D")
            target_up = by_date.loc[date].get(f"TARGET_UP_{days}D")
            if pd.isna(target_return) or pd.isna(target_up):
                continue
            records.append({"date": date, "target_up": target_up, "target_return": target_return, "prob_up": pred.prob_up, "expected_return": pred.expected_return})
        data = pd.DataFrame(records).sort_values("date")
        for window in windows:
            sample = data.tail(window)
            if sample.empty:
                continue
            cls = classification_metrics(sample["target_up"], sample["prob_up"])
            reg = regression_metrics(sample["target_return"], sample["expected_return"])
            for metric, value in {**cls.__dict__, **reg.__dict__}.items():
                rows.append(
                    {
                        "observed_at_utc": now,
                        "source": "model_monitoring",
                        "model_version": "phase4_ensemble_v1",
                        "horizon": horizon,
                        "metric": metric,
                        "value": value,
                        "window": f"{window}D",
                    }
                )
    if rows:
        upsert_rows(session, ModelPerformance, rows, ("model_version", "horizon", "metric", "window", "observed_at_utc", "source"))
    return len(rows)
