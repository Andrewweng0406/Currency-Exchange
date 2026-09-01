from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import select

from app.backtesting.dataset import add_forward_targets, load_feature_frame
from app.backtesting.metrics import classification_metrics, regression_metrics
from app.database.schema import ModelPerformance, Prediction
from app.database.upsert import upsert_rows


@dataclass(frozen=True)
class ModelHealthMetric:
    horizon: str
    window: str
    sample_count: int
    accuracy: float | None
    brier_score: float | None
    twd_depreciation_recall: float | None
    mae: float | None
    rmse: float | None
    observed_at_utc: str | None


@dataclass(frozen=True)
class ModelHealthSummary:
    status: str
    label_zh: str
    message_zh: str
    metrics: list[ModelHealthMetric]


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
            metric_values = {"sample_count": float(len(sample)), **cls.__dict__, **reg.__dict__}
            for metric, value in metric_values.items():
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


def latest_model_health_summary(session) -> ModelHealthSummary:
    rows = session.execute(
        select(ModelPerformance)
        .where(ModelPerformance.source == "model_monitoring")
        .where(ModelPerformance.model_version == "phase4_ensemble_v1")
        .order_by(ModelPerformance.observed_at_utc.desc())
    ).scalars().all()
    if not rows:
        return ModelHealthSummary(
            status="INSUFFICIENT_HISTORY",
            label_zh="追蹤資料不足",
            message_zh="目前還沒有足夠到期的每日預測可以評估準確度，需要持續累積。",
            metrics=[],
        )

    latest_by_group: dict[tuple[str, str], datetime] = {}
    for row in rows:
        key = (row.horizon, row.window or "")
        current = latest_by_group.get(key)
        if current is None or row.observed_at_utc > current:
            latest_by_group[key] = row.observed_at_utc

    metrics = []
    for horizon, window in sorted(latest_by_group, key=lambda item: (_horizon_order(item[0]), _window_order(item[1]))):
        observed_at = latest_by_group[(horizon, window)]
        group = [
            row
            for row in rows
            if row.horizon == horizon and (row.window or "") == window and row.observed_at_utc == observed_at
        ]
        values = {row.metric: row.value for row in group}
        metrics.append(
            ModelHealthMetric(
                horizon=horizon,
                window=window,
                sample_count=int(values.get("sample_count") or 0),
                accuracy=_clean(values.get("accuracy")),
                brier_score=_clean(values.get("brier_score")),
                twd_depreciation_recall=_clean(values.get("twd_depreciation_recall")),
                mae=_clean(values.get("mae")),
                rmse=_clean(values.get("rmse")),
                observed_at_utc=observed_at.isoformat() if observed_at else None,
            )
        )

    key_metrics = [item for item in metrics if item.window in {"30D", "90D"}]
    enough_history = any(item.sample_count >= 10 for item in key_metrics)
    if not enough_history:
        return ModelHealthSummary(
            status="INSUFFICIENT_HISTORY",
            label_zh="追蹤資料不足",
            message_zh="目前到期的每日預測樣本還太少，先累積資料，不急著判定模型好壞。",
            metrics=metrics,
        )
    recall_values = [item.twd_depreciation_recall for item in key_metrics if item.twd_depreciation_recall is not None]
    brier_values = [item.brier_score for item in key_metrics if item.brier_score is not None]
    if recall_values and min(recall_values) < 0.45:
        status = "WEAK"
        label = "需要觀察"
        message = "近期模型抓出台幣貶值的能力偏弱，換匯建議只能當提醒，不能單獨決策。"
    elif brier_values and sum(brier_values) / len(brier_values) > 0.28:
        status = "WEAK"
        label = "機率校準偏弱"
        message = "近期模型機率校準不理想，建議降低對百分比數字的信任。"
    else:
        status = "OK"
        label = "追蹤正常"
        message = "目前追蹤指標沒有明顯惡化，但仍需持續累積真實每日預測。"
    return ModelHealthSummary(status=status, label_zh=label, message_zh=message, metrics=metrics)


def _clean(value: float | None) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _horizon_order(value: str) -> int:
    return {"1d": 1, "5d": 5, "20d": 20}.get(value, 999)


def _window_order(value: str) -> int:
    try:
        return int(value.removesuffix("D"))
    except ValueError:
        return 999
