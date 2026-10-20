from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import select

from app.backtesting.dataset import add_forward_targets, load_feature_frame
from app.backtesting.metrics import classification_metrics, regression_metrics
from app.backtesting.uncertainty import block_mean_interval
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
    effective_sample_count: int = 0
    skill_upper_vs_coin: float | None = None
    skill_upper_vs_prior: float | None = None


@dataclass(frozen=True)
class ModelHealthSummary:
    status: str
    label_zh: str
    message_zh: str
    metrics: list[ModelHealthMetric]


def evaluate_matured_predictions(session, windows: tuple[int, ...] = (7, 30, 90, 365)) -> int:
    features = add_forward_targets(load_feature_frame(session), horizons=(1, 5, 20)).frame
    predictions = session.execute(select(Prediction).where(Prediction.model_version == "phase4_ensemble_v3")).scalars().all()
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
            try:
                snapshot = json.loads(pred.input_snapshot or "{}")
            except (ValueError, TypeError):
                continue
            if not isinstance(snapshot, dict):
                continue
            issued = pd.to_datetime(snapshot.get("issued_at_utc"), utc=True, errors="coerce")
            prior = pd.to_numeric(snapshot.get("training_up_rate"), errors="coerce")
            label_end = by_date.loc[date].get(f"TARGET_END_{days}D")
            # Validate issuance provenance. Date-only/backdated records cannot
            # establish real live forecasting skill after their outcomes are known.
            if pd.isna(issued) or prior is None or not math.isfinite(prior) or not 0 <= prior <= 1:
                continue
            if not date + pd.Timedelta(hours=9) <= issued < by_date.loc[date]["TARGET_END_1D"] + pd.Timedelta(hours=9):
                continue
            records.append({"date": date, "label_end": label_end, "target_up": target_up,
                            "target_return": target_return, "prob_up": pred.prob_up,
                            "prior": prior, "expected_return": pred.expected_return})
        if not records:
            continue
        data = pd.DataFrame(records).sort_values("date")
        for window in windows:
            sample = data[data["date"] >= pd.Timestamp(now) - pd.Timedelta(days=window)]
            if sample.empty:
                continue
            cls = classification_metrics(sample["target_up"], sample["prob_up"])
            reg = regression_metrics(sample["target_return"], sample["expected_return"])
            effective_count = 0
            next_origin = None
            for observation in sample.itertuples():
                if next_origin is None or observation.date >= next_origin:
                    effective_count += 1
                    next_origin = observation.label_end
            loss = (sample["prob_up"] - sample["target_up"]) ** 2
            upper_coin = upper_prior = float("nan")
            block_size = max(20, days * 2)
            if len(sample) >= 2 * block_size:
                upper_coin = block_mean_interval(loss - 0.25, block_size)[1]
                upper_prior = block_mean_interval(loss - (sample["prior"] - sample["target_up"]) ** 2, block_size)[1]
            metric_values = {"sample_count": float(len(sample)), "effective_sample_count": float(effective_count),
                             "skill_upper_vs_coin": upper_coin, "skill_upper_vs_prior": upper_prior,
                             **cls.__dict__, **reg.__dict__}
            for metric, value in metric_values.items():
                rows.append(
                    {
                        "observed_at_utc": now,
                        "source": "model_monitoring",
                        "model_version": "phase4_ensemble_v3",
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
        .where(ModelPerformance.model_version == "phase4_ensemble_v3")
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
                effective_sample_count=int(values.get("effective_sample_count") or 0),
                skill_upper_vs_coin=_clean(values.get("skill_upper_vs_coin")),
                skill_upper_vs_prior=_clean(values.get("skill_upper_vs_prior")),
                accuracy=_clean(values.get("accuracy")),
                brier_score=_clean(values.get("brier_score")),
                twd_depreciation_recall=_clean(values.get("twd_depreciation_recall")),
                mae=_clean(values.get("mae")),
                rmse=_clean(values.get("rmse")),
                observed_at_utc=observed_at.isoformat() if observed_at else None,
            )
        )

    # All model horizons affect risk scoring. Require current, baseline-relative
    # evidence for every horizon, rather than one group with ten overlapping rows.
    key_metrics = [item for item in metrics if item.window == "365D"]
    enough = []
    for horizon in ("1d", "5d", "20d"):
        group = next((item for item in key_metrics if item.horizon == horizon), None)
        required_rows = max(30, 2 * max(20, int(horizon.removesuffix("d")) * 2))
        if group is None or group.sample_count < required_rows or group.effective_sample_count < 10:
            continue
        age = pd.Timestamp.now(tz="UTC") - pd.to_datetime(group.observed_at_utc, utc=True)
        if not pd.Timedelta(0) <= age <= pd.Timedelta(days=7):
            continue
        if group.skill_upper_vs_coin is not None and group.skill_upper_vs_prior is not None:
            enough.append(group)
    if len(enough) != 3:
        return ModelHealthSummary(
            status="INSUFFICIENT_HISTORY", label_zh="追蹤資料不足",
            message_zh="三個期間尚未都有足夠、未重疊且及時的真實預測及基準比較，暫停模型選時。", metrics=metrics,
        )
    if any(item.skill_upper_vs_coin >= 0 or item.skill_upper_vs_prior >= 0 for item in enough):
        status, label = "WEAK", "尚未勝過基準"
        message = "真實預測尚未在區塊不確定區間內勝過固定50%與當時歷史比例基準，暫停模型選時。"
    else:
        status, label = "OK", "追蹤通過基準檢查"
        message = "三個期間的真實預測目前通過基準與資料時效檢查；仍不代表實際換匯省錢。"

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
