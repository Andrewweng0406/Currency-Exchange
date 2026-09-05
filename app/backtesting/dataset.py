from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.schema import Feature


@dataclass(frozen=True)
class BacktestDataset:
    frame: pd.DataFrame
    horizons: tuple[int, ...]


def load_feature_frame(session: Session, feature_set: str = "daily_v1") -> pd.DataFrame:
    rows = session.execute(
        select(Feature.observed_at_utc, Feature.values_json)
        .where(Feature.feature_set == feature_set)
        .order_by(Feature.observed_at_utc)
    ).all()
    records = []
    for observed_at, values_json in rows:
        item = json.loads(values_json)
        item["date"] = pd.Timestamp(observed_at).tz_localize("UTC") if pd.Timestamp(observed_at).tzinfo is None else pd.Timestamp(observed_at).tz_convert("UTC")
        records.append(item)
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records).sort_values("date").drop_duplicates("date", keep="last")
    return frame.reset_index(drop=True)


def add_forward_targets(frame: pd.DataFrame, horizons: tuple[int, ...] = (1, 5, 20)) -> BacktestDataset:
    if frame.empty:
        return BacktestDataset(frame=frame.copy(), horizons=horizons)
    if "USDTWD_CLOSE" not in frame:
        raise ValueError("USDTWD_CLOSE is required to create forward targets")
    output = frame.copy()
    close = pd.to_numeric(output["USDTWD_CLOSE"], errors="coerce")
    for horizon in horizons:
        future_return = close.shift(-horizon) / close - 1
        output[f"TARGET_RETURN_{horizon}D"] = future_return
        output[f"TARGET_UP_{horizon}D"] = (future_return > 0).astype("Int64")
        output[f"TARGET_SHOCK_UP_0_5PCT_{horizon}D"] = (future_return > 0.005).astype("Int64")
        output[f"TARGET_SHOCK_UP_1PCT_{horizon}D"] = (future_return > 0.01).astype("Int64")
        output[f"TARGET_SHOCK_UP_1_5PCT_{horizon}D"] = (future_return > 0.015).astype("Int64")
        output[f"TARGET_SHOCK_UP_2PCT_{horizon}D"] = (future_return > 0.02).astype("Int64")
    return BacktestDataset(frame=output, horizons=horizons)


def modeling_columns(frame: pd.DataFrame) -> list[str]:
    blocked_prefixes = ("TARGET_",)
    blocked = {"date"}
    columns = []
    for col in frame.columns:
        if col in blocked or col.startswith(blocked_prefixes):
            continue
        if _is_supplemental_china_fx_column(col):
            continue
        if pd.api.types.is_numeric_dtype(frame[col]):
            columns.append(col)
    return columns


def _is_supplemental_china_fx_column(column: str) -> bool:
    upper = column.upper()
    if upper.startswith("CHINA_FX_PROXY"):
        return False
    return upper.startswith(("CNH_", "CNY_"))
