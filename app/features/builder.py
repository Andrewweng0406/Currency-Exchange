from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.database.schema import Feature, ForeignFlow, FxPrice, MarketData
from app.database.upsert import upsert_rows
from app.features.indicators import atr, bollinger_position, pct_return, rolling_volatility, rsi, zscore


@dataclass(frozen=True)
class FeatureBuildResult:
    rows: int
    start_date: str | None
    end_date: str | None


def _date_only(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True).dt.tz_convert("UTC").dt.normalize()


def _clean_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, float):
        return value if isfinite(value) else None
    return value


def _json_record(row: pd.Series) -> str:
    return json.dumps({k: _clean_value(v) for k, v in row.items()}, sort_keys=True)


def _load_fx(session: Session) -> pd.DataFrame:
    rows = session.execute(
        select(FxPrice.observed_at_utc, FxPrice.source, FxPrice.open, FxPrice.high, FxPrice.low, FxPrice.close, FxPrice.volume)
        .where(FxPrice.pair == "USD/TWD")
        .order_by(FxPrice.observed_at_utc)
    ).all()
    df = pd.DataFrame(rows, columns=["date", "source", "open", "high", "low", "close", "volume"])
    if df.empty:
        return df
    df["date"] = _date_only(df["date"])
    df["priority"] = df["source"].map({"yahoo_finance": 0, "cbc_taiwan": 1}).fillna(9)
    return df.sort_values(["date", "priority"]).drop_duplicates("date", keep="first").drop(columns=["priority"])


def _load_market(session: Session) -> pd.DataFrame:
    rows = session.execute(
        select(
            MarketData.observed_at_utc,
            MarketData.symbol,
            MarketData.open,
            MarketData.high,
            MarketData.low,
            MarketData.close,
            MarketData.volume,
            MarketData.source,
        ).order_by(MarketData.observed_at_utc)
    ).all()
    df = pd.DataFrame(rows, columns=["date", "symbol", "open", "high", "low", "close", "volume", "source"])
    if df.empty:
        return df
    df["date"] = _date_only(df["date"])
    df["priority"] = df["source"].map({"fred_csv": 0, "treasury_yield_curve_xml": 1, "twse": 1, "yahoo_finance": 2}).fillna(9)
    return df.sort_values(["symbol", "date", "priority"]).drop_duplicates(["symbol", "date"], keep="first").drop(columns=["priority"])


def _load_foreign_flow(session: Session) -> pd.DataFrame:
    rows = session.execute(
        select(ForeignFlow.observed_at_utc, ForeignFlow.foreign_net_buy_sell_shares).order_by(ForeignFlow.observed_at_utc)
    ).all()
    df = pd.DataFrame(rows, columns=["date", "foreign_flow"])
    if df.empty:
        return df
    df["date"] = _date_only(df["date"])
    return df.groupby("date", as_index=False)["foreign_flow"].sum()


def _market_series_with_missing_flag(
    values: pd.Series,
    target_index: pd.DatetimeIndex,
    max_stale_days: int,
) -> tuple[pd.Series, pd.Series]:
    values = values.dropna().sort_index()
    aligned = values.reindex(target_index).ffill()
    observed_at = pd.Series(values.index, index=values.index).reindex(target_index).ffill()
    stale_days = (pd.Series(target_index, index=target_index) - observed_at).dt.days
    missing = observed_at.isna() | (stale_days > max_stale_days)
    return aligned.mask(missing), missing.astype(int)


def build_feature_frame(session: Session) -> pd.DataFrame:
    fx = _load_fx(session)
    if fx.empty:
        return pd.DataFrame()
    app_settings = settings()
    max_market_stale_days = int(app_settings.get("features", {}).get("max_market_stale_days", 7))

    frame = fx.set_index("date").sort_index()
    frame = frame.rename(
        columns={
            "open": "USDTWD_OPEN",
            "high": "USDTWD_HIGH",
            "low": "USDTWD_LOW",
            "close": "USDTWD_CLOSE",
            "volume": "USDTWD_VOLUME",
        }
    )
    close = frame["USDTWD_CLOSE"]
    high = frame["USDTWD_HIGH"].fillna(close)
    low = frame["USDTWD_LOW"].fillna(close)
    derived: dict[str, Any] = {}

    for window in [1, 5, 20]:
        derived[f"USDTWD_RETURN_{window}D"] = pct_return(close, window)
        derived[f"USDTWD_ROLLING_HIGH_{window}D"] = close.rolling(window).max()
        derived[f"USDTWD_ROLLING_LOW_{window}D"] = close.rolling(window).min()
    for window in [20, 60, 120, 252]:
        sma = close.rolling(window).mean()
        derived[f"USDTWD_SMA_{window}D"] = sma
        derived[f"USDTWD_DISTANCE_SMA_{window}D"] = close / sma - 1
    derived["USDTWD_VOLATILITY_20D"] = rolling_volatility(close, 20)
    derived["USDTWD_RSI_14D"] = rsi(close, 14)
    derived["USDTWD_ATR_14D"] = atr(high, low, close, 14)
    derived["USDTWD_BOLLINGER_POSITION_20D"] = bollinger_position(close, 20)
    derived["USDTWD_MOMENTUM_10D"] = close - close.shift(10)
    derived["USDTWD_RATE_OF_CHANGE_20D"] = pct_return(close, 20)

    market = _load_market(session)
    if not market.empty:
        close_pivot = market.pivot(index="date", columns="symbol", values="close").sort_index()
        volume_pivot = market.pivot(index="date", columns="symbol", values="volume").sort_index()
        mapping = {
            "DXY": "DXY",
            "BROAD_USD_INDEX": "BROAD_USD_INDEX",
            "US_2Y": "US2Y",
            "US_10Y": "US10Y",
            "VIX": "VIX",
            "SP500": "SP500",
            "NASDAQ": "NASDAQ",
            "USD_CNH": "CNH",
            "USD_CNY": "CNY",
            "USD_KRW": "KRW",
            "USD_JPY": "JPY",
            "TAIEX": "TAIEX",
            "2330.TW": "TSMC",
            "TSM_ADR": "TSM_ADR",
        }
        for symbol, prefix in mapping.items():
            if symbol not in close_pivot:
                derived[f"{prefix}_DATA_MISSING"] = 1
                continue
            series, missing = _market_series_with_missing_flag(
                close_pivot[symbol],
                frame.index,
                max_stale_days=max_market_stale_days,
            )
            derived[f"{prefix}_CLOSE"] = series
            for window in [1, 5, 20]:
                if prefix in {"US2Y", "US10Y", "VIX"}:
                    derived[f"{prefix}_CHANGE_{window}D"] = series.diff(window)
                else:
                    derived[f"{prefix}_RETURN_{window}D"] = pct_return(series, window)
            derived[f"{prefix}_VOLATILITY_20D"] = rolling_volatility(series, 20)
            derived[f"{prefix}_DATA_MISSING"] = missing
        china_proxy = _china_fx_proxy(derived)
        if china_proxy is not None:
            derived["CHINA_FX_PROXY_CLOSE"] = china_proxy
            for window in [1, 5, 20]:
                derived[f"CHINA_FX_PROXY_RETURN_{window}D"] = pct_return(china_proxy, window)
            derived["CHINA_FX_PROXY_VOLATILITY_20D"] = rolling_volatility(china_proxy, 20)
            derived["CHINA_FX_PROXY_DATA_MISSING"] = china_proxy.isna().astype(int)
        else:
            derived["CHINA_FX_PROXY_DATA_MISSING"] = 1
        if {"US_2Y", "US_10Y"}.issubset(close_pivot.columns):
            us2y, _ = _market_series_with_missing_flag(close_pivot["US_2Y"], frame.index, max_market_stale_days)
            us10y, _ = _market_series_with_missing_flag(close_pivot["US_10Y"], frame.index, max_market_stale_days)
            derived["US_2S10S_SPREAD"] = us10y - us2y
        if "2330.TW" in volume_pivot:
            tsmc_volume, _ = _market_series_with_missing_flag(volume_pivot["2330.TW"], frame.index, max_market_stale_days)
            derived["TSMC_VOLUME_ZSCORE"] = zscore(tsmc_volume, 252)

    foreign = _load_foreign_flow(session)
    if not foreign.empty:
        flow = foreign.set_index("date")["foreign_flow"].reindex(frame.index).fillna(0)
        derived["FOREIGN_FLOW_1D"] = flow.rolling(1).sum()
        derived["FOREIGN_FLOW_3D"] = flow.rolling(3).sum()
        derived["FOREIGN_FLOW_5D"] = flow.rolling(5).sum()
        derived["FOREIGN_FLOW_20D"] = flow.rolling(20).sum()
        derived["FOREIGN_FLOW_ZSCORE"] = zscore(flow, 252)
        derived["FOREIGN_FLOW_DATA_MISSING"] = foreign.set_index("date")["foreign_flow"].reindex(frame.index).isna().astype(int)
    else:
        derived["FOREIGN_FLOW_DATA_MISSING"] = 1

    frame = pd.concat([frame, pd.DataFrame(derived, index=frame.index)], axis=1)
    frame["DATA_COMPLETENESS"] = 1 - frame.filter(like="_DATA_MISSING").mean(axis=1).fillna(0)
    return frame.reset_index(names="date")


def _china_fx_proxy(derived: dict[str, Any]) -> pd.Series | None:
    cny = derived.get("CNY_CLOSE")
    cnh = derived.get("CNH_CLOSE")
    if cny is not None and cnh is not None:
        return cny.combine_first(cnh)
    if cny is not None:
        return cny
    if cnh is not None:
        return cnh
    return None


def persist_features(session: Session, feature_set: str = "daily_v1") -> FeatureBuildResult:
    df = build_feature_frame(session)
    if df.empty:
        return FeatureBuildResult(rows=0, start_date=None, end_date=None)
    now = datetime.now(timezone.utc)
    rows = [
        {
            "observed_at_utc": row["date"],
            "source": "feature_builder",
            "feature_set": feature_set,
            "values_json": _json_record(row.drop(labels=["date"])),
            "created_at_utc": now,
        }
        for _, row in df.iterrows()
    ]
    count = upsert_rows(session, Feature, rows, ("feature_set", "observed_at_utc", "source"))
    return FeatureBuildResult(rows=count, start_date=str(df["date"].min().date()), end_date=str(df["date"].max().date()))
