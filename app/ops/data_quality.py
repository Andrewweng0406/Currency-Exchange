from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from sqlalchemy import func, select

from app.backtesting.dataset import load_feature_frame
from app.database.schema import BankRate, Feature, ForeignFlow, FxPrice, MarketData, Prediction


@dataclass(frozen=True)
class CoverageCheck:
    dataset: str
    rows: int
    start_utc: str | None
    end_utc: str | None
    age_days: float | None
    status: str
    limitations: list[str]


@dataclass(frozen=True)
class FeatureQualityCheck:
    feature: str
    missing_ratio: float
    latest_missing: bool
    status: str


def data_coverage_report(session) -> dict[str, Any]:
    coverage = [
        _coverage(session, FxPrice, "fx_prices:USD/TWD"),
        _coverage(session, BankRate, "bank_rates:USD"),
        _coverage(session, MarketData, "market_data"),
        _coverage(session, ForeignFlow, "foreign_flows:TWSE"),
        _coverage(session, Feature, "features:daily_v1", extra_filter=(Feature.feature_set == "daily_v1")),
        _coverage(session, Prediction, "predictions"),
    ]
    features = load_feature_frame(session)
    quality = _feature_quality(features)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "coverage": [item.__dict__ for item in coverage],
        "feature_quality": [item.__dict__ for item in quality],
        "overall_status": _overall_status(coverage, quality),
    }


def _coverage(session, model, label: str, extra_filter: Any = None) -> CoverageCheck:
    query = select(func.count(), func.min(model.observed_at_utc), func.max(model.observed_at_utc)).select_from(model)
    if extra_filter is not None:
        query = query.where(extra_filter)
    rows, start, end = session.execute(query).one()
    rows = rows or 0
    age = None
    if end is not None:
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - end).total_seconds() / 86400
        age = max(age, 0)
    limitations = []
    status = "OK"
    if rows == 0:
        status = "FAIL"
        limitations.append("no rows available")
    elif age is not None and age > 7:
        status = "STALE"
        limitations.append("latest observation is older than 7 days")
    if "bank_rates" in label and rows < 2:
        limitations.append("only current/fallback bank quotes are available; family bank should be configured")
    if "foreign_flows" in label and rows < 500:
        limitations.append("foreign-flow history is less than two trading years")
    return CoverageCheck(
        dataset=label,
        rows=int(rows),
        start_utc=start.isoformat() if start else None,
        end_utc=end.isoformat() if end else None,
        age_days=round(age, 3) if age is not None else None,
        status=status,
        limitations=limitations,
    )


def _feature_quality(features: pd.DataFrame) -> list[FeatureQualityCheck]:
    if features.empty:
        return []
    critical = [
        "USDTWD_CLOSE",
        "DXY_CLOSE",
        "US2Y_CLOSE",
        "US10Y_CLOSE",
        "VIX_CLOSE",
        "SP500_CLOSE",
        "NASDAQ_CLOSE",
        "KRW_CLOSE",
        "JPY_CLOSE",
        "CNH_CLOSE",
        "TAIEX_CLOSE",
        "TSMC_CLOSE",
        "FOREIGN_FLOW_5D",
        "DATA_COMPLETENESS",
    ]
    checks = []
    for col in critical:
        if col not in features:
            checks.append(FeatureQualityCheck(feature=col, missing_ratio=1.0, latest_missing=True, status="MISSING_COLUMN"))
            continue
        series = features[col]
        missing_ratio = float(series.isna().mean())
        latest_missing = bool(pd.isna(series.iloc[-1]))
        if latest_missing or missing_ratio > 0.80:
            status = "POOR"
        elif missing_ratio > 0.35:
            status = "LIMITED"
        else:
            status = "OK"
        checks.append(
            FeatureQualityCheck(
                feature=col,
                missing_ratio=round(missing_ratio, 4),
                latest_missing=latest_missing,
                status=status,
            )
        )
    return checks


def _overall_status(coverage: list[CoverageCheck], quality: list[FeatureQualityCheck]) -> str:
    if any(item.status == "FAIL" for item in coverage):
        return "FAIL"
    if any(item.status == "STALE" for item in coverage):
        return "STALE"
    if any(item.status in {"POOR", "MISSING_COLUMN"} for item in quality):
        return "DEGRADED"
    if any(item.status == "LIMITED" for item in quality):
        return "LIMITED"
    return "OK"
