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


@dataclass(frozen=True)
class DataQualitySummary:
    status: str
    label_zh: str
    message_zh: str
    blocks_model_advice: bool
    issues: list[str]


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


def summarize_data_quality(report: dict[str, Any]) -> DataQualitySummary:
    coverage = report.get("coverage", [])
    quality = report.get("feature_quality", [])

    blocking_issues = []
    for item in coverage:
        dataset = str(item.get("dataset", ""))
        status = item.get("status")
        if status in {"FAIL", "STALE"} and _is_core_dataset(dataset):
            blocking_issues.append(f"{dataset} {status}")

    for item in quality:
        feature = str(item.get("feature", ""))
        if feature in _CORE_LATEST_FEATURES and item.get("latest_missing"):
            blocking_issues.append(f"{feature} 最新值缺失")

    if blocking_issues:
        return DataQualitySummary(
            status="BLOCKING",
            label_zh="核心資料異常",
            message_zh="今天核心資料不完整，換匯時間建議會改為保守。",
            blocks_model_advice=True,
            issues=blocking_issues[:5],
        )

    cny_proxy_ok = any(item.get("feature") == "CNY_CLOSE" and item.get("status") == "OK" for item in quality)
    cnh_latest_ok = any(item.get("feature") == "CNH_CLOSE" and not item.get("latest_missing") for item in quality)
    limited_issues = []
    for item in quality:
        feature = item.get("feature")
        if feature == "CNH_CLOSE" and cny_proxy_ok:
            continue
        if feature == "CNY_CLOSE" and cnh_latest_ok:
            continue
        if item.get("latest_missing"):
            limited_issues.append(f"{feature} 最新值缺失")
            continue
        if item.get("status") in {"POOR", "LIMITED", "MISSING_COLUMN"} and not item.get("latest_missing"):
            limited_issues.append(f"{feature} 長期資料有限")
    if limited_issues:
        return DataQualitySummary(
            status="LIMITED",
            label_zh="部分資料有限",
            message_zh="今天核心資料可用，但部分輔助資料缺失或長期歷史仍有限。",
            blocks_model_advice=False,
            issues=limited_issues[:5],
        )

    return DataQualitySummary(
        status="OK",
        label_zh="正常",
        message_zh="今天核心資料正常。",
        blocks_model_advice=False,
        issues=[],
    )


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
        "CNH_CLOSE",
        "CNY_CLOSE",
        "KRW_CLOSE",
        "JPY_CLOSE",
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


def _is_core_dataset(dataset: str) -> bool:
    return dataset.startswith(("fx_prices:USD/TWD", "bank_rates:USD", "features:daily_v1", "predictions"))


_CORE_LATEST_FEATURES = {
    "USDTWD_CLOSE",
    "DXY_CLOSE",
    "DATA_COMPLETENESS",
}
