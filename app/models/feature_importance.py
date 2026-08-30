from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np

from app.config import ROOT
from app.models.training import MODEL_VERSION


@dataclass(frozen=True)
class ImportanceItem:
    feature: str
    category: str
    importance: float
    model_source: str


def horizon_feature_importance(horizon: str, top_n: int = 12) -> list[ImportanceItem]:
    artifact_path = ROOT / "models" / "artifacts" / f"{MODEL_VERSION}_{horizon}.joblib"
    if not artifact_path.exists():
        return []
    artifact = joblib.load(artifact_path)
    columns = artifact["feature_columns"]
    items = _xgboost_gain_importance(artifact, columns)
    if not items:
        items = _logistic_abs_coef_importance(artifact, columns)
    return items[:top_n]


def _xgboost_gain_importance(artifact: dict[str, Any], columns: list[str]) -> list[ImportanceItem]:
    model = artifact["classifiers"].get("xgboost")
    if model is None:
        return []
    try:
        scores = model[-1].get_booster().get_score(importance_type="gain")
    except Exception:
        return []
    raw = []
    for raw_name, score in scores.items():
        idx = int(raw_name[1:]) if raw_name.startswith("f") and raw_name[1:].isdigit() else None
        feature = columns[idx] if idx is not None and idx < len(columns) else raw_name
        raw.append((feature, float(score)))
    return _normalize(raw, "xgboost_gain")


def _logistic_abs_coef_importance(artifact: dict[str, Any], columns: list[str]) -> list[ImportanceItem]:
    model = artifact["classifiers"].get("logistic")
    if model is None:
        return []
    try:
        coef = np.abs(model[-1].coef_[0])
    except Exception:
        return []
    raw = [(feature, float(coef[idx])) for idx, feature in enumerate(columns) if idx < len(coef)]
    return _normalize(raw, "logistic_abs_coef")


def _normalize(raw: list[tuple[str, float]], model_source: str) -> list[ImportanceItem]:
    if not raw:
        return []
    max_value = max(value for _, value in raw) or 1.0
    items = [
        ImportanceItem(
            feature=feature,
            category=feature_category(feature),
            importance=round(value / max_value, 4),
            model_source=model_source,
        )
        for feature, value in raw
        if np.isfinite(value) and value > 0
    ]
    return sorted(items, key=lambda item: item.importance, reverse=True)


def feature_category(feature: str) -> str:
    upper = feature.upper()
    if upper.startswith("USDTWD"):
        return "USD/TWD 自身走勢"
    if upper.startswith(("DXY", "BROAD_USD_INDEX")):
        return "美元強弱"
    if upper.startswith(("US2Y", "US10Y", "US_2S10S")):
        return "美國利率"
    if upper.startswith(("CNH", "CNY", "KRW", "JPY")):
        return "亞洲貨幣"
    if upper.startswith(("VIX", "SP500", "NASDAQ")):
        return "全球風險情緒"
    if upper.startswith(("TAIEX", "TSMC", "TSM_ADR")):
        return "台股與台積電"
    if upper.startswith("FOREIGN_FLOW"):
        return "台灣外資資金流"
    if "DATA_MISSING" in upper or upper == "DATA_COMPLETENESS":
        return "資料品質"
    return "其他"
