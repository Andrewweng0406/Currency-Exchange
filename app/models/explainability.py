from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.config import ROOT
from app.models.training import MODEL_VERSION


@dataclass(frozen=True)
class FeatureContribution:
    feature: str
    direction: str
    magnitude: float
    model: str
    note: str = "association, not causation"


def explain_latest_prediction(horizon: str, feature_row: pd.Series, top_n: int = 5) -> list[FeatureContribution]:
    artifact_path = ROOT / "models" / "artifacts" / f"{MODEL_VERSION}_{horizon}.joblib"
    if not artifact_path.exists():
        return []
    artifact = joblib.load(artifact_path)
    columns = artifact["feature_columns"]
    frame = pd.DataFrame([feature_row]).reindex(columns=columns)
    contributions: list[FeatureContribution] = []
    contributions.extend(_logistic_contributions(artifact, frame, columns))
    contributions.extend(_xgboost_contributions(artifact, columns))
    merged = _merge_contributions(contributions)
    merged = [item for item in merged if _is_explainable_feature(item.feature)]
    return merged[:top_n]


def _logistic_contributions(artifact: dict[str, Any], frame: pd.DataFrame, columns: list[str]) -> list[FeatureContribution]:
    model = artifact["classifiers"].get("logistic")
    if model is None:
        return []
    try:
        transformed = model[:-1].transform(frame)
        coefs = model[-1].coef_[0]
        values = transformed[0] * coefs
    except Exception:
        return []
    return [
        FeatureContribution(
            feature=columns[idx],
            direction="USD_TWD_UP" if value > 0 else "USD_TWD_DOWN",
            magnitude=float(abs(value)),
            model="logistic",
        )
        for idx, value in enumerate(values)
        if np.isfinite(value) and abs(value) > 0
    ]


def _xgboost_contributions(artifact: dict[str, Any], columns: list[str]) -> list[FeatureContribution]:
    model = artifact["classifiers"].get("xgboost")
    if model is None:
        return []
    try:
        booster = model[-1].get_booster()
        scores = booster.get_score(importance_type="gain")
    except Exception:
        return []
    out = []
    for raw_name, score in scores.items():
        idx = int(raw_name[1:]) if raw_name.startswith("f") and raw_name[1:].isdigit() else None
        feature = columns[idx] if idx is not None and idx < len(columns) else raw_name
        out.append(
            FeatureContribution(
                feature=feature,
                direction="IMPORTANT_NON_DIRECTIONAL",
                magnitude=float(score),
                model="xgboost_gain",
            )
        )
    return out


def _merge_contributions(contributions: list[FeatureContribution]) -> list[FeatureContribution]:
    if not contributions:
        return []
    max_by_model: dict[str, float] = {}
    for item in contributions:
        max_by_model[item.model] = max(max_by_model.get(item.model, 0.0), item.magnitude)
    normalized = []
    for item in contributions:
        denom = max_by_model.get(item.model) or 1.0
        normalized.append(
            FeatureContribution(
                feature=item.feature,
                direction=item.direction,
                magnitude=round(item.magnitude / denom, 4),
                model=item.model,
                note=item.note,
            )
        )
    return sorted(normalized, key=lambda item: item.magnitude, reverse=True)


def _is_explainable_feature(feature: str) -> bool:
    upper = feature.upper()
    if "DATA_MISSING" in upper:
        return False
    if upper == "DATA_COMPLETENESS":
        return False
    return True
