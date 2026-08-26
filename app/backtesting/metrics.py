from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    brier_score: float
    calibration_error: float
    twd_depreciation_recall: float


@dataclass(frozen=True)
class RegressionMetrics:
    mae: float
    rmse: float


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else math.nan


def classification_metrics(y_true: pd.Series, prob_up: pd.Series, threshold: float = 0.5) -> ClassificationMetrics:
    data = pd.DataFrame({"y": y_true, "p": prob_up}).dropna()
    if data.empty:
        return ClassificationMetrics(*(math.nan for _ in range(8)))
    y = data["y"].astype(int).to_numpy()
    p = data["p"].clip(0, 1).to_numpy()
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall) if not math.isnan(precision + recall) else math.nan
    return ClassificationMetrics(
        accuracy=_safe_div(tp + tn, len(y)),
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=roc_auc_score(data["y"], data["p"]),
        brier_score=float(np.mean((p - y) ** 2)),
        calibration_error=calibration_error(data["y"], data["p"]),
        twd_depreciation_recall=recall,
    )


def roc_auc_score(y_true: pd.Series, prob_up: pd.Series) -> float:
    data = pd.DataFrame({"y": y_true, "p": prob_up}).dropna().sort_values("p")
    positives = int((data["y"] == 1).sum())
    negatives = int((data["y"] == 0).sum())
    if positives == 0 or negatives == 0:
        return math.nan
    ranks = data["p"].rank(method="average").to_numpy()
    y = data["y"].astype(int).to_numpy()
    positive_rank_sum = float(ranks[y == 1].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def calibration_error(y_true: pd.Series, prob_up: pd.Series, bins: int = 10) -> float:
    data = pd.DataFrame({"y": y_true, "p": prob_up}).dropna()
    if data.empty:
        return math.nan
    intervals = pd.cut(data["p"].clip(0, 1), bins=np.linspace(0, 1, bins + 1), include_lowest=True)
    error = 0.0
    total = len(data)
    for _, group in data.groupby(intervals, observed=True):
        if group.empty:
            continue
        error += len(group) / total * abs(group["p"].mean() - group["y"].mean())
    return float(error)


def regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> RegressionMetrics:
    data = pd.DataFrame({"y": y_true, "pred": y_pred}).dropna()
    if data.empty:
        return RegressionMetrics(mae=math.nan, rmse=math.nan)
    error = data["pred"] - data["y"]
    return RegressionMetrics(mae=float(error.abs().mean()), rmse=float(np.sqrt((error**2).mean())))
