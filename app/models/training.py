from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from app.backtesting.dataset import add_forward_targets, load_feature_frame, modeling_columns
from app.backtesting.metrics import classification_metrics
from app.backtesting.splits import yearly_expanding_splits
from app.config import ROOT
from app.database.schema import ModelPerformance, Prediction
from app.database.upsert import upsert_rows

HORIZON_DAYS = {"1d": 1, "5d": 5, "20d": 20}
MODEL_VERSION = "phase4_ensemble_v1"


class TimeSeriesDirectionModel:
    def __init__(self, horizon_days: int) -> None:
        self.horizon_days = horizon_days
        self.base_rate = 0.5
        self.momentum_scale = 1.0
        self.return_col = "USDTWD_RETURN_1D"

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "TimeSeriesDirectionModel":
        self.base_rate = float(y.mean()) if len(y) else 0.5
        if self.horizon_days in {5, 20} and f"USDTWD_RETURN_{self.horizon_days}D" in x:
            self.return_col = f"USDTWD_RETURN_{self.horizon_days}D"
        returns = pd.to_numeric(x.get(self.return_col, pd.Series(dtype=float)), errors="coerce").dropna()
        std = float(returns.std()) if len(returns) else 0.0
        self.momentum_scale = std if std > 0 else 0.002
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        momentum = pd.to_numeric(x.get(self.return_col, pd.Series(0, index=x.index)), errors="coerce").fillna(0)
        adjustment = np.tanh(momentum.to_numpy() / (self.momentum_scale * 2)) * 0.15
        prob_up = np.clip(self.base_rate + adjustment, 0.05, 0.95)
        return np.column_stack([1 - prob_up, prob_up])


@dataclass(frozen=True)
class ModelScore:
    model_name: str
    horizon: str
    accuracy: float
    f1: float
    roc_auc: float
    brier_score: float
    twd_depreciation_recall: float
    mae: float
    rmse: float
    weight: float


@dataclass(frozen=True)
class HorizonTrainResult:
    horizon: str
    rows: int
    features: int
    splits: int
    scores: list[ModelScore]
    artifact_path: str


def _classification_models(random_state: int = 42) -> dict[str, Any]:
    return {
        "logistic": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        ),
        "xgboost": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=120,
                        max_depth=3,
                        learning_rate=0.04,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        eval_metric="logloss",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def _regression_models(random_state: int = 42) -> dict[str, Any]:
    return {
        "ridge": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=5.0)),
            ]
        ),
        "xgboost_reg": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                (
                    "model",
                    XGBRegressor(
                        n_estimators=120,
                        max_depth=3,
                        learning_rate=0.04,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def prepare_training_frame(session, horizon: str) -> tuple[pd.DataFrame, list[str], str, str]:
    days = HORIZON_DAYS[horizon]
    dataset = add_forward_targets(load_feature_frame(session), horizons=(days,))
    frame = dataset.frame.replace([np.inf, -np.inf], np.nan)
    target_up = f"TARGET_UP_{days}D"
    target_return = f"TARGET_RETURN_{days}D"
    cols = modeling_columns(frame)
    usable = frame.dropna(subset=[target_up, target_return, "USDTWD_CLOSE"]).reset_index(drop=True)
    return usable, cols, target_up, target_return


def train_horizon(session, horizon: str, min_train_years: int = 5) -> HorizonTrainResult:
    frame, cols, target_up, target_return = prepare_training_frame(session, horizon)
    splits = yearly_expanding_splits(frame, min_train_years=min_train_years)
    if not splits:
        raise ValueError(f"Not enough data to create walk-forward splits for {horizon}")

    classification_models = _classification_models()
    classification_models["time_series"] = TimeSeriesDirectionModel(HORIZON_DAYS[horizon])
    clf_predictions: dict[str, list[pd.DataFrame]] = {name: [] for name in classification_models}
    reg_predictions: dict[str, list[pd.DataFrame]] = {name: [] for name in _regression_models()}

    for split in splits:
        train = frame.loc[split.train_index]
        test = frame.loc[split.test_index]
        x_train = train[cols]
        x_test = test[cols]
        y_train = train[target_up].astype(int)
        r_train = train[target_return].astype(float)
        for name, model in classification_models.items():
            model.fit(x_train, y_train)
            prob = model.predict_proba(x_test)[:, 1]
            clf_predictions[name].append(pd.DataFrame({"date": test["date"], "y": test[target_up], "prob": prob}))
        for name, model in _regression_models().items():
            model.fit(x_train, r_train)
            pred = model.predict(x_test)
            reg_predictions[name].append(pd.DataFrame({"date": test["date"], "y": test[target_return], "pred": pred}))

    clf_scores = {}
    for name, parts in clf_predictions.items():
        pred = pd.concat(parts, ignore_index=True)
        metrics = classification_metrics(pred["y"], pred["prob"])
        clf_scores[name] = metrics

    reg_scores = {}
    for name, parts in reg_predictions.items():
        pred = pd.concat(parts, ignore_index=True).dropna()
        mae = float(mean_absolute_error(pred["y"], pred["pred"]))
        rmse = float(mean_squared_error(pred["y"], pred["pred"]) ** 0.5)
        reg_scores[name] = {"mae": mae, "rmse": rmse}

    weights = _weights_from_scores(clf_scores)
    scores = [
        ModelScore(
            model_name=name,
            horizon=horizon,
            accuracy=metrics.accuracy,
            f1=metrics.f1,
            roc_auc=metrics.roc_auc,
            brier_score=metrics.brier_score,
            twd_depreciation_recall=metrics.twd_depreciation_recall,
            mae=reg_scores.get("ridge", {}).get("mae", math.nan),
            rmse=reg_scores.get("ridge", {}).get("rmse", math.nan),
            weight=weights[name],
        )
        for name, metrics in clf_scores.items()
    ]

    final_classifiers = _classification_models()
    final_classifiers["time_series"] = TimeSeriesDirectionModel(HORIZON_DAYS[horizon])
    final_regressors = _regression_models()
    x = frame[cols]
    y = frame[target_up].astype(int)
    r = frame[target_return].astype(float)
    for model in final_classifiers.values():
        model.fit(x, y)
    for model in final_regressors.values():
        model.fit(x, r)

    artifact = {
        "model_version": MODEL_VERSION,
        "horizon": horizon,
        "horizon_days": HORIZON_DAYS[horizon],
        "feature_columns": cols,
        "classifiers": final_classifiers,
        "regressors": final_regressors,
        "classification_weights": weights,
        "prediction_interval_80": {
            "lower": float(frame[target_return].quantile(0.10)),
            "upper": float(frame[target_return].quantile(0.90)),
            "method": "historical_forward_return_quantiles",
        },
        "scores": [asdict(score) for score in scores],
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    artifact_path = ROOT / "models" / "artifacts" / f"{MODEL_VERSION}_{horizon}.joblib"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, artifact_path)
    _persist_performance(session, scores)
    _persist_latest_prediction(session, _latest_feature_frame(session), artifact)
    return HorizonTrainResult(
        horizon=horizon,
        rows=len(frame),
        features=len(cols),
        splits=len(splits),
        scores=scores,
        artifact_path=str(artifact_path.relative_to(ROOT)),
    )


def _weights_from_scores(scores: dict[str, Any]) -> dict[str, float]:
    raw = {}
    for name, metrics in scores.items():
        auc_component = 0.5 if math.isnan(metrics.roc_auc) else max(0.0, metrics.roc_auc)
        brier_component = max(0.0, 1.0 - metrics.brier_score)
        recall_component = 0.5 if math.isnan(metrics.twd_depreciation_recall) else metrics.twd_depreciation_recall
        raw[name] = max(0.001, 0.45 * auc_component + 0.35 * brier_component + 0.20 * recall_component)
    total = sum(raw.values())
    return {name: value / total for name, value in raw.items()}


def _persist_performance(session, scores: list[ModelScore]) -> None:
    now = datetime.now(timezone.utc)
    rows = []
    for score in scores:
        for metric, value in asdict(score).items():
            if metric in {"model_name", "horizon"}:
                continue
            rows.append(
                {
                    "observed_at_utc": now,
                    "source": "walk_forward_validation",
                    "model_version": f"{MODEL_VERSION}:{score.model_name}",
                    "horizon": score.horizon,
                    "metric": metric,
                    "value": value,
                    "window": "walk_forward",
                }
            )
    upsert_rows(session, ModelPerformance, rows, ("model_version", "horizon", "metric", "observed_at_utc", "source"))


def _latest_feature_frame(session) -> pd.DataFrame:
    return load_feature_frame(session).replace([np.inf, -np.inf], np.nan)


def _persist_latest_prediction(session, frame: pd.DataFrame, artifact: dict[str, Any]) -> None:
    latest = frame.tail(1)
    if latest.empty:
        return
    row = latest.iloc[0]
    x = latest.reindex(columns=artifact["feature_columns"])
    probs = {
        name: float(model.predict_proba(x)[:, 1][0])
        for name, model in artifact["classifiers"].items()
    }
    returns = {
        name: float(model.predict(x)[0])
        for name, model in artifact["regressors"].items()
    }
    prob_up = sum(probs[name] * artifact["classification_weights"][name] for name in probs)
    expected_return = float(np.mean(list(returns.values())))
    agreement = 1 - float(np.std(list(probs.values())))
    data_completeness = float(row.get("DATA_COMPLETENESS", 0.5) or 0.5)
    confidence = max(0.0, min(1.0, 0.65 * agreement + 0.35 * data_completeness))
    snapshot_keys = [
        "USDTWD_CLOSE",
        "DXY_RETURN_1D",
        "US2Y_CHANGE_1D",
        "US10Y_CHANGE_1D",
        "CHINA_FX_PROXY_RETURN_5D",
        "DATA_COMPLETENESS",
    ]
    snapshot = {key: _json_safe(row.get(key)) for key in snapshot_keys if key in row}
    snapshot["component_probabilities"] = probs
    snapshot["component_expected_returns"] = returns
    snapshot["prediction_interval_80"] = artifact.get("prediction_interval_80")
    upsert_rows(
        session,
        Prediction,
        [
            {
                "observed_at_utc": pd.Timestamp(row["date"]).to_pydatetime(),
                "source": "model_training_latest",
                "model_version": artifact["model_version"],
                "horizon": artifact["horizon"],
                "prob_up": prob_up,
                "prob_down": 1 - prob_up,
                "expected_return": expected_return,
                "confidence": confidence,
                "risk_score": None,
                "input_snapshot": json.dumps(snapshot, sort_keys=True),
                "recommendation": None,
            }
        ],
        ("model_version", "horizon", "observed_at_utc", "source"),
    )


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if isinstance(value, np.generic):
        return value.item()
    return value
