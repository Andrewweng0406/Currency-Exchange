import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.database.schema import Base, Prediction
from app.models.training import _persist_latest_prediction


class _DummyClassifier:
    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, x):
        return np.column_stack(
            [
                np.full(len(x), 1 - self.probability),
                np.full(len(x), self.probability),
            ]
        )


class _DummyRegressor:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, x):
        return np.full(len(x), self.value)


def test_persist_latest_prediction_uses_latest_feature_row_not_latest_training_target_row():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    frame = pd.DataFrame(
        [
            {
                "date": datetime(2026, 1, 1, tzinfo=timezone.utc),
                "USDTWD_CLOSE": 31.0,
                "CHINA_FX_PROXY_RETURN_5D": 0.01,
                "DATA_COMPLETENESS": 0.8,
                "FEATURE_A": 1.0,
            },
            {
                "date": datetime(2026, 1, 2, tzinfo=timezone.utc),
                "USDTWD_CLOSE": 31.2,
                "CHINA_FX_PROXY_RETURN_5D": -0.02,
                "DATA_COMPLETENESS": 0.9,
                "FEATURE_A": 2.0,
            },
        ]
    )
    artifact = {
        "model_version": "phase4_ensemble_v1",
        "horizon": "5d",
        "feature_columns": ["FEATURE_A", "FEATURE_B"],
        "classifiers": {"logistic": _DummyClassifier(0.7), "time_series": _DummyClassifier(0.5)},
        "regressors": {"ridge": _DummyRegressor(0.001)},
        "classification_weights": {"logistic": 0.75, "time_series": 0.25},
        "prediction_interval_80": {"lower": -0.01, "upper": 0.01},
    }

    with Session(engine) as session:
        _persist_latest_prediction(session, frame, artifact)
        saved = session.execute(select(Prediction)).scalar_one()

    assert saved.observed_at_utc.date().isoformat() == "2026-01-02"
    assert abs(saved.prob_up - 0.65) < 1e-9
    snapshot = json.loads(saved.input_snapshot)
    assert snapshot["USDTWD_CLOSE"] == 31.2
    assert snapshot["CHINA_FX_PROXY_RETURN_5D"] == -0.02
