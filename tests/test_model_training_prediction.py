import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.database.schema import Base, Prediction
from app.models.training import _persist_latest_prediction


def test_failed_model_publication_preserves_previous_complete_model(tmp_path, monkeypatch):
    import pytest
    import joblib
    from app.models.training import _save_artifact
    destination = tmp_path / "model.joblib"
    _save_artifact({"version": "old"}, destination)
    def fail_dump(artifact, path):
        path.write_bytes(b"partial")
        assert joblib.load(destination) == {"version": "old"}
        raise OSError("disk full")
    monkeypatch.setattr("app.models.training.joblib.dump", fail_dump)
    with pytest.raises(OSError, match="disk full"):
        _save_artifact({"version": "new"}, destination)
    assert joblib.load(destination) == {"version": "old"}
    assert list(tmp_path.iterdir()) == [destination]


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
        "model_version": "phase4_ensemble_v3",
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
        first_snapshot = saved.input_snapshot
        artifact["classifiers"]["logistic"] = _DummyClassifier(0.99)
        _persist_latest_prediction(session, frame, artifact)
        session.refresh(saved)
        assert saved.input_snapshot == first_snapshot
        assert len(session.execute(select(Prediction)).scalars().all()) == 1

    assert saved.observed_at_utc.date().isoformat() == "2026-01-02"
    assert abs(saved.prob_up - 0.65) < 1e-9
    snapshot = json.loads(saved.input_snapshot)
    assert snapshot["USDTWD_CLOSE"] == 31.2
    assert snapshot["CHINA_FX_PROXY_RETURN_5D"] == -0.02
