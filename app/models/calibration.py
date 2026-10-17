"""Research-only sigmoid calibration with a purged chronological holdout."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _logit(probability):
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


@dataclass
class CalibratedModel:
    estimator: object
    calibrator: object
    fit_label_end: pd.Timestamp
    calibration_start: pd.Timestamp
    calibration_end: pd.Timestamp

    def predict_proba(self, x):
        return self.calibrator.predict_proba(_logit(self.estimator.predict_proba(x)[:, 1]))


def fit_chronological_calibration(estimator, x, y, dates, label_end, calibration_rows=252):
    """Fit preprocessing/base model before calibration; never refit on holdout."""
    dates = pd.to_datetime(dates, utc=True)
    label_end = pd.to_datetime(label_end, utc=True)
    if len(x) < calibration_rows + 252:
        raise ValueError("Insufficient chronological calibration history")
    boundary = dates.iloc[-calibration_rows]
    train = (dates < boundary) & (label_end < boundary)
    calibration = dates >= boundary
    if train.sum() < 252 or y[train].nunique() != 2 or y[calibration].nunique() != 2:
        raise ValueError("Insufficient training or calibration classes")
    estimator.fit(x.loc[train], y.loc[train])
    predicted = estimator.predict_proba(x.loc[calibration])[:, 1]
    calibrator = LogisticRegression(C=1.0, max_iter=1000)
    calibrator.fit(_logit(predicted), y.loc[calibration])
    return CalibratedModel(estimator, calibrator, label_end[train].max(), boundary, dates[calibration].max())
