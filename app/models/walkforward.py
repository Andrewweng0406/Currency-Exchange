from __future__ import annotations

import numpy as np
import pandas as pd

from app.backtesting.dataset import add_forward_targets, load_feature_frame, modeling_columns
from app.backtesting.splits import yearly_expanding_splits
from app.models.training import HORIZON_DAYS, TimeSeriesDirectionModel, _classification_models, _weights_from_scores
from app.backtesting.metrics import classification_metrics


def walk_forward_probabilities(session, horizon: str = "5d", min_train_years: int = 5) -> pd.DataFrame:
    days = HORIZON_DAYS[horizon]
    dataset = add_forward_targets(load_feature_frame(session), horizons=(days,))
    frame = dataset.frame.replace([np.inf, -np.inf], np.nan).dropna(subset=[f"TARGET_UP_{days}D", "USDTWD_CLOSE"]).reset_index(drop=True)
    cols = modeling_columns(frame)
    splits = yearly_expanding_splits(frame, min_train_years=min_train_years)
    results = []
    prior_weights = None
    for split in splits:
        train = frame.loc[split.train_index]
        test = frame.loc[split.test_index]
        models = _classification_models()
        models["time_series"] = TimeSeriesDirectionModel(days)
        model_probs = {}
        for name, model in models.items():
            model.fit(train[cols], train[f"TARGET_UP_{days}D"].astype(int))
            model_probs[name] = model.predict_proba(test[cols])[:, 1]
        if prior_weights is None:
            weights = {name: 1 / len(models) for name in models}
        else:
            weights = prior_weights
        ensemble = sum(model_probs[name] * weights[name] for name in models)
        part = pd.DataFrame(
            {
                "date": test["date"].to_numpy(),
                "USDTWD_CLOSE": test["USDTWD_CLOSE"].to_numpy(),
                "prob_up": ensemble,
                "target_up": test[f"TARGET_UP_{days}D"].to_numpy(),
                "target_return": test[f"TARGET_RETURN_{days}D"].to_numpy(),
            }
        )
        for name, values in model_probs.items():
            part[f"prob_{name}"] = values
        results.append(part)
        prior_scores = {name: classification_metrics(test[f"TARGET_UP_{days}D"], pd.Series(values)) for name, values in model_probs.items()}
        prior_weights = _weights_from_scores(prior_scores)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()
