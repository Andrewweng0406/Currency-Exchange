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
    if dataset.frame.empty:
        return pd.DataFrame()
    frame = dataset.frame.replace([np.inf, -np.inf], np.nan).dropna(subset=[f"TARGET_UP_{days}D", "USDTWD_CLOSE"]).reset_index(drop=True)
    cols = modeling_columns(frame)
    splits = yearly_expanding_splits(
        frame, min_train_years=min_train_years,
        label_end_col=f"TARGET_END_{HORIZON_DAYS[horizon]}D",
    )
    results = []
    prior_validation = {}
    for split in splits:
        train = frame.loc[split.train_index]
        test = frame.loc[split.test_index]
        models = _classification_models()
        models["time_series"] = TimeSeriesDirectionModel(days)
        model_probs = {}
        for name, model in models.items():
            model.fit(train[cols], train[f"TARGET_UP_{days}D"].astype(int))
            model_probs[name] = model.predict_proba(test[cols])[:, 1]
        prior_scores = {}
        for name, parts in prior_validation.items():
            known = pd.concat(parts, ignore_index=True)
            known = known[known["label_end"] < test["date"].min()]
            if not known.empty:
                prior_scores[name] = classification_metrics(known["y"], known["prob"])
        if not prior_scores:
            weights = {name: float(name == "coin_flip") for name in models}
        else:
            weights = _weights_from_scores(prior_scores)
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
        for name, values in model_probs.items():
            prior_validation.setdefault(name, []).append(pd.DataFrame({
                "label_end": test[f"TARGET_END_{days}D"].to_numpy(),
                "y": test[f"TARGET_UP_{days}D"].to_numpy(), "prob": values,
            }))
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()
