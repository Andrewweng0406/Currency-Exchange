from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app.backtesting.dataset import add_forward_targets


@dataclass(frozen=True)
class TailRiskSnapshot:
    horizon: str
    probabilities: dict[str, float]
    method: str


def estimate_tail_risk(features: pd.DataFrame, horizon_days: int = 5, lookback: int = 756) -> TailRiskSnapshot:
    """Estimate shock probabilities from realized historical forward returns.

    This intentionally starts as an empirical baseline. Later model phases can replace
    it with calibrated tail classifiers after there is enough clean history.
    """
    if features.empty:
        return TailRiskSnapshot(horizon=f"{horizon_days}d", probabilities={}, method="empirical_forward_return_baseline")
    frame = add_forward_targets(features, horizons=(horizon_days,)).frame
    returns = pd.to_numeric(frame[f"TARGET_RETURN_{horizon_days}D"], errors="coerce").dropna().tail(lookback)
    thresholds = {
        "USD_TWD_UP_GT_0_5PCT": 0.005,
        "USD_TWD_UP_GT_1PCT": 0.01,
        "USD_TWD_UP_GT_1_5PCT": 0.015,
        "USD_TWD_UP_GT_2PCT": 0.02,
    }
    if returns.empty:
        probabilities = {key: 0.0 for key in thresholds}
    else:
        probabilities = {key: float((returns > threshold).mean()) for key, threshold in thresholds.items()}
    return TailRiskSnapshot(
        horizon=f"{horizon_days}d",
        probabilities=probabilities,
        method=f"empirical_forward_return_baseline_last_{len(returns)}_observations",
    )
