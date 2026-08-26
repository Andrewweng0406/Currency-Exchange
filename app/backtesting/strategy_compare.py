from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app.models.walkforward import walk_forward_probabilities
from app.risk.scoring import opportunity_score


@dataclass(frozen=True)
class StrategyComparison:
    strategy: str
    payments: int
    average_rate: float
    worst_rate: float
    cost_volatility: float
    maximum_regret: float
    savings_vs_fixed_day_twd: float


def compare_exchange_strategies(session, target_usd: float = 10_000, start_year: int = 2023) -> list[StrategyComparison]:
    predictions = walk_forward_probabilities(session, "5d")
    if predictions.empty:
        return []
    predictions = predictions[pd.to_datetime(predictions["date"], utc=True).dt.year >= start_year].copy()
    if predictions.empty:
        return []
    predictions["date"] = pd.to_datetime(predictions["date"], utc=True)
    payments = predictions.groupby([predictions["date"].dt.year, predictions["date"].dt.month]).tail(1)["date"].tolist()
    records = {"fixed_day_once": [], "equal_tranches": [], "risk_based_strategy": []}
    for payment in payments:
        window = predictions[(predictions["date"] <= payment) & (predictions["date"] >= payment - pd.Timedelta(days=30))]
        if len(window) < 5:
            continue
        fixed_cost = _fixed(window, target_usd)
        tranche_cost = _tranches(window, target_usd)
        risk_cost = _risk_based(window, target_usd)
        best_possible = float(window["USDTWD_CLOSE"].min() * target_usd)
        records["fixed_day_once"].append((fixed_cost, best_possible))
        records["equal_tranches"].append((tranche_cost, best_possible))
        records["risk_based_strategy"].append((risk_cost, best_possible))
    fixed_avg = _average_cost(records["fixed_day_once"])
    comparisons = []
    for name, values in records.items():
        if not values:
            continue
        costs = pd.Series([v[0] for v in values])
        regrets = pd.Series([v[0] - v[1] for v in values])
        comparisons.append(
            StrategyComparison(
                strategy=name,
                payments=len(values),
                average_rate=float(costs.mean() / target_usd),
                worst_rate=float(costs.max() / target_usd),
                cost_volatility=float(costs.std(ddof=0) / target_usd),
                maximum_regret=float(regrets.max()),
                savings_vs_fixed_day_twd=float(fixed_avg - costs.mean()),
            )
        )
    return comparisons


def _fixed(window: pd.DataFrame, target_usd: float) -> float:
    return float(window.iloc[0]["USDTWD_CLOSE"] * target_usd)


def _tranches(window: pd.DataFrame, target_usd: float, tranches: int = 4) -> float:
    positions = [round(i * (len(window) - 1) / (tranches - 1)) for i in range(tranches)]
    selected = window.iloc[positions]
    return float((selected["USDTWD_CLOSE"] * (target_usd / tranches)).sum())


def _risk_based(window: pd.DataFrame, target_usd: float) -> float:
    remaining = target_usd
    cost = 0.0
    close_history = []
    for idx, row in window.reset_index(drop=True).iterrows():
        days_left = len(window) - idx - 1
        close_history.append({"USDTWD_CLOSE": row["USDTWD_CLOSE"]})
        opp = opportunity_score(pd.DataFrame(close_history))
        fraction = 0.0
        if days_left <= 1:
            fraction = 1.0
        elif row["prob_up"] >= 0.70:
            fraction = 0.50
        elif row["prob_up"] >= 0.60 or opp >= 80:
            fraction = 0.25
        elif idx in {0, len(window) // 3, 2 * len(window) // 3}:
            fraction = 0.15
        buy = min(remaining, remaining * fraction)
        cost += float(buy * row["USDTWD_CLOSE"])
        remaining -= buy
        if remaining <= 0:
            break
    if remaining > 0:
        cost += float(remaining * window.iloc[-1]["USDTWD_CLOSE"])
    return cost


def _average_cost(values: list[tuple[float, float]]) -> float:
    return float(pd.Series([v[0] for v in values]).mean()) if values else 0.0
