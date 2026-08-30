from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app.config import risk_policy
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
    average_regret: float
    beat_fixed_rate: float


@dataclass(frozen=True)
class StrategySummary:
    model_strategy: str
    fixed_strategy: str
    passed: bool
    savings_vs_fixed_day_twd: float
    average_rate_difference: float
    worst_rate_difference: float
    volatility_difference: float
    beat_fixed_rate: float
    conclusion_zh: str


def compare_exchange_strategies(session, target_usd: float = 10_000, start_year: int = 2023) -> list[StrategyComparison]:
    predictions = walk_forward_probabilities(session, "5d")
    if predictions.empty:
        return []
    predictions["date"] = pd.to_datetime(predictions["date"], utc=True)
    predictions["opportunity_score"] = _expanding_opportunity_scores(predictions)
    predictions = predictions[pd.to_datetime(predictions["date"], utc=True).dt.year >= start_year].copy()
    if predictions.empty:
        return []
    payments = predictions.groupby([predictions["date"].dt.year, predictions["date"].dt.month]).tail(1)["date"].tolist()
    records = {"fixed_day_once": [], "equal_tranches": [], "model_timing_once": []}
    for payment in payments:
        window = predictions[(predictions["date"] <= payment) & (predictions["date"] >= payment - pd.Timedelta(days=30))]
        if len(window) < 5:
            continue
        fixed_cost = _fixed(window, target_usd)
        tranche_cost = _tranches(window, target_usd)
        timing_cost = _model_timing_once(window, target_usd)
        best_possible = float(window["USDTWD_CLOSE"].min() * target_usd)
        records["fixed_day_once"].append((fixed_cost, best_possible, fixed_cost))
        records["equal_tranches"].append((tranche_cost, best_possible, fixed_cost))
        records["model_timing_once"].append((timing_cost, best_possible, fixed_cost))
    fixed_avg = _average_cost(records["fixed_day_once"])
    comparisons = []
    for name, values in records.items():
        if not values:
            continue
        costs = pd.Series([v[0] for v in values])
        regrets = pd.Series([v[0] - v[1] for v in values])
        fixed_costs = pd.Series([v[2] for v in values])
        comparisons.append(
            StrategyComparison(
                strategy=name,
                payments=len(values),
                average_rate=float(costs.mean() / target_usd),
                worst_rate=float(costs.max() / target_usd),
                cost_volatility=float(costs.std(ddof=0) / target_usd),
                maximum_regret=float(regrets.max()),
                savings_vs_fixed_day_twd=float(fixed_avg - costs.mean()),
                average_regret=float(regrets.mean()),
                beat_fixed_rate=float((costs < fixed_costs).mean()),
            )
        )
    return comparisons


def summarize_strategy_comparison(comparisons: list[StrategyComparison]) -> StrategySummary | None:
    by_name = {item.strategy: item for item in comparisons}
    fixed = by_name.get("fixed_day_once")
    model = by_name.get("model_timing_once")
    if fixed is None or model is None:
        return None
    average_rate_difference = model.average_rate - fixed.average_rate
    worst_rate_difference = model.worst_rate - fixed.worst_rate
    volatility_difference = model.cost_volatility - fixed.cost_volatility
    passed = model.savings_vs_fixed_day_twd > 0 and worst_rate_difference <= 0 and volatility_difference <= 0
    if passed:
        conclusion = "模型選時策略在此回測期間優於固定日期，且沒有提高最差匯率或成本波動。"
    elif model.savings_vs_fixed_day_twd > 0:
        conclusion = "模型選時策略平均成本較低，但最差情境或成本波動沒有同步改善，暫時不應視為全面勝出。"
    else:
        conclusion = "模型選時策略在此回測期間沒有打敗固定日期，目前只能作為風險提醒，不能宣稱能省錢。"
    return StrategySummary(
        model_strategy=model.strategy,
        fixed_strategy=fixed.strategy,
        passed=passed,
        savings_vs_fixed_day_twd=model.savings_vs_fixed_day_twd,
        average_rate_difference=average_rate_difference,
        worst_rate_difference=worst_rate_difference,
        volatility_difference=volatility_difference,
        beat_fixed_rate=model.beat_fixed_rate,
        conclusion_zh=conclusion,
    )


def _fixed(window: pd.DataFrame, target_usd: float) -> float:
    return float(window.iloc[0]["USDTWD_CLOSE"] * target_usd)


def _tranches(window: pd.DataFrame, target_usd: float, tranches: int = 4) -> float:
    positions = [round(i * (len(window) - 1) / (tranches - 1)) for i in range(tranches)]
    selected = window.iloc[positions]
    return float((selected["USDTWD_CLOSE"] * (target_usd / tranches)).sum())


def _model_timing_once(window: pd.DataFrame, target_usd: float) -> float:
    policy = risk_policy()["strategy_backtest"]["model_timing_once"]
    deadline_buffer_days = int(policy["deadline_buffer_days"])
    risk_probability = float(policy["risk_probability_up_min"])
    opportunity_min = float(policy["opportunity_score_min"])
    favorable_probability_max = float(policy["favorable_probability_up_max"])
    fallback = window.tail(max(1, deadline_buffer_days + 1)).head(1).iloc[0]
    for idx, row in window.reset_index(drop=True).iterrows():
        days_left = len(window) - idx - 1
        if days_left <= deadline_buffer_days:
            return float(row["USDTWD_CLOSE"] * target_usd)
        prob_up = float(row.get("prob_up", 0.5))
        opportunity = float(row.get("opportunity_score", 0))
        if prob_up >= risk_probability:
            return float(row["USDTWD_CLOSE"] * target_usd)
        if opportunity >= opportunity_min and prob_up <= favorable_probability_max:
            return float(row["USDTWD_CLOSE"] * target_usd)
    return float(fallback["USDTWD_CLOSE"] * target_usd)


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


def _average_cost(values: list[tuple[float, ...]]) -> float:
    return float(pd.Series([v[0] for v in values]).mean()) if values else 0.0


def _expanding_opportunity_scores(predictions: pd.DataFrame) -> list[int]:
    history = []
    scores = []
    for _, row in predictions.iterrows():
        history.append({"USDTWD_CLOSE": row["USDTWD_CLOSE"]})
        scores.append(opportunity_score(pd.DataFrame(history)))
    return scores
