from __future__ import annotations

from dataclasses import dataclass
from itertools import product

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


@dataclass(frozen=True)
class StrategyReport:
    summary: StrategySummary | None
    strategies: list[StrategyComparison]
    target_usd: float
    verdict: str
    label_zh: str
    should_use_for_timing: bool
    key_findings_zh: list[str]
    caution_zh: str


@dataclass(frozen=True)
class TimingPolicy:
    risk_probability_up_min: float
    opportunity_score_min: float
    favorable_probability_up_max: float
    deadline_buffer_days: int


@dataclass(frozen=True)
class TunedPolicyYear:
    test_year: int
    policy: TimingPolicy
    train_payments: int
    test_payments: int
    train_score: float
    test_savings_vs_fixed_day_twd: float
    test_average_rate_difference: float
    test_worst_rate_difference: float
    test_volatility_difference: float
    test_beat_fixed_rate: float
    passed: bool


@dataclass(frozen=True)
class WalkForwardTuningResult:
    start_year: int
    target_usd: float
    years: list[TunedPolicyYear]
    summary: StrategySummary | None


def compare_exchange_strategies(session, target_usd: float = 10_000, start_year: int = 2023) -> list[StrategyComparison]:
    predictions = walk_forward_probabilities(session, "5d")
    if predictions.empty:
        return []
    predictions["date"] = pd.to_datetime(predictions["date"], utc=True)
    predictions["opportunity_score"] = _expanding_opportunity_scores(predictions)
    predictions = predictions[pd.to_datetime(predictions["date"], utc=True).dt.year >= start_year].copy()
    if predictions.empty:
        return []
    return _compare_from_predictions(predictions, target_usd)


def walk_forward_tune_strategy(session, target_usd: float = 10_000, start_year: int = 2023, min_train_years: int = 2) -> WalkForwardTuningResult:
    predictions = walk_forward_probabilities(session, "5d")
    if predictions.empty:
        return WalkForwardTuningResult(start_year=start_year, target_usd=target_usd, years=[], summary=None)
    predictions["date"] = pd.to_datetime(predictions["date"], utc=True)
    predictions["opportunity_score"] = _expanding_opportunity_scores(predictions)
    years = sorted(int(year) for year in predictions["date"].dt.year.unique() if int(year) >= start_year)
    tuned_years = []
    combined_records = {"fixed_day_once": [], "equal_tranches": [], "model_timing_once": []}
    for year in years:
        train = predictions[predictions["date"].dt.year < year]
        if train["date"].dt.year.nunique() < min_train_years:
            continue
        test = predictions[predictions["date"].dt.year == year]
        policy, train_score, train_payments = _select_policy(train, target_usd)
        test_records = _strategy_records(test, target_usd, policy)
        comparisons = _comparisons_from_records(test_records, target_usd)
        summary = summarize_strategy_comparison(comparisons)
        if summary is None:
            continue
        for name, values in test_records.items():
            combined_records[name].extend(values)
        tuned_years.append(
            TunedPolicyYear(
                test_year=year,
                policy=policy,
                train_payments=train_payments,
                test_payments=next((item.payments for item in comparisons if item.strategy == "model_timing_once"), 0),
                train_score=train_score,
                test_savings_vs_fixed_day_twd=summary.savings_vs_fixed_day_twd,
                test_average_rate_difference=summary.average_rate_difference,
                test_worst_rate_difference=summary.worst_rate_difference,
                test_volatility_difference=summary.volatility_difference,
                test_beat_fixed_rate=summary.beat_fixed_rate,
                passed=summary.passed,
            )
        )
    combined_summary = summarize_strategy_comparison(_comparisons_from_records(combined_records, target_usd))
    return WalkForwardTuningResult(start_year=start_year, target_usd=target_usd, years=tuned_years, summary=combined_summary)


def _compare_from_predictions(predictions: pd.DataFrame, target_usd: float, policy: TimingPolicy | None = None) -> list[StrategyComparison]:
    return _comparisons_from_records(_strategy_records(predictions, target_usd, policy), target_usd)


def _strategy_records(predictions: pd.DataFrame, target_usd: float, policy: TimingPolicy | None = None) -> dict[str, list[tuple[float, float, float]]]:
    if predictions.empty:
        return {"fixed_day_once": [], "equal_tranches": [], "model_timing_once": []}
    predictions = predictions.copy()
    records = {"fixed_day_once": [], "equal_tranches": [], "model_timing_once": []}
    payments = predictions.groupby([predictions["date"].dt.year, predictions["date"].dt.month]).tail(1)["date"].tolist()
    for payment in payments:
        window = predictions[(predictions["date"] <= payment) & (predictions["date"] >= payment - pd.Timedelta(days=30))]
        if len(window) < 5:
            continue
        fixed_cost = _fixed(window, target_usd)
        tranche_cost = _tranches(window, target_usd)
        timing_cost = _model_timing_once(window, target_usd, policy)
        best_possible = float(window["USDTWD_CLOSE"].min() * target_usd)
        records["fixed_day_once"].append((fixed_cost, best_possible, fixed_cost))
        records["equal_tranches"].append((tranche_cost, best_possible, fixed_cost))
        records["model_timing_once"].append((timing_cost, best_possible, fixed_cost))
    return records


def _comparisons_from_records(records: dict[str, list[tuple[float, float, float]]], target_usd: float) -> list[StrategyComparison]:
    fixed_avg = _average_cost(records.get("fixed_day_once", []))
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


def build_strategy_report(comparisons: list[StrategyComparison], target_usd: float = 10_000) -> StrategyReport:
    summary = summarize_strategy_comparison(comparisons)
    by_name = {item.strategy: item for item in comparisons}
    fixed = by_name.get("fixed_day_once")
    model = by_name.get("model_timing_once")
    tranches = by_name.get("equal_tranches")

    if summary is None or fixed is None or model is None:
        return StrategyReport(
            summary=summary,
            strategies=comparisons,
            target_usd=target_usd,
            verdict="INSUFFICIENT_DATA",
            label_zh="回測資料不足",
            should_use_for_timing=False,
            key_findings_zh=["目前資料不足，不能判斷模型選時是否優於固定日期。"],
            caution_zh="請先累積更多歷史特徵與 walk-forward 預測，再評估策略表現。",
        )

    findings = [
        f"平均換匯成本差異：以每次 USD {target_usd:,.0f} 需求估算，模型選時比固定日期 {'少' if summary.savings_vs_fixed_day_twd > 0 else '多'} NT${abs(summary.savings_vs_fixed_day_twd):,.0f}。",
        f"最差匯率差異：模型選時 {'較好' if summary.worst_rate_difference <= 0 else '較差'} {abs(summary.worst_rate_difference):.4f}。",
        f"成本波動差異：模型選時 {'較低' if summary.volatility_difference <= 0 else '較高'} {abs(summary.volatility_difference):.4f}。",
        f"模型選時打敗固定日期比例：{summary.beat_fixed_rate:.0%}。",
    ]
    if tranches is not None:
        findings.append(f"平均分批最大後悔成本：NT${tranches.maximum_regret:,.0f}。")

    if summary.passed:
        verdict = "PASS"
        label = "回測通過"
        should_use = True
        caution = "可作為換匯時間參考，但仍只能用機率與風險管理角度使用。"
    elif summary.savings_vs_fixed_day_twd > 0:
        verdict = "MIXED"
        label = "結果分歧"
        should_use = False
        caution = "平均成本有改善，但風險指標未同步改善，暫時只能當提醒，不能宣稱策略已勝出。"
    else:
        verdict = "FAIL"
        label = "尚未通過"
        should_use = False
        caution = "模型選時沒有打敗固定日期，目前應視為風險提醒，不應視為省錢工具。"

    return StrategyReport(
        summary=summary,
        strategies=comparisons,
        target_usd=target_usd,
        verdict=verdict,
        label_zh=label,
        should_use_for_timing=should_use,
        key_findings_zh=findings,
        caution_zh=caution,
    )


def _fixed(window: pd.DataFrame, target_usd: float) -> float:
    return float(window.iloc[0]["USDTWD_CLOSE"] * target_usd)


def _tranches(window: pd.DataFrame, target_usd: float, tranches: int = 4) -> float:
    positions = [round(i * (len(window) - 1) / (tranches - 1)) for i in range(tranches)]
    selected = window.iloc[positions]
    return float((selected["USDTWD_CLOSE"] * (target_usd / tranches)).sum())


def _model_timing_once(window: pd.DataFrame, target_usd: float, policy: TimingPolicy | None = None) -> float:
    policy = policy or _default_timing_policy()
    deadline_buffer_days = int(policy.deadline_buffer_days)
    risk_probability = float(policy.risk_probability_up_min)
    opportunity_min = float(policy.opportunity_score_min)
    favorable_probability_max = float(policy.favorable_probability_up_max)
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


def _default_timing_policy() -> TimingPolicy:
    policy = risk_policy()["strategy_backtest"]["model_timing_once"]
    return TimingPolicy(
        risk_probability_up_min=float(policy["risk_probability_up_min"]),
        opportunity_score_min=float(policy["opportunity_score_min"]),
        favorable_probability_up_max=float(policy["favorable_probability_up_max"]),
        deadline_buffer_days=int(policy["deadline_buffer_days"]),
    )


def _candidate_policies() -> list[TimingPolicy]:
    cfg = risk_policy()["strategy_backtest"].get("tuning_grid", {})
    risks = cfg.get("risk_probability_up_min", [0.60, 0.65, 0.70])
    opportunities = cfg.get("opportunity_score_min", [65, 75, 85])
    favorable = cfg.get("favorable_probability_up_max", [0.50, 0.55])
    deadlines = cfg.get("deadline_buffer_days", [2, 3, 5])
    return [
        TimingPolicy(float(risk), float(opp), float(fav), int(deadline))
        for risk, opp, fav, deadline in product(risks, opportunities, favorable, deadlines)
    ]


def _select_policy(train: pd.DataFrame, target_usd: float) -> tuple[TimingPolicy, float, int]:
    best_policy = _default_timing_policy()
    best_score = float("-inf")
    best_payments = 0
    for policy in _candidate_policies():
        comparisons = _compare_from_predictions(train, target_usd, policy)
        summary = summarize_strategy_comparison(comparisons)
        if summary is None:
            continue
        payments = next((item.payments for item in comparisons if item.strategy == "model_timing_once"), 0)
        score = _policy_score(summary)
        if score > best_score:
            best_policy = policy
            best_score = score
            best_payments = payments
    return best_policy, best_score, best_payments


def _policy_score(summary: StrategySummary) -> float:
    return (
        summary.savings_vs_fixed_day_twd
        - max(0.0, summary.worst_rate_difference) * 10_000
        - max(0.0, summary.volatility_difference) * 5_000
    )
