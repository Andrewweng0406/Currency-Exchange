from app.backtesting.strategy_compare import (
    StrategyComparison,
    TimingPolicy,
    _fixed,
    _model_timing_once,
    _risk_based,
    _select_policy,
    _strategy_records,
    _tranches,
    build_strategy_report,
    summarize_strategy_comparison,
)
import pandas as pd


def test_strategy_costs():
    window = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=10, tz="UTC"),
            "USDTWD_CLOSE": [31, 31.1, 31.2, 31.3, 31.4, 31.5, 31.6, 31.7, 31.8, 31.9],
            "prob_up": [0.5] * 10,
        }
    )
    assert _fixed(window, 1000) == 31000
    assert _tranches(window, 1000) > 31000
    assert _risk_based(window, 1000) > 0


def test_model_timing_once_buys_when_depreciation_risk_is_high():
    window = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=10, tz="UTC"),
            "USDTWD_CLOSE": [31, 31.1, 31.2, 31.3, 31.4, 31.5, 31.6, 31.7, 31.8, 31.9],
            "prob_up": [0.4, 0.66, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
            "opportunity_score": [0] * 10,
        }
    )
    assert _model_timing_once(window, 1000) == 31100


def test_model_timing_once_waits_until_deadline_when_no_signal():
    window = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=10, tz="UTC"),
            "USDTWD_CLOSE": [31, 31.1, 31.2, 31.3, 31.4, 31.5, 31.6, 31.7, 31.8, 31.9],
            "prob_up": [0.5] * 10,
            "opportunity_score": [0] * 10,
        }
    )
    assert _model_timing_once(window, 1000) == 31600


def test_strategy_summary_marks_model_timing_failure_honestly():
    summary = summarize_strategy_comparison(
        [
            StrategyComparison("fixed_day_once", 12, 31.0, 32.0, 0.5, 1000, 0, 500, 0),
            StrategyComparison("model_timing_once", 12, 31.1, 32.1, 0.6, 1200, -1000, 600, 0.25),
        ]
    )
    assert summary is not None
    assert summary.passed is False
    assert "不能宣稱能省錢" in summary.conclusion_zh


def test_strategy_summary_requires_volatility_not_worse():
    summary = summarize_strategy_comparison(
        [
            StrategyComparison("fixed_day_once", 12, 31.0, 32.0, 0.5, 1000, 0, 500, 0),
            StrategyComparison("model_timing_once", 12, 30.9, 31.9, 0.6, 900, 1000, 400, 0.5),
        ]
    )
    assert summary is not None
    assert summary.passed is False
    assert "成本波動沒有同步改善" in summary.conclusion_zh


def test_strategy_report_is_conservative_when_results_are_mixed():
    report = build_strategy_report(
        [
            StrategyComparison("fixed_day_once", 12, 31.0, 32.0, 0.5, 1000, 0, 500, 0),
            StrategyComparison("equal_tranches", 12, 31.0, 31.9, 0.4, 800, 0, 300, 0.5),
            StrategyComparison("model_timing_once", 12, 30.9, 31.9, 0.6, 900, 1000, 400, 0.5),
        ],
        target_usd=5000,
    )
    assert report.verdict == "MIXED"
    assert report.target_usd == 5000
    assert not report.should_use_for_timing
    assert "不能宣稱策略已勝出" in report.caution_zh
    assert any("USD 5,000" in item for item in report.key_findings_zh)


def test_strategy_records_accepts_policy_override():
    window = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=10, tz="UTC"),
            "USDTWD_CLOSE": [31, 31.1, 31.2, 31.3, 31.4, 31.5, 31.6, 31.7, 31.8, 31.9],
            "prob_up": [0.70] * 10,
            "opportunity_score": [0] * 10,
        }
    )
    policy = TimingPolicy(
        risk_probability_up_min=0.80,
        opportunity_score_min=99,
        favorable_probability_up_max=0.20,
        deadline_buffer_days=1,
    )
    records = _strategy_records(window, 1000, policy)
    assert len(records["model_timing_once"]) == 1
    assert records["model_timing_once"][0][0] == 31800


def test_select_policy_uses_training_window_only():
    train = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=70, tz="UTC"),
            "USDTWD_CLOSE": [31 + (i % 20) * 0.01 for i in range(70)],
            "prob_up": [0.5] * 70,
            "opportunity_score": [80] * 70,
        }
    )
    policy, score, payments = _select_policy(train, 1000)
    assert 0.6 <= policy.risk_probability_up_min <= 0.7
    assert policy.opportunity_score_min in {65, 75, 85}
    assert payments >= 2
    assert score == score
