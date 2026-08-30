from app.backtesting.strategy_compare import StrategyComparison, _fixed, _model_timing_once, _risk_based, _tranches, summarize_strategy_comparison
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
