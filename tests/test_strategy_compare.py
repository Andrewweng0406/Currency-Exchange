from app.backtesting.strategy_compare import _fixed, _risk_based, _tranches
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
