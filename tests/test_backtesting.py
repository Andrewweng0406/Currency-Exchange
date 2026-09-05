import math

import pandas as pd

from app.backtesting.dataset import add_forward_targets, modeling_columns
from app.backtesting.exchange_strategy import equal_tranche_exchange, fixed_day_exchange
from app.backtesting.metrics import classification_metrics, regression_metrics
from app.backtesting.splits import yearly_expanding_splits


def test_add_forward_targets():
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=30, tz="UTC"),
            "USDTWD_CLOSE": range(30, 60),
            "SOME_FEATURE": range(30),
        }
    )
    dataset = add_forward_targets(frame, horizons=(1, 5, 20))
    assert dataset.frame["TARGET_UP_1D"].iloc[0] == 1
    assert math.isclose(dataset.frame["TARGET_RETURN_5D"].iloc[0], 5 / 30)
    assert "SOME_FEATURE" in modeling_columns(dataset.frame)
    assert "TARGET_RETURN_1D" not in modeling_columns(dataset.frame)


def test_modeling_columns_uses_china_fx_proxy_not_raw_cnh_cny():
    frame = pd.DataFrame(
        {
            "CNH_RETURN_5D": [0.01],
            "CNY_RETURN_5D": [0.02],
            "CHINA_FX_PROXY_RETURN_5D": [0.015],
            "KRW_RETURN_5D": [0.01],
        }
    )
    cols = modeling_columns(frame)
    assert "CHINA_FX_PROXY_RETURN_5D" in cols
    assert "KRW_RETURN_5D" in cols
    assert "CNH_RETURN_5D" not in cols
    assert "CNY_RETURN_5D" not in cols


def test_yearly_expanding_splits_are_chronological():
    frame = pd.DataFrame({"date": pd.date_range("2016-01-01", "2024-12-31", freq="30D", tz="UTC")})
    splits = yearly_expanding_splits(frame, min_train_years=5)
    assert splits
    first = splits[0]
    assert first.train_end < first.test_start
    assert len(first.train_index) > 0
    assert len(first.test_index) > 0


def test_metrics():
    cls = classification_metrics(pd.Series([1, 0, 1, 0]), pd.Series([0.8, 0.4, 0.6, 0.7]))
    assert cls.accuracy == 0.75
    assert 0 <= cls.roc_auc <= 1
    assert cls.brier_score >= 0
    reg = regression_metrics(pd.Series([0.1, 0.2]), pd.Series([0.0, 0.25]))
    assert math.isclose(reg.mae, 0.075)


def test_exchange_strategy_results():
    rates = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=40, tz="UTC"), "USDTWD_CLOSE": range(30, 70)})
    payment = pd.Timestamp("2026-02-09", tz="UTC")
    fixed = fixed_day_exchange(rates, 10_000, payment, days_before=30)
    tranches = equal_tranche_exchange(rates, 10_000, payment, days_before=30, tranches=4)
    assert fixed.average_rate == 39
    assert len(tranches.orders) == 4
    assert tranches.total_twd_cost > 0
