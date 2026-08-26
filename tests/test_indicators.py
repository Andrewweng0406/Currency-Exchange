import pandas as pd

from app.features.indicators import bollinger_position, pct_return, rsi


def test_pct_return():
    series = pd.Series([100, 110, 121])
    result = pct_return(series, 1).round(2)
    assert pd.isna(result.iloc[0])
    assert result.iloc[1:].tolist() == [0.10, 0.10]


def test_rsi_bounds():
    values = pd.Series(range(1, 40), dtype=float)
    out = rsi(values, 14).dropna()
    assert ((out >= 0) & (out <= 100)).all()


def test_bollinger_position_midpoint():
    values = pd.Series([10.0] * 25)
    out = bollinger_position(values, 20)
    assert out.dropna().empty
