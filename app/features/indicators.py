from __future__ import annotations

import pandas as pd


def pct_return(series: pd.Series, periods: int) -> pd.Series:
    return series.pct_change(periods=periods)


def rolling_volatility(series: pd.Series, window: int) -> pd.Series:
    return series.pct_change().rolling(window).std()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window).mean()


def bollinger_position(close: pd.Series, window: int = 20, stds: float = 2.0) -> pd.Series:
    mean = close.rolling(window).mean()
    std = close.rolling(window).std()
    lower = mean - stds * std
    upper = mean + stds * std
    return (close - lower) / (upper - lower).replace(0, pd.NA)


def zscore(series: pd.Series, window: int = 252) -> pd.Series:
    mean = series.rolling(window, min_periods=max(20, window // 5)).mean()
    std = series.rolling(window, min_periods=max(20, window // 5)).std()
    return (series - mean) / std.replace(0, pd.NA)
