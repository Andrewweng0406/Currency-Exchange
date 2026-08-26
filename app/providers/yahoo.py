from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from app.providers.base import ProviderError


class YahooFinanceProvider:
    source = "yahoo_finance"

    def fetch_ohlcv(self, symbol: str, years: int = 10) -> pd.DataFrame:
        start = date.today() - timedelta(days=365 * years + 10)
        try:
            df = yf.download(symbol, start=start.isoformat(), progress=False, auto_adjust=False, threads=False)
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"Yahoo Finance download failed for {symbol}: {exc}") from exc
        if df.empty:
            raise ProviderError(f"Yahoo Finance returned no rows for {symbol}")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df = df.reset_index().rename(
            columns={"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
        )
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df["symbol"] = symbol
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df.get(col), errors="coerce")
        return df[["date", "symbol", "open", "high", "low", "close", "volume"]].dropna(subset=["close"])
