from __future__ import annotations

from io import StringIO

import pandas as pd

from app.providers.base import HttpProvider, ProviderError


class TaiwanCentralBankProvider(HttpProvider):
    source = "cbc_taiwan"

    def fetch_usdtwd_closing(self) -> pd.DataFrame:
        response = self.get("https://www.cbc.gov.tw/en/lp-700-2.html")
        tables = pd.read_html(StringIO(response.text))
        for table in tables:
            cols = [str(c).strip() for c in table.columns]
            if "Date" in cols and any("NTD" in c or "USD" in c for c in cols):
                value_col = [c for c in cols if c != "Date"][0]
                df = table.rename(columns={value_col: "close"})
                df["date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
                df["close"] = pd.to_numeric(df["close"], errors="coerce")
                df["pair"] = "USD/TWD"
                return df[["date", "pair", "close"]].dropna()
        raise ProviderError("CBC USD/TWD closing-rate table not found")
