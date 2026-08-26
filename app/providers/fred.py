from __future__ import annotations

from io import StringIO

import pandas as pd

from app.providers.base import HttpProvider


class FredCsvProvider(HttpProvider):
    source = "fred_csv"

    def fetch_series(self, series_id: str) -> pd.DataFrame:
        response = self.get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": series_id})
        df = pd.read_csv(StringIO(response.text))
        df = df.rename(columns={"observation_date": "date", series_id: "value"})
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df["value"] = pd.to_numeric(df["value"].replace(".", pd.NA), errors="coerce")
        df = df.dropna(subset=["value"])
        df["symbol"] = series_id
        return df[["date", "symbol", "value"]]
