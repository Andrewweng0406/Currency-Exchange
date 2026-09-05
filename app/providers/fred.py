from __future__ import annotations

from io import BytesIO
from io import StringIO
from zipfile import ZipFile

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

    def fetch_series_batch(self, series: dict[str, str]) -> dict[str, pd.DataFrame]:
        response = self.get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": ",".join(series.values())})
        raw = pd.read_csv(StringIO(_csv_text(response)))
        raw["date"] = pd.to_datetime(raw["observation_date"], utc=True)
        frames = {}
        for name, series_id in series.items():
            if series_id not in raw:
                continue
            df = raw[["date", series_id]].rename(columns={series_id: "value"}).copy()
            df["value"] = pd.to_numeric(df["value"].replace(".", pd.NA), errors="coerce")
            df = df.dropna(subset=["value"])
            df["symbol"] = series_id
            frames[name] = df[["date", "symbol", "value"]]
        return frames


def _csv_text(response) -> str:
    content = getattr(response, "content", b"") or b""
    if content.startswith(b"PK"):
        with ZipFile(BytesIO(content)) as archive:
            names = archive.namelist()
            csv_name = "daily.csv" if "daily.csv" in names else next((name for name in names if name.endswith(".csv")), None)
            if csv_name is None:
                raise ProviderError("FRED zip response did not contain a CSV file")
            return archive.read(csv_name).decode("utf-8")
    return response.text
