from __future__ import annotations

from io import BytesIO
from io import StringIO
from zipfile import ZipFile

import pandas as pd
from requests.utils import default_user_agent

from app.providers.base import HttpProvider, ProviderError


class FredCsvProvider(HttpProvider):
    source = "fred_csv"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # FRED graph exports time out for the shared application's custom UA
        # from production, while the standard requests UA receives the CSV.
        self.session.headers["User-Agent"] = default_user_agent()

    def fetch_series(self, series_id: str) -> pd.DataFrame:
        response = self.get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": series_id})
        df = pd.read_csv(StringIO(_csv_text(response)))
        if not {"observation_date", series_id}.issubset(df.columns):
            raise ProviderError(f"FRED response missing columns for {series_id}")
        df = df.rename(columns={"observation_date": "date", series_id: "value"})
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df["value"] = pd.to_numeric(df["value"].replace(".", pd.NA), errors="coerce")
        df = df.dropna(subset=["value"])
        df["symbol"] = series_id
        return df[["date", "symbol", "value"]]

    def fetch_series_batch(self, series: dict[str, str]) -> dict[str, pd.DataFrame]:
        response = self.get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": ",".join(series.values())})
        raw_frames = [pd.read_csv(StringIO(value)) for value in _csv_texts(response)]
        for raw in raw_frames:
            if "observation_date" not in raw:
                raise ProviderError("FRED batch response missing observation_date")
            raw["date"] = pd.to_datetime(raw["observation_date"], utc=True)
        frames = {}
        for name, series_id in series.items():
            candidates = [raw for raw in raw_frames if series_id in raw]
            if not candidates:
                continue
            if len(candidates) != 1:
                raise ProviderError(f"FRED batch contains ambiguous copies of {series_id}")
            raw = candidates[0]
            df = raw[["date", series_id]].rename(columns={series_id: "value"}).copy()
            df["value"] = pd.to_numeric(df["value"].replace(".", pd.NA), errors="coerce")
            df = df.dropna(subset=["value"])
            df["symbol"] = series_id
            frames[name] = df[["date", "symbol", "value"]]
        return frames


def _csv_text(response) -> str:
    return _csv_texts(response)[0]


def _csv_texts(response) -> list[str]:
    content = getattr(response, "content", b"") or b""
    if content.startswith(b"PK"):
        with ZipFile(BytesIO(content)) as archive:
            names = sorted((name for name in archive.namelist() if name.endswith(".csv")), key=lambda name: (name != "daily.csv", name))
            if not names:
                raise ProviderError("FRED zip response did not contain a CSV file")
            return [archive.read(name).decode("utf-8-sig") for name in names]
    return [response.text]
