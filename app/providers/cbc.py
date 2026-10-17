from __future__ import annotations

from io import StringIO

import pandas as pd

from app.providers.base import HttpProvider, ProviderError


class TaiwanCentralBankProvider(HttpProvider):
    source = "cbc_taiwan"
    history_url = "https://cpx.cbc.gov.tw/api/OpenData/FTDOpenData_Day"

    def fetch_usdtwd_closing(self) -> pd.DataFrame:
        """Use the official daily archive, rather than the recent HTML page."""
        response = self.get(self.history_url)
        try:
            return parse_daily_archive(response.json())
        except (ValueError, TypeError, KeyError) as exc:
            raise ProviderError("Invalid CBC daily archive response") from exc

    def fetch_recent_html(self) -> pd.DataFrame:
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


def parse_daily_archive(payload) -> pd.DataFrame:
    if not isinstance(payload, list) or not payload:
        raise ProviderError("CBC daily archive must be a nonempty list")
    frame = pd.DataFrame(payload)
    if not {"日期", "NTD_USD"}.issubset(frame):
        raise ProviderError("CBC daily archive fields missing")
    frame["date"] = pd.to_datetime(frame["日期"].astype(str), format="%Y%m%d", utc=True, errors="coerce")
    frame["close"] = pd.to_numeric(frame["NTD_USD"], errors="coerce")
    if frame["date"].isna().any() or frame["close"].isna().any() or not frame["close"].between(0, float("inf"), inclusive="neither").all():
        raise ProviderError("CBC daily archive contains invalid dates or prices")
    if frame.groupby("date")["close"].nunique().gt(1).any():
        raise ProviderError("CBC daily archive has conflicting duplicate dates")
    frame["pair"] = "USD/TWD"
    return frame[["date", "pair", "close"]].drop_duplicates("date").sort_values("date").reset_index(drop=True)
