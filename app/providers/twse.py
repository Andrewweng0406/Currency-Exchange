from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from app.providers.base import HttpProvider, ProviderError


def roc_date_to_timestamp(value: str) -> pd.Timestamp:
    year, month, day = value.split("/")
    return pd.Timestamp(year=int(year) + 1911, month=int(month), day=int(day), tz="UTC")


def twse_date(value: date | None = None) -> str:
    value = value or date.today()
    return value.strftime("%Y%m%d")


def parse_number(value: str | int | float | None) -> float | None:
    if value in (None, "", "--"):
        return None
    parsed = pd.to_numeric(str(value).replace(",", "").replace("+", ""), errors="coerce")
    return None if pd.isna(parsed) else float(parsed)


class TwseProvider(HttpProvider):
    source = "twse"

    def fetch_latest_foreign_flow(self, as_of: date | None = None) -> pd.DataFrame:
        # Daily automation often runs before the next Taiwan session or on a
        # holiday. Preserve the actual observation date; never relabel it today.
        as_of = as_of or datetime.now(ZoneInfo("Asia/Taipei")).date()
        for offset in range(15):
            candidate = as_of - timedelta(days=offset)
            if candidate.weekday() >= 5:
                continue
            try:
                return self.fetch_foreign_flow(candidate)
            except ProviderError as exc:
                if "沒有符合條件的資料" not in str(exc):
                    raise
        raise ProviderError("TWSE has no foreign-flow observations in the last 15 calendar days")

    def fetch_foreign_flow(self, target_date: date | None = None) -> pd.DataFrame:
        target_date = target_date or datetime.now(ZoneInfo("Asia/Taipei")).date()
        response = self.get(
            "https://www.twse.com.tw/rwd/zh/fund/T86",
            params={"date": twse_date(target_date), "selectType": "ALLBUT0999", "response": "json"},
        )
        payload = response.json()
        if payload.get("stat") != "OK":
            raise ProviderError(f"TWSE T86 returned non-OK status: {payload.get('stat')}")
        if not payload.get("data"):
            raise ProviderError("TWSE T86 returned an empty OK response")
        observed = payload.get("date")
        if observed is not None and observed != twse_date(target_date):
            raise ProviderError("TWSE T86 response date does not match requested date")
        idx = payload["fields"].index("外陸資買賣超股數(不含外資自營商)")
        net = sum(parse_number(row[idx]) or 0 for row in payload.get("data", []))
        return pd.DataFrame(
            [{"date": pd.Timestamp(target_date, tz="UTC"), "market": "TWSE", "foreign_net_buy_sell_shares": net}]
        )

    def fetch_taiex_month(self, target_date: date | None = None) -> pd.DataFrame:
        response = self.get(
            "https://www.twse.com.tw/rwd/zh/TAIEX/MI_5MINS_HIST",
            params={"date": twse_date(target_date), "response": "json"},
        )
        payload = response.json()
        if payload.get("stat") != "OK":
            raise ProviderError(f"TWSE TAIEX returned non-OK status: {payload.get('stat')}")
        rows = []
        for row in payload.get("data", []):
            rows.append(
                {
                    "date": roc_date_to_timestamp(row[0]),
                    "symbol": "TAIEX",
                    "open": parse_number(row[1]),
                    "high": parse_number(row[2]),
                    "low": parse_number(row[3]),
                    "close": parse_number(row[4]),
                    "volume": None,
                }
            )
        return pd.DataFrame(rows)

    def fetch_stock_month(self, stock_no: str, target_date: date | None = None) -> pd.DataFrame:
        response = self.get(
            "https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY",
            params={"date": twse_date(target_date), "stockNo": stock_no, "response": "json"},
        )
        payload = response.json()
        if payload.get("stat") != "OK":
            raise ProviderError(f"TWSE STOCK_DAY returned non-OK status: {payload.get('stat')}")
        rows = []
        for row in payload.get("data", []):
            rows.append(
                {
                    "date": roc_date_to_timestamp(row[0]),
                    "symbol": f"{stock_no}.TW",
                    "open": parse_number(row[3]),
                    "high": parse_number(row[4]),
                    "low": parse_number(row[5]),
                    "close": parse_number(row[6]),
                    "volume": parse_number(row[1]),
                }
            )
        return pd.DataFrame(rows)
