from __future__ import annotations

import time
import math

import pandas as pd

from app.providers.base import HttpProvider, ProviderError

_SPOT_BUY_KEYS = ("BUY_AMT_BOARD", "SPOT_BUY_AMT_BOARD")
_SPOT_SELL_KEYS = ("SELL_AMT_BOARD", "SPOT_SELL_AMT_BOARD")
_CASH_BUY_KEYS = ("CASH_BUY_AMT_BOARD", "CASH_BUY")
_CASH_SELL_KEYS = ("CASH_SELL_AMT_BOARD", "CASH_SELL")


def _first_present(row: dict, keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key in row:
            value = pd.to_numeric(row[key], errors="coerce")
            if pd.notna(value) and math.isfinite(value) and value > 0:
                return float(value)
    return None


def find_usd_spot_row(rows: list[dict]) -> dict:
    if not isinstance(rows, list):
        raise ProviderError("Hua Nan Bank response must be a list")
    for row in rows:
        if not isinstance(row, dict):
            continue
        kind = str(row.get("TYPE") or "").strip().lower()
        if kind not in {"spot", ""}:
            continue
        desc = f"{row.get('DESC_ENG', '')} {row.get('DESC_CHI', '')}".upper()
        if "USD" not in desc and "美金" not in desc:
            continue
        if "DAY" in desc or "天" in desc:
            continue
        if kind == "" and str(row.get("DESC_ENG", "")).strip().upper() != "USD":
            continue
        return row
    raise ProviderError("USD spot row not found in Hua Nan Bank rates")


class HuaNanBankProvider(HttpProvider):
    """Public spot USD rates from Hua Nan Commercial Bank's rate JSON endpoint.

    The endpoint was confirmed reachable during development, but Hua Nan's
    site enforces bot mitigation that can block scripted access without
    warning (same behavior observed from Bank of Taiwan). Callers should keep
    a fallback provider configured.
    """

    source = "hua_nan_bank"

    def fetch_current_rates(self) -> pd.DataFrame:
        response = self.get(
            "https://www.hncb.com.tw/hncb/rest/exRate/all",
            params={"_": str(int(time.time() * 1000))},
            headers={
                "Referer": "https://www.hncb.com.tw/wps/portal/HNCB/per_finance/query_forex_rate_all/query_forex_rate"
            },
        )
        try:
            rows = response.json()
        except ValueError as exc:
            raise ProviderError("Hua Nan Bank response was not valid JSON") from exc
        row = find_usd_spot_row(rows)
        buying = _first_present(row, _SPOT_BUY_KEYS)
        selling = _first_present(row, _SPOT_SELL_KEYS)
        if buying is None or selling is None or selling < buying:
            raise ProviderError("Hua Nan USD spot quote is invalid or crossed")
        try:
            observed = pd.Timestamp(f"{row['DATE']} {row['TIME']}").tz_localize("Asia/Taipei").tz_convert("UTC")
            if pd.isna(observed):
                raise ValueError("Missing quote timestamp")
        except (KeyError, ValueError, TypeError) as exc:
            raise ProviderError("Hua Nan quote timestamp missing or invalid") from exc
        return pd.DataFrame(
            [
                {
                    "bank_name": "Hua Nan Commercial Bank",
                    "currency": "USD",
                    "cash_buying": _first_present(row, _CASH_BUY_KEYS),
                    "cash_selling": _first_present(row, _CASH_SELL_KEYS),
                    "spot_buying": buying,
                    "spot_selling": selling,
                    "observed_at_utc": observed,
                }
            ]
        )
