from __future__ import annotations

from datetime import date
from xml.etree import ElementTree

import pandas as pd

from app.providers.base import HttpProvider, ProviderError


class TreasuryYieldCurveProvider(HttpProvider):
    source = "treasury_yield_curve_xml"
    _URL = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml"
    _NS = {
        "atom": "http://www.w3.org/2005/Atom",
        "d": "http://schemas.microsoft.com/ado/2007/08/dataservices",
        "m": "http://schemas.microsoft.com/ado/2007/08/dataservices/metadata",
    }

    def fetch_recent_yields(self, months: int = 2) -> pd.DataFrame:
        frames = []
        for month in _recent_months(months):
            frames.append(self.fetch_yield_curve_month(month))
        if not frames:
            return pd.DataFrame(columns=["date", "US_2Y", "US_10Y"])
        return pd.concat(frames, ignore_index=True).sort_values("date").drop_duplicates("date", keep="last")

    def fetch_yield_curve_month(self, yyyymm: str) -> pd.DataFrame:
        response = self.get(
            self._URL,
            params={"data": "daily_treasury_yield_curve", "field_tdr_date_value_month": yyyymm},
        )
        return parse_yield_curve_xml(response.text)


def parse_yield_curve_xml(xml_text: str) -> pd.DataFrame:
    root = ElementTree.fromstring(xml_text)
    rows = []
    for entry in root.findall("atom:entry", TreasuryYieldCurveProvider._NS):
        props = entry.find("atom:content/m:properties", TreasuryYieldCurveProvider._NS)
        if props is None:
            continue
        observed = _text(props, "NEW_DATE")
        us2y = _float(_text(props, "BC_2YEAR"))
        us10y = _float(_text(props, "BC_10YEAR"))
        if observed is None:
            continue
        rows.append({"date": pd.to_datetime(observed, utc=True), "US_2Y": us2y, "US_10Y": us10y})
    if not rows:
        raise ProviderError("Treasury yield curve XML returned no usable rows")
    return pd.DataFrame(rows).dropna(subset=["US_2Y", "US_10Y"], how="all")


def _text(props: ElementTree.Element, tag: str) -> str | None:
    child = props.find(f"d:{tag}", TreasuryYieldCurveProvider._NS)
    return child.text if child is not None else None


def _float(value: str | None) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _recent_months(count: int) -> list[str]:
    today = date.today()
    year = today.year
    month = today.month
    out = []
    for _ in range(max(1, count)):
        out.append(f"{year:04d}{month:02d}")
        month -= 1
        if month == 0:
            year -= 1
            month = 12
    return out
