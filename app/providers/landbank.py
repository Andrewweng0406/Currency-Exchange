from __future__ import annotations

from io import StringIO

import pandas as pd

from app.providers.base import HttpProvider, ProviderError


class LandBankProvider(HttpProvider):
    source = "land_bank_taiwan"

    def fetch_current_rates(self) -> pd.DataFrame:
        response = self.get("https://rate.landbank.com.tw/en-US/Foreign", params={"mid": "69"})
        tables = pd.read_html(StringIO(response.text))
        if not tables:
            raise ProviderError("Land Bank page did not contain rate tables")
        table = tables[0]
        table.columns = ["currency", "spot_buying", "spot_selling", "cash_buying", "cash_selling", "historical"]
        usd = table[table["currency"].astype(str).str.upper().eq("USD")]
        if usd.empty:
            raise ProviderError("USD row not found in Land Bank rates")
        row = usd.iloc[0]
        return pd.DataFrame(
            [
                {
                    "bank_name": "Land Bank of Taiwan",
                    "currency": "USD",
                    "cash_buying": pd.to_numeric(row["cash_buying"], errors="coerce"),
                    "cash_selling": pd.to_numeric(row["cash_selling"], errors="coerce"),
                    "spot_buying": pd.to_numeric(row["spot_buying"], errors="coerce"),
                    "spot_selling": pd.to_numeric(row["spot_selling"], errors="coerce"),
                    "observed_at_utc": pd.Timestamp.utcnow(),
                }
            ]
        )
