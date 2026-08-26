from __future__ import annotations

from io import StringIO

import pandas as pd

from app.providers.base import HttpProvider, ProviderError


class BankOfTaiwanProvider(HttpProvider):
    source = "bank_of_taiwan"

    def fetch_current_rates(self) -> pd.DataFrame:
        response = self.get("https://rate.bot.com.tw/xrt?Lang=en-US")
        tables = pd.read_html(StringIO(response.text))
        if not tables:
            raise ProviderError("Bank of Taiwan page did not contain rate tables")
        table = tables[0]
        table.columns = [str(col[-1] if isinstance(col, tuple) else col).strip() for col in table.columns]
        rows = []
        for _, row in table.iterrows():
            currency_text = " ".join(str(v) for v in row.tolist()[:2])
            if "American Dollar" not in currency_text and "USD" not in currency_text:
                continue
            numeric = [pd.to_numeric(v, errors="coerce") for v in row.tolist()]
            values = [float(v) for v in numeric if pd.notna(v)]
            if len(values) < 4:
                raise ProviderError("Could not parse USD cash/spot rates from Bank of Taiwan table")
            rows.append(
                {
                    "bank_name": "Bank of Taiwan",
                    "currency": "USD",
                    "cash_buying": values[0],
                    "cash_selling": values[1],
                    "spot_buying": values[2],
                    "spot_selling": values[3],
                    "observed_at_utc": pd.Timestamp.utcnow(),
                }
            )
        if not rows:
            raise ProviderError("USD row not found in Bank of Taiwan rates")
        return pd.DataFrame(rows)
