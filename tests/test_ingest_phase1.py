import pandas as pd

from scripts import ingest_phase1


class FakeTreasuryProvider:
    source = "treasury_yield_curve_xml"

    def fetch_recent_yields(self, months=2):
        return pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-09-03"], utc=True),
                "US_2Y": [4.34],
                "US_10Y": [4.77],
            }
        )


def test_ingest_treasury_yields_writes_2y_and_10y(monkeypatch):
    calls = []

    def fake_write(session, source, symbol, frame):
        calls.append((source, symbol, frame.iloc[0]["value"]))
        return len(frame)

    monkeypatch.setattr(ingest_phase1, "_write_market_data", fake_write)
    rows = ingest_phase1._ingest_treasury_yields(object(), FakeTreasuryProvider())

    assert rows == 2
    assert calls == [
        ("treasury_yield_curve_xml", "US_2Y", 4.34),
        ("treasury_yield_curve_xml", "US_10Y", 4.77),
    ]
