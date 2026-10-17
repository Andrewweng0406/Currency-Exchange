from datetime import date
from io import BytesIO
from zipfile import ZipFile
from types import SimpleNamespace

import pandas as pd
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.database.schema import Base, MarketData
from app.providers.base import ProviderError
from app.providers.fred import FredCsvProvider
from app.providers.twse import TwseProvider
from scripts.ingest_phase1 import _ingest_fred_with_fallback


def test_fred_uses_standard_requests_header_without_changing_other_providers():
    from requests.utils import default_user_agent
    fred = FredCsvProvider(timeout=9, attempts=1)
    assert fred.session.headers["User-Agent"] == default_user_agent()
    assert fred.timeout == 9
    assert "twd-fx-monitor" in TwseProvider().session.headers["User-Agent"]


@pytest.mark.parametrize("failed_series", [None, "DEXCHUS"])
def test_fred_batch_timeout_recovers_individual_series_and_preserves_partial_success(failed_series):
    class Provider:
        source = "fred_csv"
        def fetch_series_batch(self, series):
            raise ProviderError("batch timed out")
        def fetch_series(self, series_id):
            if series_id == failed_series:
                raise ProviderError("single timed out")
            return pd.DataFrame({"date": [pd.Timestamp("2026-10-09", tz="UTC")], "value": [6.7]})
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        results, written = _ingest_fred_with_fallback(session, Provider(), {"broad_usd_index": "DTWEXBGS", "usd_cny": "DEXCHUS"})
        rows = session.scalars(select(MarketData)).all()
        assert "broad_usd_index" in written
        assert all(r.source == "fred_csv" for r in rows)
        assert len(rows) == (1 if failed_series else 2)
        assert results[0].ok
        assert results[1].ok == (failed_series is None)


def test_fred_single_series_parses_official_zip_and_drops_missing_values(monkeypatch):
    content = BytesIO()
    with ZipFile(content, "w") as archive:
        archive.writestr("daily.csv", "observation_date,DEXCHUS\n2026-10-08,6.7\n2026-10-09,.\n")
    provider = FredCsvProvider()
    monkeypatch.setattr(provider, "get", lambda *a, **k: SimpleNamespace(content=content.getvalue()))
    frame = provider.fetch_series("DEXCHUS")
    assert frame.value.tolist() == [6.7]
    assert frame.date.dt.strftime("%Y-%m-%d").tolist() == ["2026-10-08"]


def test_fred_batch_reads_both_daily_and_daily_close_members(monkeypatch):
    content = BytesIO()
    with ZipFile(content, "w") as archive:
        archive.writestr("daily.csv", "observation_date,DGS2,DEXCHUS\n2026-10-08,3.7,6.7\n")
        archive.writestr("daily,_close.csv", "observation_date,VIXCLS\n2026-10-08,18.2\n")
    provider = FredCsvProvider()
    monkeypatch.setattr(provider, "get", lambda *a, **k: SimpleNamespace(content=content.getvalue()))
    frames = provider.fetch_series_batch({"us_2y": "DGS2", "usd_cny": "DEXCHUS", "vix": "VIXCLS"})
    assert set(frames) == {"us_2y", "usd_cny", "vix"}
    assert frames["vix"].value.tolist() == [18.2]


def test_weekend_foreign_flow_preserves_friday_date(monkeypatch):
    provider = TwseProvider()
    calls = []
    def fetch(target):
        calls.append(target)
        return pd.DataFrame({"date": [pd.Timestamp(target, tz="UTC")]})
    monkeypatch.setattr(provider, "fetch_foreign_flow", fetch)
    frame = provider.fetch_latest_foreign_flow(date(2026, 10, 17))
    assert calls == [date(2026, 10, 16)]
    assert frame.date.iloc[0] == pd.Timestamp("2026-10-16", tz="UTC")


def test_holiday_lookup_rolls_back_but_network_errors_are_not_hidden(monkeypatch):
    provider = TwseProvider()
    calls = []
    def fetch(target):
        calls.append(target)
        if len(calls) == 1:
            raise ProviderError("TWSE T86 returned non-OK status: 很抱歉，沒有符合條件的資料!")
        return pd.DataFrame({"date": [pd.Timestamp(target, tz="UTC")]})
    monkeypatch.setattr(provider, "fetch_foreign_flow", fetch)
    frame = provider.fetch_latest_foreign_flow(date(2026, 10, 12))
    assert calls == [date(2026, 10, 12), date(2026, 10, 9)]
    assert frame.date.iloc[0].date() == date(2026, 10, 9)
    def fail(target):
        raise ProviderError("HTTP 503")
    monkeypatch.setattr(provider, "fetch_foreign_flow", fail)
    with pytest.raises(ProviderError, match="HTTP 503"):
        provider.fetch_latest_foreign_flow(date(2026, 10, 12))


@pytest.mark.parametrize("payload", [
    {"stat": "OK", "data": []},
    {"stat": "OK", "date": "20261015", "data": [["0"]]},
])
def test_foreign_flow_rejects_empty_or_wrong_day_response(payload, monkeypatch):
    provider = TwseProvider()
    monkeypatch.setattr(provider, "get", lambda *a, **k: SimpleNamespace(json=lambda: payload))
    with pytest.raises(ProviderError):
        provider.fetch_foreign_flow(date(2026, 10, 16))
