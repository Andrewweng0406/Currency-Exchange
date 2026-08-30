from datetime import date

import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.database.schema import Base, MarketData
from scripts.backfill_twse_history import _backfill_market_month, month_starts


class _Log:
    def __init__(self):
        self.warnings = []
        self.infos = []

    def warning(self, event, **kwargs):
        self.warnings.append((event, kwargs))

    def info(self, event, **kwargs):
        self.infos.append((event, kwargs))


class _Provider:
    source = "twse"
    taiex_calls = 0

    def fetch_taiex_month(self, month):
        self.taiex_calls += 1
        raise RuntimeError("rate limited")

    def fetch_stock_month(self, stock_no, month):
        return pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2026-01-02", tz="UTC"),
                    "open": 100,
                    "high": 110,
                    "low": 95,
                    "close": 105,
                    "volume": 1000,
                }
            ]
        )


def test_month_starts_inclusive():
    assert month_starts(date(2026, 1, 15), date(2026, 3, 1)) == [
        date(2026, 1, 1),
        date(2026, 2, 1),
        date(2026, 3, 1),
    ]


def test_backfill_market_month_continues_after_taiex_failure(monkeypatch):
    monkeypatch.setattr("scripts.backfill_twse_history.time.sleep", lambda _: None)
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    log = _Log()
    cfg = {"providers": {"twse": {"tsmc_stock_no": "2330"}}}

    with Session(engine) as session:
        taiex_rows, tsmc_rows = _backfill_market_month(session, _Provider(), cfg, date(2026, 1, 1), 0, log)
        row = session.execute(select(MarketData).where(MarketData.symbol == "2330.TW")).scalar_one()

    assert taiex_rows == 0
    assert tsmc_rows == 1
    assert row.close == 105
    assert log.warnings[0][0] == "twse_taiex_month_skipped"


def test_backfill_market_month_uses_yahoo_fallback_when_twse_fails(monkeypatch):
    monkeypatch.setattr("scripts.backfill_twse_history.time.sleep", lambda _: None)
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    log = _Log()
    cfg = {"providers": {"twse": {"tsmc_stock_no": "2330"}}}
    yahoo_cache = {
        "TAIEX": pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2026-01-05", tz="UTC"),
                    "open": 18000,
                    "high": 18100,
                    "low": 17900,
                    "close": 18050,
                    "volume": 1000,
                },
                {
                    "date": pd.Timestamp("2026-02-02", tz="UTC"),
                    "open": 19000,
                    "high": 19100,
                    "low": 18900,
                    "close": 19050,
                    "volume": 1000,
                },
            ]
        )
    }

    with Session(engine) as session:
        taiex_rows, _ = _backfill_market_month(session, _Provider(), cfg, date(2026, 1, 1), 0, log, yahoo_cache)
        row = session.execute(select(MarketData).where(MarketData.symbol == "TAIEX")).scalar_one()

    assert taiex_rows == 1
    assert row.source == "yahoo_finance"
    assert row.close == 18050
    assert log.infos[0][0] == "yahoo_taiex_month_backfilled"


def test_backfill_market_month_can_skip_twse_and_use_yahoo(monkeypatch):
    monkeypatch.setattr("scripts.backfill_twse_history.time.sleep", lambda _: None)
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    log = _Log()
    cfg = {"providers": {"twse": {"tsmc_stock_no": "2330"}}}
    provider = _Provider()
    yahoo_cache = {
        "TAIEX": pd.DataFrame(
            [{"date": pd.Timestamp("2026-01-05", tz="UTC"), "open": 1, "high": 1, "low": 1, "close": 18050, "volume": 1}]
        ),
        "2330.TW": pd.DataFrame(
            [{"date": pd.Timestamp("2026-01-05", tz="UTC"), "open": 1, "high": 1, "low": 1, "close": 650, "volume": 1}]
        ),
    }

    with Session(engine) as session:
        taiex_rows, tsmc_rows = _backfill_market_month(session, provider, cfg, date(2026, 1, 1), 0, log, yahoo_cache, False)

    assert taiex_rows == 1
    assert tsmc_rows == 1
    assert provider.taiex_calls == 0


def test_backfill_help_exposes_empty_month_guard():
    from typer.testing import CliRunner

    from scripts.backfill_twse_history import cli

    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "empty/failed market" in result.output
    assert "twse-enabled" in result.output
