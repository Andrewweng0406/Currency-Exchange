from datetime import date

import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.database.schema import Base, MarketData
from scripts.backfill_twse_history import _backfill_market_month, month_starts


class _Log:
    def __init__(self):
        self.warnings = []

    def warning(self, event, **kwargs):
        self.warnings.append((event, kwargs))


class _Provider:
    source = "twse"

    def fetch_taiex_month(self, month):
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


def test_backfill_help_exposes_empty_month_guard():
    from typer.testing import CliRunner

    from scripts.backfill_twse_history import cli

    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "empty/failed market" in result.output
