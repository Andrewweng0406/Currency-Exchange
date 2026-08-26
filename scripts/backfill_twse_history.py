from __future__ import annotations

import sys
import time
from datetime import date, timedelta
from pathlib import Path

import structlog
import typer
from sqlalchemy import select

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.config import settings
from app.database.schema import ForeignFlow, MarketData
from app.database.session import make_session
from app.database.upsert import upsert_rows
from app.logging import configure_logging
from app.providers.twse import TwseProvider

cli = typer.Typer(help="Conservative TWSE historical backfill.")


def month_starts(start: date, end: date) -> list[date]:
    current = date(start.year, start.month, 1)
    values = []
    while current <= end:
        values.append(current)
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)
    return values


def _write_market(session, provider: TwseProvider, symbol: str, df):
    rows = [
        {
            "observed_at_utc": row["date"],
            "source": provider.source,
            "symbol": symbol,
            "open": row.get("open"),
            "high": row.get("high"),
            "low": row.get("low"),
            "close": row.get("close"),
            "volume": row.get("volume"),
        }
        for row in df.to_dict("records")
    ]
    return upsert_rows(session, MarketData, rows, ("symbol", "observed_at_utc", "source"))


def _trading_dates_from_taiex(session, start: date, end: date) -> list[date]:
    rows = session.execute(
        select(MarketData.observed_at_utc)
        .where(MarketData.symbol == "TAIEX")
        .where(MarketData.observed_at_utc >= start)
        .where(MarketData.observed_at_utc <= end)
        .order_by(MarketData.observed_at_utc)
    ).all()
    return [row[0].date() for row in rows]


def _existing_foreign_flow_dates(session) -> set[date]:
    rows = session.execute(select(ForeignFlow.observed_at_utc).where(ForeignFlow.market == "TWSE")).all()
    return {row[0].date() for row in rows}


@cli.command()
def run(
    years: int = typer.Option(1, help="Years to backfill conservatively."),
    include_foreign_flow: bool = typer.Option(True, help="Backfill daily T86 foreign-flow endpoint."),
    sleep_seconds: float = typer.Option(0.5, help="Delay between TWSE requests."),
) -> None:
    configure_logging()
    log = structlog.get_logger()
    cfg = settings()
    session = make_session(cfg["database"]["url"])
    provider = TwseProvider(
        timeout=cfg["providers"].get("default_timeout_seconds", 20),
        attempts=cfg["providers"].get("retry_attempts", 3),
        wait_seconds=cfg["providers"].get("retry_wait_seconds", 2),
    )
    end = date.today()
    start = end - timedelta(days=365 * years + 31)

    for month in month_starts(start, end):
        taiex_rows = _write_market(session, provider, "TAIEX", provider.fetch_taiex_month(month))
        tsmc_rows = _write_market(
            session, provider, "2330.TW", provider.fetch_stock_month(cfg["providers"]["twse"]["tsmc_stock_no"], month)
        )
        log.info("twse_month_backfilled", month=month.isoformat(), taiex_rows=taiex_rows, tsmc_rows=tsmc_rows)
        time.sleep(sleep_seconds)

    if include_foreign_flow:
        existing = _existing_foreign_flow_dates(session)
        for current in _trading_dates_from_taiex(session, start, end):
            if current in existing:
                continue
            try:
                df = provider.fetch_foreign_flow(current)
                rows = [
                    {
                        "observed_at_utc": row["date"],
                        "source": provider.source,
                        "market": row["market"],
                        "foreign_net_buy_sell_shares": row["foreign_net_buy_sell_shares"],
                        "raw_payload": None,
                    }
                    for row in df.to_dict("records")
                ]
                count = upsert_rows(session, ForeignFlow, rows, ("market", "observed_at_utc", "source"))
                log.info("twse_foreign_flow_backfilled", date=current.isoformat(), rows=count)
            except Exception as exc:  # noqa: BLE001
                log.warning("twse_foreign_flow_skipped", date=current.isoformat(), error=str(exc))
            time.sleep(sleep_seconds)


if __name__ == "__main__":
    cli()
