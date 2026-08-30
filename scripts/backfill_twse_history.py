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
from app.providers.yahoo import YahooFinanceProvider

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


def _write_market(session, source: str, symbol: str, df):
    rows = [
        {
            "observed_at_utc": row["date"],
            "source": source,
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


def _month_slice(df, month: date):
    if df is None or df.empty:
        return df
    dates = df["date"].dt.date
    return df[(dates >= month) & (dates < _next_month(month))]


def _next_month(month: date) -> date:
    if month.month == 12:
        return date(month.year + 1, 1, 1)
    return date(month.year, month.month + 1, 1)


def _backfill_market_month(
    session,
    provider: TwseProvider,
    cfg: dict,
    month: date,
    sleep_seconds: float,
    log,
    yahoo_cache: dict[str, object] | None = None,
    twse_enabled: bool = True,
) -> tuple[int, int]:
    taiex_rows = 0
    tsmc_rows = 0
    if twse_enabled:
        try:
            taiex_rows = _write_market(session, provider.source, "TAIEX", provider.fetch_taiex_month(month))
        except Exception as exc:  # noqa: BLE001
            log.warning("twse_taiex_month_skipped", month=month.isoformat(), error=str(exc))
    if taiex_rows == 0:
        yahoo_rows = _month_slice((yahoo_cache or {}).get("TAIEX"), month)
        if yahoo_rows is not None and not yahoo_rows.empty:
            taiex_rows = _write_market(session, "yahoo_finance", "TAIEX", yahoo_rows)
            log.info("yahoo_taiex_month_backfilled", month=month.isoformat(), rows=taiex_rows)
    time.sleep(sleep_seconds)
    if twse_enabled:
        try:
            tsmc_rows = _write_market(
                session, provider.source, "2330.TW", provider.fetch_stock_month(cfg["providers"]["twse"]["tsmc_stock_no"], month)
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("twse_tsmc_month_skipped", month=month.isoformat(), error=str(exc))
    if tsmc_rows == 0:
        yahoo_rows = _month_slice((yahoo_cache or {}).get("2330.TW"), month)
        if yahoo_rows is not None and not yahoo_rows.empty:
            tsmc_rows = _write_market(session, "yahoo_finance", "2330.TW", yahoo_rows)
            log.info("yahoo_tsmc_month_backfilled", month=month.isoformat(), rows=tsmc_rows)
    return taiex_rows, tsmc_rows


@cli.command()
def run(
    years: int = typer.Option(1, help="Years to backfill conservatively."),
    include_foreign_flow: bool = typer.Option(True, help="Backfill daily T86 foreign-flow endpoint."),
    sleep_seconds: float = typer.Option(0.5, help="Delay between TWSE requests."),
    max_consecutive_empty_months: int = typer.Option(3, help="Stop after this many empty/failed market months."),
    yahoo_fallback: bool = typer.Option(True, help="Use Yahoo Finance fallback for TAIEX/2330 months when TWSE is unavailable."),
    twse_enabled: bool = typer.Option(True, help="Try TWSE official monthly endpoints before Yahoo fallback."),
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
    yahoo_cache = _load_yahoo_market_cache(years, log) if yahoo_fallback else {}

    consecutive_empty_months = 0
    for month in month_starts(start, end):
        log.info("twse_market_month_started", month=month.isoformat(), twse_enabled=twse_enabled, yahoo_fallback=bool(yahoo_cache))
        taiex_rows, tsmc_rows = _backfill_market_month(session, provider, cfg, month, sleep_seconds, log, yahoo_cache, twse_enabled)
        log.info("twse_month_backfilled", month=month.isoformat(), taiex_rows=taiex_rows, tsmc_rows=tsmc_rows)
        if taiex_rows == 0 and tsmc_rows == 0:
            consecutive_empty_months += 1
            if consecutive_empty_months >= max_consecutive_empty_months:
                log.warning(
                    "twse_backfill_stopped_after_empty_months",
                    consecutive_empty_months=consecutive_empty_months,
                    last_month=month.isoformat(),
                )
                break
        else:
            consecutive_empty_months = 0
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


def _load_yahoo_market_cache(years: int, log) -> dict[str, object]:
    provider = YahooFinanceProvider()
    cache = {}
    symbols = {"TAIEX": "^TWII", "2330.TW": "2330.TW"}
    for stored_symbol, yahoo_symbol in symbols.items():
        try:
            cache[stored_symbol] = provider.fetch_ohlcv(yahoo_symbol, years=years)
            log.info("yahoo_market_history_loaded", symbol=stored_symbol, yahoo_symbol=yahoo_symbol, rows=len(cache[stored_symbol]))
        except Exception as exc:  # noqa: BLE001
            log.warning("yahoo_market_history_unavailable", symbol=stored_symbol, yahoo_symbol=yahoo_symbol, error=str(exc))
    return cache


if __name__ == "__main__":
    cli()
