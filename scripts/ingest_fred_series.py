from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path

import pandas as pd
import structlog
import typer

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.config import settings
from app.database.schema import MarketData
from app.database.session import make_session
from app.database.upsert import upsert_rows
from app.logging import configure_logging
from app.providers.fred import FredCsvProvider

cli = typer.Typer(help="Ingest one configured FRED series without running the full provider pipeline.")


SYMBOL_BY_SERIES_KEY = {
    "us_2y": "US_2Y",
    "us_10y": "US_10Y",
    "vix": "VIX",
    "sp500": "SP500",
    "nasdaq": "NASDAQ",
    "broad_usd_index": "BROAD_USD_INDEX",
    "usd_cny": "USD_CNY",
}


def _provider_kwargs(cfg: dict) -> dict:
    providers = cfg["providers"]
    return {
        "timeout": providers.get("default_timeout_seconds", 20),
        "attempts": providers.get("retry_attempts", 3),
        "wait_seconds": providers.get("retry_wait_seconds", 2),
    }


def _write_market_data(session, source: str, symbol: str, df: pd.DataFrame) -> int:
    rows = [
        {
            "observed_at_utc": item["date"],
            "source": source,
            "symbol": symbol,
            "open": None,
            "high": None,
            "low": None,
            "close": item["value"],
            "volume": None,
        }
        for item in df.to_dict("records")
    ]
    return upsert_rows(session, MarketData, rows, ("symbol", "observed_at_utc", "source"))


def _parse_fred_csv(csv_text: str, series_id: str) -> pd.DataFrame:
    df = pd.read_csv(StringIO(csv_text))
    df = df.rename(columns={"observation_date": "date", series_id: "value"})
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["value"] = pd.to_numeric(df["value"].replace(".", pd.NA), errors="coerce")
    df = df.dropna(subset=["value"])
    df["symbol"] = series_id
    return df[["date", "symbol", "value"]]


@cli.command()
def run(
    series_key: str = typer.Option("usd_cny", help="Key under providers.fred.series in config/settings.yaml."),
    symbol: str | None = typer.Option(None, help="Override stored market_data symbol."),
    timeout_seconds: int | None = typer.Option(None, help="Override provider timeout for large historical CSV downloads."),
    csv_path: str | None = typer.Option(None, help="Read a FRED CSV from a local path, or '-' for stdin."),
    database_url: str | None = typer.Option(None, help="Override DATABASE_URL/settings.yaml."),
) -> None:
    configure_logging()
    log = structlog.get_logger()
    cfg = settings()
    series_map = cfg["providers"]["fred"]["series"]
    if series_key not in series_map:
        valid = ", ".join(sorted(series_map))
        raise typer.BadParameter(f"Unknown FRED series_key '{series_key}'. Valid keys: {valid}")
    stored_symbol = symbol or SYMBOL_BY_SERIES_KEY.get(series_key) or series_key.upper()
    provider = None
    session = make_session(database_url or cfg["database"]["url"])
    if csv_path:
        csv_text = sys.stdin.read() if csv_path == "-" else Path(csv_path).read_text(encoding="utf-8")
        df = _parse_fred_csv(csv_text, series_map[series_key])
        source = "fred_csv"
    else:
        provider_kwargs = _provider_kwargs(cfg)
        if timeout_seconds is not None:
            provider_kwargs["timeout"] = timeout_seconds
        provider = FredCsvProvider(**provider_kwargs)
        df = provider.fetch_series(series_map[series_key])
        source = provider.source
    rows = _write_market_data(session, source, stored_symbol, df)
    log.info(
        "fred_series_ingested",
        series_key=series_key,
        series_id=series_map[series_key],
        symbol=stored_symbol,
        rows=rows,
        start_date=str(df["date"].min().date()) if not df.empty else None,
        end_date=str(df["date"].max().date()) if not df.empty else None,
    )


if __name__ == "__main__":
    cli()
