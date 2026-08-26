from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import structlog
import typer

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.config import settings
from app.database.schema import BankRate, ForeignFlow, FxPrice, MarketData
from app.database.session import make_session
from app.database.upsert import upsert_rows
from app.logging import configure_logging
from app.providers.base import ProviderError, ProviderResult
from app.providers.bot import BankOfTaiwanProvider
from app.providers.cbc import TaiwanCentralBankProvider
from app.providers.fred import FredCsvProvider
from app.providers.landbank import LandBankProvider
from app.providers.twse import TwseProvider
from app.providers.yahoo import YahooFinanceProvider

cli = typer.Typer(help="Phase 1 data ingestion for real production sources.")


def _common_provider_kwargs(cfg: dict) -> dict:
    providers = cfg["providers"]
    return {
        "timeout": providers.get("default_timeout_seconds", 20),
        "attempts": providers.get("retry_attempts", 3),
        "wait_seconds": providers.get("retry_wait_seconds", 2),
    }


def _write_market_data(session, source: str, symbol: str, df: pd.DataFrame) -> int:
    rows = []
    for item in df.to_dict("records"):
        rows.append(
            {
                "observed_at_utc": item["date"],
                "source": source,
                "symbol": symbol,
                "open": item.get("open"),
                "high": item.get("high"),
                "low": item.get("low"),
                "close": item.get("close") if "close" in item else item.get("value"),
                "volume": item.get("volume"),
            }
        )
    return upsert_rows(session, MarketData, rows, ("symbol", "observed_at_utc", "source"))


def _result(name: str, fn) -> ProviderResult:
    try:
        rows = fn()
        return ProviderResult(name=name, ok=True, rows=rows)
    except Exception as exc:  # noqa: BLE001
        return ProviderResult(name=name, ok=False, error=str(exc))


def _ingest_bank_rates(session, provider) -> int:
    return upsert_rows(
        session,
        BankRate,
        [
            {
                "observed_at_utc": row["observed_at_utc"],
                "source": provider.source,
                "bank_name": row["bank_name"],
                "currency": row["currency"],
                "cash_buying": row["cash_buying"],
                "cash_selling": row["cash_selling"],
                "spot_buying": row["spot_buying"],
                "spot_selling": row["spot_selling"],
            }
            for row in provider.fetch_current_rates().to_dict("records")
        ],
        ("bank_name", "currency", "observed_at_utc", "source"),
    )


def _ingest_bank_with_fallback(session, provider_kwargs: dict) -> int:
    cfg = settings()
    bank_cfg = cfg["providers"].get("bank", {})
    registry = {"bot": BankOfTaiwanProvider, "landbank": LandBankProvider}
    primary_key = bank_cfg.get("provider", "bot")
    fallback_key = bank_cfg.get("fallback_provider", "landbank")
    primary = registry.get(primary_key, BankOfTaiwanProvider)(**provider_kwargs)
    try:
        return _ingest_bank_rates(session, primary)
    except Exception as primary_exc:  # noqa: BLE001
        fallback = registry.get(fallback_key, LandBankProvider)(**provider_kwargs)
        try:
            return _ingest_bank_rates(session, fallback)
        except Exception as fallback_exc:  # noqa: BLE001
            raise ProviderError(
                f"primary bank provider {primary_key} failed: {primary_exc}; "
                f"fallback bank provider {fallback_key} failed: {fallback_exc}"
            ) from fallback_exc


@cli.command()
def run(database_url: str | None = typer.Option(None, help="Override DATABASE_URL/settings.yaml")) -> None:
    configure_logging()
    log = structlog.get_logger()
    cfg = settings()
    session = make_session(database_url or cfg["database"]["url"])
    provider_kwargs = _common_provider_kwargs(cfg)
    results: list[ProviderResult] = []

    fred = FredCsvProvider(**provider_kwargs)
    for name, series_id in cfg["providers"]["fred"]["series"].items():
        results.append(
            _result(
                f"fred:{name}",
                lambda series_id=series_id, name=name: _write_market_data(
                    session, fred.source, name.upper(), fred.fetch_series(series_id)
                ),
            )
        )

    results.append(
        _result(
            "bank:usd_sell_with_fallback",
            lambda: _ingest_bank_with_fallback(session, provider_kwargs),
        )
    )

    cbc = TaiwanCentralBankProvider(**provider_kwargs)
    results.append(
        _result(
            "fx:cbc_usdtwd_close",
            lambda: upsert_rows(
                session,
                FxPrice,
                [
                    {
                        "observed_at_utc": row["date"],
                        "source": cbc.source,
                        "pair": "USD/TWD",
                        "open": None,
                        "high": None,
                        "low": None,
                        "close": row["close"],
                        "volume": None,
                    }
                    for row in cbc.fetch_usdtwd_closing().to_dict("records")
                ],
                ("pair", "observed_at_utc", "source"),
            ),
        )
    )

    twse = TwseProvider(**provider_kwargs)
    results.append(
        _result(
            "twse:foreign_flow",
            lambda: upsert_rows(
                session,
                ForeignFlow,
                [
                    {
                        "observed_at_utc": row["date"],
                        "source": twse.source,
                        "market": row["market"],
                        "foreign_net_buy_sell_shares": row["foreign_net_buy_sell_shares"],
                        "raw_payload": None,
                    }
                    for row in twse.fetch_foreign_flow().to_dict("records")
                ],
                ("market", "observed_at_utc", "source"),
            ),
        )
    )
    results.append(_result("twse:taiex", lambda: _write_market_data(session, twse.source, "TAIEX", twse.fetch_taiex_month())))
    results.append(
        _result(
            "twse:2330",
            lambda: _write_market_data(
                session, twse.source, "2330.TW", twse.fetch_stock_month(cfg["providers"]["twse"]["tsmc_stock_no"])
            ),
        )
    )

    if cfg["providers"]["yahoo"].get("enabled", True):
        yahoo = YahooFinanceProvider()
        years = int(cfg["providers"]["yahoo"].get("period_years", 10))
        for name, symbol in cfg["providers"]["yahoo"]["symbols"].items():
            if name == "market_usdtwd":
                results.append(
                    _result(
                        f"yahoo:{name}",
                        lambda symbol=symbol: upsert_rows(
                            session,
                            FxPrice,
                            [
                                {
                                    "observed_at_utc": row["date"],
                                    "source": yahoo.source,
                                    "pair": "USD/TWD",
                                    "open": row["open"],
                                    "high": row["high"],
                                    "low": row["low"],
                                    "close": row["close"],
                                    "volume": row["volume"],
                                }
                                for row in yahoo.fetch_ohlcv(symbol, years=years).to_dict("records")
                            ],
                            ("pair", "observed_at_utc", "source"),
                        ),
                    )
                )
            else:
                results.append(
                    _result(
                        f"yahoo:{name}",
                        lambda name=name, symbol=symbol: _write_market_data(
                            session, yahoo.source, name.upper(), yahoo.fetch_ohlcv(symbol, years=years)
                        ),
                    )
                )

    for item in results:
        if item.ok:
            log.info("ingestion_provider_ok", provider=item.name, rows=item.rows)
        else:
            log.warning("ingestion_provider_failed", provider=item.name, error=item.error)

    failed = [item for item in results if not item.ok]
    log.info("ingestion_finished", providers=len(results), failed=len(failed), rows=sum(item.rows for item in results if item.ok))
    if failed:
        raise typer.Exit(code=2)


if __name__ == "__main__":
    cli()
