from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import typer

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.backtesting.strategy_compare import compare_exchange_strategies, summarize_strategy_comparison
from app.config import settings
from app.database.session import make_session

cli = typer.Typer(help="Compare fixed-date, tranche, and model-timing exchange strategies.")


@cli.command()
def run(
    target_usd: float = typer.Option(10_000, help="Monthly USD need used for cost comparison."),
    start_year: int = typer.Option(2023, help="First year included in the comparison."),
) -> None:
    session = make_session(settings()["database"]["url"])
    result = compare_exchange_strategies(session, target_usd=target_usd, start_year=start_year)
    summary = summarize_strategy_comparison(result)
    payload = {
        "summary": asdict(summary) if summary else None,
        "strategies": [asdict(item) for item in result],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    cli()
