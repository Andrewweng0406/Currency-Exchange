from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import typer

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.config import settings
from app.database.session import make_session
from app.models.training import HORIZON_DAYS, train_horizon

cli = typer.Typer(help="Train walk-forward validated horizon models.")


@cli.command()
def run(horizon: str = typer.Option("all", help="One of 1d, 5d, 20d, or all.")) -> None:
    session = make_session(settings()["database"]["url"])
    horizons = list(HORIZON_DAYS) if horizon == "all" else [horizon]
    results = []
    for item in horizons:
        if item not in HORIZON_DAYS:
            raise typer.BadParameter(f"Unknown horizon {item}")
        results.append(asdict(train_horizon(session, item)))
    print(json.dumps(results, indent=2, ensure_ascii=False, default=str, allow_nan=False))


if __name__ == "__main__":
    cli()
