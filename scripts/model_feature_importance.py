from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import typer

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.models.feature_importance import horizon_feature_importance
from app.models.training import HORIZON_DAYS

cli = typer.Typer(help="Report model feature importance from saved artifacts.")


@cli.command()
def run(
    horizon: str = typer.Option("all", help="One of 1d, 5d, 20d, or all."),
    top_n: int = typer.Option(12, help="Number of features per horizon."),
) -> None:
    horizons = list(HORIZON_DAYS) if horizon == "all" else [horizon]
    output = {}
    for item in horizons:
        if item not in HORIZON_DAYS:
            raise typer.BadParameter(f"Unknown horizon {item}")
        output[item] = [asdict(row) for row in horizon_feature_importance(item, top_n=top_n)]
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    cli()
