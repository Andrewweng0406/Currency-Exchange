from __future__ import annotations

import sys
from pathlib import Path

import typer

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.config import settings
from app.database.session import make_session
from app.economic_events.importer import import_events_csv

cli = typer.Typer(help="Import economic events from an official/manual CSV. No fabricated consensus data.")


@cli.command()
def run(path: str) -> None:
    session = make_session(settings()["database"]["url"])
    print(import_events_csv(session, path))


if __name__ == "__main__":
    cli()
