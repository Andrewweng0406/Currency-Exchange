from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import typer
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.config import ROOT

cli = typer.Typer(help="Configure student exchange planning profile.")


@cli.command()
def run(
    target_usd_amount: float = typer.Option(..., min=0),
    usd_already_held: float = typer.Option(..., min=0),
    next_payment_date: date | None = typer.Option(None),
    monthly_usd_need: float | None = typer.Option(None, min=0),
    twd_available: float | None = typer.Option(None, min=0),
) -> None:
    path = ROOT / "config" / "risk_policy.yaml"
    policy = yaml.safe_load(path.read_text(encoding="utf-8"))
    profile = policy["exchange_recommendation"]["default_profile"]
    profile["target_usd_amount"] = float(target_usd_amount)
    profile["usd_already_held"] = float(usd_already_held)
    profile["next_payment_date"] = next_payment_date.isoformat() if next_payment_date else None
    profile["monthly_usd_need"] = float(monthly_usd_need) if monthly_usd_need is not None else None
    profile["twd_available"] = float(twd_available) if twd_available is not None else None
    path.write_text(yaml.safe_dump(policy, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"Updated {path}")


if __name__ == "__main__":
    cli()
