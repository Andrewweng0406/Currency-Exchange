from __future__ import annotations

import pandas as pd
from sqlalchemy import select

from app.config import settings
from app.database.schema import BankRate


def latest_bank_quote(session, now=None):
    """Use a fresh configured family-bank quote before a fallback bank quote."""
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    primary_name = settings()["providers"]["bank"]["name"]
    rows = session.execute(select(BankRate).where(
        BankRate.currency == "USD", BankRate.spot_selling > 0,
    ).order_by(BankRate.observed_at_utc.desc())).scalars().all()
    rows = [row for row in rows if pd.to_datetime(row.observed_at_utc, utc=True) <= now]
    fresh = [row for row in rows if now - pd.to_datetime(row.observed_at_utc, utc=True) <= pd.Timedelta(days=7)]
    return next((row for row in fresh if row.bank_name == primary_name), fresh[0] if fresh else (rows[0] if rows else None))
