from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    observed_at_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    source: Mapped[str] = mapped_column(String(80), index=True)
    created_at_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)


class FxPrice(TimestampMixin, Base):
    __tablename__ = "fx_prices"
    pair: Mapped[str] = mapped_column(String(20), index=True)
    open: Mapped[float | None] = mapped_column(Float)
    high: Mapped[float | None] = mapped_column(Float)
    low: Mapped[float | None] = mapped_column(Float)
    close: Mapped[float | None] = mapped_column(Float)
    volume: Mapped[float | None] = mapped_column(Float)
    __table_args__ = (UniqueConstraint("pair", "observed_at_utc", "source", name="uq_fx_price"),)


class BankRate(TimestampMixin, Base):
    __tablename__ = "bank_rates"
    bank_name: Mapped[str] = mapped_column(String(120), index=True)
    currency: Mapped[str] = mapped_column(String(10), index=True)
    cash_buying: Mapped[float | None] = mapped_column(Float)
    cash_selling: Mapped[float | None] = mapped_column(Float)
    spot_buying: Mapped[float | None] = mapped_column(Float)
    spot_selling: Mapped[float | None] = mapped_column(Float)
    __table_args__ = (UniqueConstraint("bank_name", "currency", "observed_at_utc", "source", name="uq_bank_rate"),)


class ForeignFlow(TimestampMixin, Base):
    __tablename__ = "foreign_flows"
    market: Mapped[str] = mapped_column(String(40), index=True)
    foreign_net_buy_sell_shares: Mapped[float | None] = mapped_column(Float)
    raw_payload: Mapped[str | None] = mapped_column(Text)
    __table_args__ = (UniqueConstraint("market", "observed_at_utc", "source", name="uq_foreign_flow"),)


class MarketData(TimestampMixin, Base):
    __tablename__ = "market_data"
    symbol: Mapped[str] = mapped_column(String(40), index=True)
    open: Mapped[float | None] = mapped_column(Float)
    high: Mapped[float | None] = mapped_column(Float)
    low: Mapped[float | None] = mapped_column(Float)
    close: Mapped[float | None] = mapped_column(Float)
    volume: Mapped[float | None] = mapped_column(Float)
    __table_args__ = (UniqueConstraint("symbol", "observed_at_utc", "source", name="uq_market_data"),)


class MacroData(TimestampMixin, Base):
    __tablename__ = "macro_data"
    variable: Mapped[str] = mapped_column(String(80), index=True)
    value: Mapped[float | None] = mapped_column(Float)
    release_time_utc: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    raw_payload: Mapped[str | None] = mapped_column(Text)


class EconomicEvent(TimestampMixin, Base):
    __tablename__ = "economic_events"
    event_name: Mapped[str] = mapped_column(String(120), index=True)
    release_time_utc: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    previous: Mapped[float | None] = mapped_column(Float)
    forecast: Mapped[float | None] = mapped_column(Float)
    actual: Mapped[float | None] = mapped_column(Float)
    surprise: Mapped[float | None] = mapped_column(Float)
    surprise_zscore: Mapped[float | None] = mapped_column(Float)


class Feature(TimestampMixin, Base):
    __tablename__ = "features"
    feature_set: Mapped[str] = mapped_column(String(80), index=True)
    values_json: Mapped[str] = mapped_column(Text)


class Prediction(TimestampMixin, Base):
    __tablename__ = "predictions"
    model_version: Mapped[str] = mapped_column(String(80), index=True)
    horizon: Mapped[str] = mapped_column(String(20), index=True)
    prob_up: Mapped[float | None] = mapped_column(Float)
    prob_down: Mapped[float | None] = mapped_column(Float)
    expected_return: Mapped[float | None] = mapped_column(Float)
    confidence: Mapped[float | None] = mapped_column(Float)
    risk_score: Mapped[float | None] = mapped_column(Float)
    input_snapshot: Mapped[str | None] = mapped_column(Text)
    recommendation: Mapped[str | None] = mapped_column(String(80))


class ModelPerformance(TimestampMixin, Base):
    __tablename__ = "model_performance"
    model_version: Mapped[str] = mapped_column(String(80), index=True)
    horizon: Mapped[str] = mapped_column(String(20), index=True)
    metric: Mapped[str] = mapped_column(String(80), index=True)
    value: Mapped[float | None] = mapped_column(Float)
    window: Mapped[str | None] = mapped_column(String(20))


class Alert(TimestampMixin, Base):
    __tablename__ = "alerts"
    alert_type: Mapped[str] = mapped_column(String(80), index=True)
    message: Mapped[str] = mapped_column(Text)
    severity: Mapped[str | None] = mapped_column(String(40))
    dedupe_key: Mapped[str | None] = mapped_column(String(160), index=True)


class ExchangePlan(TimestampMixin, Base):
    __tablename__ = "exchange_plans"
    monthly_usd_need: Mapped[float | None] = mapped_column(Float)
    target_usd_amount: Mapped[float | None] = mapped_column(Float)
    usd_already_held: Mapped[float | None] = mapped_column(Float)
    twd_available: Mapped[float | None] = mapped_column(Float)
    next_payment_date: Mapped[str | None] = mapped_column(String(20))
    recommendation: Mapped[str | None] = mapped_column(String(80))
