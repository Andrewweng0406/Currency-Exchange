from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd


@dataclass(frozen=True)
class ExchangeOrder:
    date: pd.Timestamp
    usd_amount: float
    rate: float
    twd_cost: float


@dataclass(frozen=True)
class StrategyResult:
    name: str
    target_usd: float
    total_twd_cost: float
    average_rate: float
    worst_rate: float
    orders: list[ExchangeOrder]


def fixed_day_exchange(
    rates: pd.DataFrame,
    target_usd: float,
    payment_date: pd.Timestamp,
    days_before: int = 30,
    rate_col: str = "USDTWD_CLOSE",
) -> StrategyResult:
    buy_date = payment_date - pd.Timedelta(days=days_before)
    _validate_request(target_usd)
    rates = rates.sort_values("date")
    dates = pd.to_datetime(rates["date"], utc=True)
    available = rates[(dates >= buy_date) & (dates <= payment_date)].head(1)
    if available.empty:
        raise ValueError("No executable quote between scheduled buy date and payment deadline")
    row = available.iloc[0]
    return _single_order("fixed_day_once", row, target_usd, rate_col)


def equal_tranche_exchange(
    rates: pd.DataFrame,
    target_usd: float,
    payment_date: pd.Timestamp,
    days_before: int = 30,
    tranches: int = 4,
    rate_col: str = "USDTWD_CLOSE",
) -> StrategyResult:
    _validate_request(target_usd)
    if tranches < 1:
        raise ValueError("tranches must be positive")
    start = payment_date - pd.Timedelta(days=days_before)
    rates = rates.sort_values("date")
    dates = pd.to_datetime(rates["date"], utc=True)
    schedules = pd.date_range(start, payment_date, periods=tranches)
    orders = []
    for scheduled in schedules:
        eligible = rates[(dates >= scheduled) & (dates <= payment_date)]
        if eligible.empty:
            raise ValueError("No executable quote for scheduled tranche before deadline")
        orders.append(_order_from_row(eligible.iloc[0], target_usd / tranches, rate_col))
    return _result("equal_tranches", target_usd, orders)


def _single_order(name: str, row: pd.Series, target_usd: float, rate_col: str) -> StrategyResult:
    return _result(name, target_usd, [_order_from_row(row, target_usd, rate_col)])


def _order_from_row(row: pd.Series, usd_amount: float, rate_col: str) -> ExchangeOrder:
    rate = float(row[rate_col])
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("Exchange rate must be finite and positive")
    return ExchangeOrder(
        date=pd.Timestamp(row["date"]),
        usd_amount=float(usd_amount),
        rate=rate,
        twd_cost=float(usd_amount * rate),
    )


def _validate_request(target_usd):
    if not math.isfinite(target_usd) or target_usd <= 0:
        raise ValueError("target_usd must be finite and positive")


def _result(name: str, target_usd: float, orders: list[ExchangeOrder]) -> StrategyResult:
    total = sum(order.twd_cost for order in orders)
    return StrategyResult(
        name=name,
        target_usd=target_usd,
        total_twd_cost=total,
        average_rate=total / target_usd,
        worst_rate=max(order.rate for order in orders),
        orders=orders,
    )
