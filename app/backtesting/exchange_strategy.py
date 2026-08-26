from __future__ import annotations

from dataclasses import dataclass

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
    available = rates[pd.to_datetime(rates["date"], utc=True) <= buy_date].tail(1)
    if available.empty:
        available = rates.head(1)
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
    start = payment_date - pd.Timedelta(days=days_before)
    eligible = rates[(pd.to_datetime(rates["date"], utc=True) >= start) & (pd.to_datetime(rates["date"], utc=True) <= payment_date)]
    if eligible.empty:
        eligible = rates.tail(min(tranches, len(rates)))
    if len(eligible) > tranches:
        positions = [round(i * (len(eligible) - 1) / (tranches - 1)) for i in range(tranches)]
        selected = eligible.iloc[positions]
    else:
        selected = eligible
    usd_each = target_usd / len(selected)
    orders = [_order_from_row(row, usd_each, rate_col) for _, row in selected.iterrows()]
    return _result("equal_tranches", target_usd, orders)


def _single_order(name: str, row: pd.Series, target_usd: float, rate_col: str) -> StrategyResult:
    return _result(name, target_usd, [_order_from_row(row, target_usd, rate_col)])


def _order_from_row(row: pd.Series, usd_amount: float, rate_col: str) -> ExchangeOrder:
    rate = float(row[rate_col])
    return ExchangeOrder(
        date=pd.Timestamp(row["date"]),
        usd_amount=float(usd_amount),
        rate=rate,
        twd_cost=float(usd_amount * rate),
    )


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
