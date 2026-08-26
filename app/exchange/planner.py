from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any

from app.config import risk_policy
from app.database.schema import ExchangePlan
from app.database.upsert import upsert_rows


@dataclass(frozen=True)
class ExchangeInputs:
    target_usd_amount: float
    usd_already_held: float
    twd_available: float | None = None
    next_payment_date: date | None = None
    monthly_usd_need: float | None = None


@dataclass(frozen=True)
class ExchangeRecommendation:
    action: str
    exchange_percent: int
    usd_shortfall: float
    suggested_usd_to_exchange: float
    deadline_risk: int
    rationale: list[str]


def default_inputs() -> ExchangeInputs:
    profile = risk_policy()["exchange_recommendation"]["default_profile"]
    payment_date = profile.get("next_payment_date")
    return ExchangeInputs(
        target_usd_amount=float(profile["target_usd_amount"]),
        usd_already_held=float(profile["usd_already_held"]),
        twd_available=profile.get("twd_available"),
        next_payment_date=date.fromisoformat(payment_date) if payment_date else None,
        monthly_usd_need=profile.get("monthly_usd_need"),
    )


def recommend_exchange(inputs: ExchangeInputs, risk_score: int, opportunity_score: int, today: date | None = None) -> ExchangeRecommendation:
    policy = risk_policy()["exchange_recommendation"]
    usd_shortfall = max(0.0, inputs.target_usd_amount - inputs.usd_already_held)
    deadline_risk = payment_deadline_risk(inputs.next_payment_date, today=today, policy=policy)
    action = "WAIT"
    for rule in policy["rules"]:
        if _rule_matches(rule, risk_score, opportunity_score, deadline_risk):
            action = rule["recommendation"]
            break
    percent = _percent_from_action(action)
    suggested = usd_shortfall * percent / 100
    rationale = [
        f"TWD Risk Score={risk_score}",
        f"Opportunity Score={opportunity_score}",
        f"Payment Deadline Risk={deadline_risk}",
    ]
    if deadline_risk >= 90:
        rationale.append("付款期限非常近，優先確保美元需求")
    elif risk_score >= 65:
        rationale.append("模型與市場特徵偏向台幣貶值風險")
    elif opportunity_score >= 80:
        rationale.append("目前美元價格位於相對有利區間")
    else:
        rationale.append("風險與機會訊號尚未強烈，維持分批彈性")
    return ExchangeRecommendation(
        action=action,
        exchange_percent=percent,
        usd_shortfall=usd_shortfall,
        suggested_usd_to_exchange=suggested,
        deadline_risk=deadline_risk,
        rationale=rationale,
    )


def persist_exchange_plan(session, inputs: ExchangeInputs, recommendation: ExchangeRecommendation) -> int:
    now = datetime.now(timezone.utc)
    return upsert_rows(
        session,
        ExchangePlan,
        [
            {
                "observed_at_utc": now,
                "source": "exchange_planner",
                "monthly_usd_need": inputs.monthly_usd_need,
                "target_usd_amount": inputs.target_usd_amount,
                "usd_already_held": inputs.usd_already_held,
                "twd_available": inputs.twd_available,
                "next_payment_date": inputs.next_payment_date.isoformat() if inputs.next_payment_date else None,
                "recommendation": recommendation.action,
            }
        ],
        ("observed_at_utc", "source"),
    )


def payment_deadline_risk(payment_date: date | None, today: date | None = None, policy: dict[str, Any] | None = None) -> int:
    if payment_date is None:
        return 35
    today = today or datetime.now(timezone.utc).date()
    policy = policy or risk_policy()["exchange_recommendation"]
    days = (payment_date - today).days
    thresholds = policy["deadline_days"]
    if days <= thresholds["critical"]:
        return 100
    if days <= thresholds["urgent"]:
        return 85
    if days <= thresholds["elevated"]:
        return 65
    if days <= thresholds["flexible"]:
        return 45
    return 20


def _rule_matches(rule: dict[str, Any], risk_score: int, opportunity_score: int, deadline_risk: int) -> bool:
    checks = {
        "min_deadline_risk": deadline_risk,
        "min_risk_score": risk_score,
        "max_risk_score": risk_score,
        "min_opportunity_score": opportunity_score,
    }
    for key, value in rule.items():
        if key == "recommendation":
            continue
        if key.startswith("min_") and checks[key] < value:
            return False
        if key.startswith("max_") and checks[key] > value:
            return False
    return True


def _percent_from_action(action: str) -> int:
    return {
        "WAIT": 0,
        "EXCHANGE_25_PERCENT": 25,
        "EXCHANGE_50_PERCENT": 50,
        "EXCHANGE_75_PERCENT": 75,
        "EXCHANGE_100_PERCENT": 100,
    }[action]
