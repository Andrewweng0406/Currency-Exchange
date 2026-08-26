from datetime import date

import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.database.schema import Base, ExchangePlan
from app.exchange.planner import ExchangeInputs, payment_deadline_risk, persist_exchange_plan, recommend_exchange
from app.risk.cbc import estimate_cbc_intervention_risk
from app.risk.scoring import detect_regime, opportunity_score
from app.risk.tail import estimate_tail_risk


def test_payment_deadline_risk():
    assert payment_deadline_risk(date(2026, 1, 2), today=date(2026, 1, 1)) == 100
    assert payment_deadline_risk(date(2026, 2, 15), today=date(2026, 1, 1)) == 20


def test_recommend_exchange_high_risk():
    rec = recommend_exchange(ExchangeInputs(target_usd_amount=10000, usd_already_held=3000), 82, 30, today=date(2026, 1, 1))
    assert rec.action == "EXCHANGE_75_PERCENT"
    assert rec.suggested_usd_to_exchange == 5250


def test_opportunity_score_high_when_usdtwd_low():
    frame = pd.DataFrame({"USDTWD_CLOSE": list(range(100, 352)) + [90]})
    assert opportunity_score(frame) > 90


def test_detect_regime():
    regimes = detect_regime(pd.Series({"DXY_RETURN_20D": 0.02, "USDTWD_VOLATILITY_20D": 0.005, "VIX_CHANGE_5D": 5}))
    assert "USD_STRONG" in regimes
    assert "HIGH_VOL" in regimes


def test_tail_risk_estimate():
    frame = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=30, tz="UTC"), "USDTWD_CLOSE": list(range(30, 60))})
    tail = estimate_tail_risk(frame, horizon_days=5)
    assert tail.probabilities["USD_TWD_UP_GT_1PCT"] > 0


def test_cbc_intervention_risk_is_estimated():
    risk = estimate_cbc_intervention_risk(pd.Series({"USDTWD_RETURN_5D": 0.02, "USDTWD_VOLATILITY_20D": 0.005, "CNH_RETURN_5D": 0.0}))
    assert risk.estimated
    assert risk.level in {"LOW", "MEDIUM", "HIGH"}


def test_persist_exchange_plan():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        inputs = ExchangeInputs(target_usd_amount=10000, usd_already_held=2500)
        rec = recommend_exchange(inputs, 80, 40)
        assert persist_exchange_plan(session, inputs, rec) == 1
        saved = session.execute(select(ExchangePlan)).scalar_one()
        assert saved.recommendation == rec.action
