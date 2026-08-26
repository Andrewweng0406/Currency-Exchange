from datetime import date

import pandas as pd

from app.exchange.planner import ExchangeInputs, payment_deadline_risk, recommend_exchange
from app.risk.scoring import detect_regime, opportunity_score


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
