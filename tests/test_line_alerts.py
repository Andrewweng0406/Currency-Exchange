from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.alerts.engine import AlertCandidate, persist_alert, should_send_alert
from app.database.schema import Base
from app.exchange.planner import ExchangeInputs, ExchangeRecommendation
from app.line.client import verify_line_signature
from app.line.formatter import daily_report
from app.risk.cbc import CbcInterventionRisk
from app.risk.scoring import RiskSnapshot
from app.risk.tail import TailRiskSnapshot


def test_daily_report_contains_required_language():
    text = daily_report(
        current_usdtwd=31.42,
        bank_spot_selling=31.47,
        changes={"1d": 0.003, "5d": 0.008, "20d": -0.012},
        predictions={"1d": {"prob_up": 0.61, "prob_down": 0.39, "prediction_interval_80": {"lower": -0.01, "upper": 0.02}}},
        risk=RiskSnapshot(
            "2026-01-01T00:00:00Z",
            72,
            40,
            ["RISK_OFF"],
            0.74,
            [{"name": "dxy_momentum", "contribution": 7}],
            TailRiskSnapshot("5d", {"USD_TWD_UP_GT_1PCT": 0.32, "USD_TWD_UP_GT_2PCT": 0.05}, "test"),
            CbcInterventionRisk("LOW", True, ["test"]),
        ),
        exchange=ExchangeRecommendation("EXCHANGE_50_PERCENT", 50, 7000, 3500, 45, []),
        exchange_inputs=ExchangeInputs(target_usd_amount=10000, usd_already_held=3000),
    )
    assert "USD/TWD 留學生換匯監控" in text
    assert "預測報酬區間" in text
    assert "未來美元需求" in text
    assert "不代表匯率一定會上漲或下跌" in text


def test_alert_dedupe():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    candidate = AlertCandidate("TEST", "HIGH", "Title", "Body", "key")
    with Session(engine) as session:
        assert should_send_alert(session, candidate, datetime(2026, 1, 1, tzinfo=timezone.utc))
        persist_alert(session, candidate, datetime(2026, 1, 1, tzinfo=timezone.utc))
        assert not should_send_alert(session, candidate, datetime(2026, 1, 1, 0, 30, tzinfo=timezone.utc))


def test_line_signature_verification():
    import base64
    import hashlib
    import hmac

    body = b'{"events":[]}'
    secret = "secret"
    signature = base64.b64encode(hmac.new(secret.encode(), body, hashlib.sha256).digest()).decode()
    assert verify_line_signature(body, signature, secret)
    assert not verify_line_signature(body, "bad", secret)
