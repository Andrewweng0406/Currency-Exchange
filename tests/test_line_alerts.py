from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.alerts.engine import AlertCandidate, generate_alerts, persist_alert, should_send_alert
from app.database.schema import Base
from app.exchange.planner import ExchangeInputs, ExchangeRecommendation
from app.line.client import verify_line_signature
from app.line.formatter import daily_report
from app.ops.data_quality import DataQualitySummary
from app.risk.cbc import CbcInterventionRisk
from app.risk.scoring import RiskSnapshot
from app.risk.tail import TailRiskSnapshot


def test_daily_report_contains_required_language():
    text = daily_report(
        current_usdtwd=31.42,
        bank_spot_selling=31.47,
        changes={"1d": 0.003, "5d": 0.008, "20d": -0.012},
        predictions={
            "1d": {"prob_up": 0.61, "prob_down": 0.39, "prediction_interval_80": {"lower": -0.01, "upper": 0.02}},
            "5d": {"prob_up": 0.68, "prob_down": 0.32},
        },
        risk=RiskSnapshot(
            "2026-01-01T00:00:00Z",
            72,
            40,
            ["RISK_OFF"],
            0.74,
            [{"name": "dxy_momentum", "contribution": 7}],
            TailRiskSnapshot("5d", {"USD_TWD_UP_GT_1PCT": 0.32, "USD_TWD_UP_GT_2PCT": 0.05}, "test"),
            CbcInterventionRisk("LOW", True, ["test"]),
            {"5d": []},
        ),
        exchange=ExchangeRecommendation("EXCHANGE_50_PERCENT", 50, 7000, 3500, 45, []),
        exchange_inputs=ExchangeInputs(target_usd_amount=10000, usd_already_held=3000),
        data_quality=DataQualitySummary("OK", "正常", "今天核心資料正常。", False, []),
    )
    assert "美元換匯提醒" in text
    assert "主要判斷依據" in text
    assert "美元指數 DXY" in text
    assert "現在偏向適合換" in text
    assert "原因：" in text
    assert "資料狀態：正常" in text
    assert "5天美元上漲機率約68%" in text
    assert "這是機率與風險提醒，不是保證漲跌" in text
    assert "美元需求" not in text
    assert "已持有" not in text
    assert "尚缺" not in text
    assert "建議先換：約" not in text
    assert "預測報酬區間" not in text
    assert "模型特徵貢獻" not in text


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


def test_generate_sudden_fx_and_macro_alerts():
    risk = RiskSnapshot(
        "2026-01-01T00:00:00Z",
        70,
        20,
        ["RISK_OFF"],
        0.6,
        [],
        TailRiskSnapshot("5d", {"USD_TWD_UP_GT_1PCT": 0.1, "USD_TWD_UP_GT_2PCT": 0.02}, "test"),
        CbcInterventionRisk("MEDIUM", True, ["test"]),
        {"5d": []},
    )
    alerts = generate_alerts(
        risk,
        {"5d": {"prob_up": 0.5}},
        {"USDTWD_RETURN_1D": 0.02, "USDTWD_VOLATILITY_20D": 0.004},
        [{"event_name": "CPI", "release_time_utc": "2026-01-01T13:30:00+00:00"}],
    )
    types = {alert.alert_type for alert in alerts}
    assert "SUDDEN_FX_MOVE" in types
    assert "MACRO_EVENT" in types


def test_data_quality_warning_alert():
    risk = RiskSnapshot(
        "2026-01-01T00:00:00Z",
        50,
        20,
        ["RISK_OFF"],
        0.6,
        [],
        TailRiskSnapshot("5d", {}, "test"),
        CbcInterventionRisk("LOW", True, ["test"]),
        {"5d": []},
    )
    alerts = generate_alerts(
        risk,
        {},
        {},
        [],
        DataQualitySummary("BLOCKING", "核心資料異常", "今天核心資料不完整，換匯時間建議會改為保守。", True, ["USDTWD_CLOSE 最新值缺失"]),
    )
    assert any(alert.alert_type == "DATA_QUALITY_WARNING" for alert in alerts)
