from app.ai.openai_interpreter import _parse_json, _result_from_json, interpret_risk_context
from app.exchange.planner import ExchangeRecommendation
from app.line.formatter import daily_report
from app.risk.cbc import CbcInterventionRisk
from app.risk.scoring import RiskSnapshot
from app.risk.tail import TailRiskSnapshot


def test_openai_interpreter_disabled_without_config(monkeypatch):
    monkeypatch.delenv("OPENAI_ENABLED", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = interpret_risk_context(risk=None, predictions={}, current={})
    assert not result.enabled


def test_openai_json_parsing_and_adjustment_clamp():
    data = _parse_json('```json\n{"macro_sentiment":"mixed","confidence_adjustment":-99,"summary_zh_tw":"偏向觀望"}\n```')
    result = _result_from_json("test-model", data, "{}")
    assert result.confidence_adjustment == -10
    assert result.macro_sentiment == "MIXED"


def test_daily_report_keeps_ai_details_out_of_family_summary():
    risk = RiskSnapshot(
        "2026-01-01T00:00:00Z",
        50,
        55,
        ["RISK_ON"],
        0.7,
        [],
        TailRiskSnapshot("5d", {"USD_TWD_UP_GT_1PCT": 0.1, "USD_TWD_UP_GT_2PCT": 0.02}, "test"),
        CbcInterventionRisk("LOW", True, ["test"]),
        {"5d": []},
    )
    ai = _result_from_json(
        "test-model",
        {
            "macro_sentiment": "MIXED",
            "risk_off_level": "LOW",
            "confidence_adjustment": -2,
            "summary_zh_tw": "目前訊號混合，較適合維持分批彈性。",
        },
        "{}",
    )
    text = daily_report(
        current_usdtwd=31.4,
        bank_spot_selling=31.5,
        changes={"1d": 0.0, "5d": 0.0, "20d": 0.0},
        predictions={},
        risk=risk,
        exchange=ExchangeRecommendation("WAIT", 0, 10000, 0, 35, []),
        ai_interpretation=ai,
    )
    assert "美元換匯提醒" in text
    assert "AI 風險解讀" not in text
    assert "非預測模型" not in text
