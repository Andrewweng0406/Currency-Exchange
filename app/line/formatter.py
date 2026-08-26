from __future__ import annotations

from dataclasses import asdict

from app.exchange.planner import ExchangeRecommendation
from app.risk.scoring import RiskSnapshot


def daily_report(
    current_usdtwd: float,
    bank_spot_selling: float | None,
    changes: dict[str, float | None],
    predictions: dict[str, dict[str, float | None]],
    risk: RiskSnapshot,
    exchange: ExchangeRecommendation,
) -> str:
    pred_lines = []
    for horizon in ["1d", "5d", "20d"]:
        item = predictions.get(horizon, {})
        up = _pct(item.get("prob_up"))
        down = _pct(item.get("prob_down"))
        pred_lines.append(f"未來 {horizon.upper()}\n美元上漲：{up}\n台幣升值：{down}")
    contributors = "\n".join(_contributor_line(item) for item in risk.contributors)
    return (
        "🇹🇼 USD/TWD 留學生換匯監控\n\n"
        f"目前市場匯率：\n{current_usdtwd:.3f}\n\n"
        f"銀行美元即期賣出：\n{_num(bank_spot_selling)}\n\n"
        f"1D: {_signed_pct(changes.get('1d'))}\n"
        f"5D: {_signed_pct(changes.get('5d'))}\n"
        f"20D: {_signed_pct(changes.get('20d'))}\n"
        "━━━━━━━━━━\n\n"
        "📊 預測\n\n"
        + "\n\n".join(pred_lines)
        + "\n\n━━━━━━━━━━\n\n"
        "⚠️ 台幣風險\n\n"
        f"TWD Risk Score：\n{risk.twd_risk_score} / 100\n\n"
        f"{_risk_label(risk.twd_risk_score)}\n\n"
        f"Confidence：\n{_pct(risk.confidence)}\n\n"
        f"CBC Intervention Risk：\n{risk.cbc_intervention_risk.level} (ESTIMATED)\n\n"
        "━━━━━━━━━━\n\n"
        "📉 Tail Risk\n\n"
        f"5日 USD/TWD > +1%：{_pct(risk.tail_risk.probabilities.get('USD_TWD_UP_GT_1PCT'))}\n"
        f"5日 USD/TWD > +2%：{_pct(risk.tail_risk.probabilities.get('USD_TWD_UP_GT_2PCT'))}\n\n"
        "━━━━━━━━━━\n\n"
        "🔎 主要原因\n\n"
        f"{contributors}\n\n"
        "━━━━━━━━━━\n\n"
        "💵 換匯建議\n\n"
        f"尚缺：\n${exchange.usd_shortfall:,.0f}\n\n"
        "Recommendation：\n"
        f"{exchange.action.replace('_', ' ')}\n\n"
        f"建議目前先換：\n約 ${exchange.suggested_usd_to_exchange:,.0f}\n\n"
        "此為機率與風險管理建議，不代表匯率一定會上漲或下跌。"
    )


def alert_message(alert_type: str, title: str, body: str, risk_score: int | None = None) -> str:
    risk = f"\n\nRisk Score：\n{risk_score}" if risk_score is not None else ""
    return f"{title}\n\n{body}{risk}\n\n請依近期美元需求重新評估換匯比例。"


def _pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.0f}%"


def _signed_pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:+.2%}"


def _num(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.3f}"


def _risk_label(score: int) -> str:
    if score <= 20:
        return "🟢 台幣偏強 / 美元相對便宜"
    if score <= 40:
        return "🟢 中度有利"
    if score <= 60:
        return "🟡 中性"
    if score <= 80:
        return "🟠 台幣偏弱風險"
    return "🔴 高度台幣貶值風險"


def _contributor_line(item: dict) -> str:
    name = str(item.get("name", "")).replace("_", " ").upper()
    contribution = float(item.get("contribution", 0))
    marker = "🔴" if contribution >= 6 else "🟡" if contribution >= 3 else "🟢"
    return f"{marker} {name}: {contribution:.1f}"
