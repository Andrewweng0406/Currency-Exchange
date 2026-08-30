from __future__ import annotations

from app.exchange.planner import ExchangeInputs, ExchangeRecommendation
from app.ops.data_quality import DataQualitySummary
from app.risk.scoring import RiskSnapshot


def daily_report(
    current_usdtwd: float,
    bank_spot_selling: float | None,
    changes: dict[str, float | None],
    predictions: dict[str, dict[str, float | None]],
    risk: RiskSnapshot,
    exchange: ExchangeRecommendation,
    exchange_inputs: ExchangeInputs | None = None,
    upcoming_events: list[dict] | None = None,
    ai_interpretation: object | None = None,
    data_quality: DataQualitySummary | None = None,
) -> str:
    _ = ai_interpretation
    pred_lines = [
        _simple_prediction_line("1天", predictions.get("1d", {})),
        _simple_prediction_line("5天", predictions.get("5d", {})),
        _simple_prediction_line("20天", predictions.get("20d", {})),
    ]
    reason_lines = _simple_reason_lines(risk.contributors)
    timing_action = _simple_timing_action(exchange.action, data_quality)
    timing_reason = _simple_timing_reason(exchange.action, predictions, risk, changes, reason_lines, data_quality)
    event_block = _simple_event_block(upcoming_events or [])
    _ = exchange_inputs
    return (
        "🇹🇼 美元換匯提醒\n\n"
        f"目前匯率：{current_usdtwd:.3f}\n"
        f"銀行美元賣出價：{_num(bank_spot_selling)}\n"
        f"近5天變化：{_signed_pct(changes.get('5d'))}\n\n"
        "━━━━━━━━━━\n\n"
        "未來方向\n"
        + "\n".join(pred_lines)
        + "\n\n━━━━━━━━━━\n\n"
        f"台幣風險：{risk.twd_risk_score}/100（{_simple_risk_label(risk.twd_risk_score)}）\n"
        f"信心：{_pct(risk.confidence)}\n\n"
        f"{_data_quality_line(data_quality)}\n\n"
        "━━━━━━━━━━\n\n"
        "主要判斷依據\n"
        + "\n".join(reason_lines)
        + "\n\n"
        "━━━━━━━━━━\n\n"
        "換匯建議\n"
        f"{timing_action}\n"
        f"原因：{timing_reason}\n\n"
        f"{event_block}"
        "這是機率與風險提醒，不是保證漲跌。"
    )


def alert_message(alert_type: str, title: str, body: str, risk_score: int | None = None) -> str:
    risk = f"\n\nRisk Score：\n{risk_score}" if risk_score is not None else ""
    return f"{title}\n\n{body}{risk}\n\n請依近期是否需要美元，重新評估換匯時間。"


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


def _interval(value: dict) -> str:
    lower = value.get("lower")
    upper = value.get("upper")
    if lower is None or upper is None:
        return "N/A"
    return f"{lower:+.2%} ~ {upper:+.2%}"


def _simple_prediction_line(label: str, item: dict[str, float | None]) -> str:
    prob_up = item.get("prob_up")
    prob_down = item.get("prob_down")
    if prob_up is None or prob_down is None:
        return f"{label}：資料不足"
    direction = "美元偏漲" if prob_up > prob_down else "美元偏跌" if prob_down > prob_up else "方向中性"
    return f"{label}：{direction}（漲{_pct(prob_up)} / 跌{_pct(prob_down)}）"


def _simple_reason_lines(contributors: list[dict]) -> list[str]:
    if not contributors:
        return ["• 目前資料不足，先保守觀望"]

    lines = []
    for item in contributors[:5]:
        line = _simple_reason_line(item)
        if line:
            lines.append(line)
    return lines or ["• 目前資料不足，先保守觀望"]


def _simple_reason_line(item: dict) -> str:
    name = str(item.get("name", "")).lower()
    value = _as_float(item.get("value"))

    if name in {"prediction_5d", "prediction 5d"}:
        return f"• 模型預測：{_pressure_text(value, '未來5天美元上漲壓力偏低', '未來5天美元上漲壓力偏高')}"
    if name in {"prediction_20d", "prediction 20d"}:
        return f"• 中期預測：{_pressure_text(value, '未來20天美元上漲壓力偏低', '未來20天美元上漲壓力偏高')}"
    if "usdtwd" in name and "momentum" in name:
        return f"• 近期匯率走勢：{_pressure_text(value, '美元最近偏弱', '美元最近偏強')}"
    if "dxy" in name:
        return f"• 美元指數 DXY：{_pressure_text(value, '美元指數偏弱', '美元指數偏強')}"
    if "rate" in name or "us2y" in name or "yield" in name:
        return f"• 美國利率：{_pressure_text(value, '利率壓力偏低', '利率壓力偏高')}"
    if "asia" in name or "cnh" in name or "krw" in name or "jpy" in name:
        return f"• 亞洲貨幣：{_pressure_text(value, '亞洲貨幣相對穩定', '亞洲貨幣對美元偏弱')}"
    if "foreign" in name or "flow" in name:
        return f"• 外資資金流：{_pressure_text(value, '外資壓力偏低', '外資賣壓偏高')}"
    if "risk" in name or "vix" in name:
        return f"• 全球市場風險：{_pressure_text(value, '市場風險偏低', '市場風險偏高')}"
    return f"• {name.replace('_', ' ').upper()}：{_pressure_text(value, '偏有利', '偏不利')}"


def _pressure_text(value: float | None, low_text: str, high_text: str) -> str:
    if value is None:
        return "資料有限"
    if value >= 60:
        return high_text
    if value <= 40:
        return low_text
    return "大致中性"


def _simple_risk_label(score: int) -> str:
    if score <= 40:
        return "目前壓力偏低"
    if score <= 60:
        return "中性觀望"
    if score <= 80:
        return "台幣有貶值風險"
    return "台幣貶值風險偏高"


def _data_quality_line(data_quality: DataQualitySummary | None) -> str:
    if data_quality is None:
        return "資料狀態：未檢查"
    return f"資料狀態：{data_quality.label_zh}"


def _simple_timing_action(action: str, data_quality: DataQualitySummary | None = None) -> str:
    if data_quality and data_quality.blocks_model_advice:
        return "今天資料不完整，先不要只看模型決定。"
    labels = {
        "WAIT": "可以再等一下，現在不用急著換。",
        "EXCHANGE_25_PERCENT": "可以開始留意，現在不是太差的換匯時間。",
        "EXCHANGE_50_PERCENT": "現在偏向適合換，近期需要美元的話可以認真考慮。",
        "EXCHANGE_75_PERCENT": "現在換匯風險偏高，近期需要美元的話建議優先處理。",
        "EXCHANGE_100_PERCENT": "時間或風險已經偏緊，近期需要美元的話建議盡快處理。",
    }
    return labels.get(action, action.replace("_", " "))


def _simple_timing_reason(
    action: str,
    predictions: dict[str, dict[str, float | None]],
    risk: RiskSnapshot,
    changes: dict[str, float | None],
    reason_lines: list[str],
    data_quality: DataQualitySummary | None = None,
) -> str:
    if data_quality and data_quality.blocks_model_advice:
        issue = data_quality.issues[0] if data_quality.issues else "核心資料異常"
        return f"{data_quality.message_zh}主要問題：{issue}。"

    five_day = predictions.get("5d", {})
    prob_up = five_day.get("prob_up")
    prob_down = five_day.get("prob_down")
    change_5d = changes.get("5d")
    main_reason = _clean_reason(reason_lines[0]) if reason_lines else "目前資料有限"

    if action == "WAIT":
        if prob_down is not None and prob_down >= 0.55:
            return f"模型偏向美元短期走弱，5天美元下跌機率約{_pct(prob_down)}；{main_reason}。"
        if risk.twd_risk_score <= 40:
            return f"台幣風險分數偏低，目前沒有明顯美元急漲壓力；{main_reason}。"
        return f"目前訊號不夠強，適合先觀望；{main_reason}。"

    if action in {"EXCHANGE_25_PERCENT", "EXCHANGE_50_PERCENT"}:
        if prob_up is not None and prob_up >= 0.55:
            return f"模型偏向美元短期走強，5天美元上漲機率約{_pct(prob_up)}；{main_reason}。"
        if change_5d is not None and change_5d > 0:
            return f"近期美元已有上漲跡象，等太久可能遇到更貴匯率；{main_reason}。"
        return f"台幣風險開始升高，近期要用美元時不建議完全等待；{main_reason}。"

    if risk.twd_risk_score >= 80:
        return f"台幣風險分數偏高，代表美元上漲或台幣轉弱壓力較明顯；{main_reason}。"
    return f"付款時間或市場風險偏緊，近期要用美元時應優先處理；{main_reason}。"


def _clean_reason(line: str) -> str:
    return line.removeprefix("• ").rstrip("。")


def _simple_event_block(events: list[dict]) -> str:
    if not events:
        return ""
    event = events[0]
    return f"近期重要事件：{event.get('event_name', '重要經濟數據')}\n\n"


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _event_block(events: list[dict]) -> str:
    if not events:
        return ""
    lines = ["━━━━━━━━━━", "", "📅 Upcoming Risk", ""]
    for event in events[:3]:
        lines.append(str(event.get("event_name", "Unknown Event")))
        lines.append(f"發布：{event.get('release_time_utc', 'N/A')}")
        lines.append("⚠️ High Impact")
        lines.append("")
    return "\n".join(lines) + "\n"


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


def _model_explanation_block(items: list) -> str:
    if not items:
        return ""
    lines = ["模型特徵貢獻 Top 5（非因果）："]
    for item in items[:5]:
        feature = getattr(item, "feature", "")
        direction = getattr(item, "direction", "")
        magnitude = getattr(item, "magnitude", 0)
        lines.append(f"- {feature}: {direction}, {magnitude:.2f}")
    return "\n".join(lines) + "\n\n"


def _ai_block(item: object | None) -> str:
    if item is None or getattr(item, "error", None):
        return ""
    summary = getattr(item, "summary_zh_tw", None)
    if not summary:
        return ""
    sentiment = getattr(item, "macro_sentiment", None) or "N/A"
    risk_off = getattr(item, "risk_off_level", None) or "N/A"
    adjustment = getattr(item, "confidence_adjustment", 0)
    return (
        "AI 風險解讀（非預測模型）：\n"
        f"{summary}\n"
        f"Macro: {sentiment} / Risk-off: {risk_off} / Confidence adj: {adjustment:+.0f}\n\n"
    )
