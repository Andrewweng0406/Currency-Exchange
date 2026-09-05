from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any


@dataclass(frozen=True)
class FamilyReason:
    category: str
    text: str
    pressure: str
    importance: float


def family_reason_lines(features: dict[str, Any] | None, limit: int = 4) -> list[str]:
    reasons = family_reasons(features, limit=limit)
    if not reasons:
        return ["• 目前主要資料不足，先保守觀望"]
    return [f"• {reason.text}" for reason in reasons]


def family_reasons(features: dict[str, Any] | None, limit: int = 4) -> list[FamilyReason]:
    if not features:
        return []

    candidates = [
        _dxy_reason(features),
        _us2y_reason(features),
        _foreign_flow_reason(features),
        _taiwan_equity_reason(features),
        _asia_fx_reason(features),
        _risk_off_reason(features),
    ]
    clean = [item for item in candidates if item is not None]
    clean.sort(key=lambda item: item.importance, reverse=True)
    return clean[:limit]


def _dxy_reason(row: dict[str, Any]) -> FamilyReason | None:
    value = _num(row.get("DXY_RETURN_5D"))
    label = "美元指數 DXY"
    if value is None:
        value = _num(row.get("BROAD_USD_INDEX_RETURN_5D"))
        label = "美元指數"
    if value is None:
        return None
    if value > 0:
        text = f"{label} 近5天上漲 {_pct(value)}，代表美元整體偏強"
        pressure = "USD_TWD_UP"
    elif value < 0:
        text = f"{label} 近5天下跌 {_pct(value)}，代表美元整體偏弱"
        pressure = "USD_TWD_DOWN"
    else:
        text = f"{label} 近5天變化不大，美元強弱訊號中性"
        pressure = "NEUTRAL"
    return FamilyReason("美元指數", text, pressure, _score(value, 0.015))


def _us2y_reason(row: dict[str, Any]) -> FamilyReason | None:
    value = _num(row.get("US2Y_CHANGE_5D"))
    if value is None:
        return None
    bp = value * 100
    if value > 0:
        text = f"美國2年債利率近5天上升 {_bp(bp)}，通常會增加美元支撐"
        pressure = "USD_TWD_UP"
    elif value < 0:
        text = f"美國2年債利率近5天下降 {_bp(bp)}，通常會減少美元支撐"
        pressure = "USD_TWD_DOWN"
    else:
        text = "美國2年債利率近5天變化不大，利率訊號中性"
        pressure = "NEUTRAL"
    return FamilyReason("美國利率", text, pressure, _score(value, 0.15))


def _foreign_flow_reason(row: dict[str, Any]) -> FamilyReason | None:
    value = _num(row.get("FOREIGN_FLOW_5D"))
    zscore = _num(row.get("FOREIGN_FLOW_ZSCORE"))
    if value is None and zscore is None:
        return None
    reference = zscore if zscore is not None else value
    if reference is None:
        return None
    if reference > 0:
        text = "外資近5日偏買超台股，資金流入對台幣較有利"
        pressure = "USD_TWD_DOWN"
    elif reference < 0:
        text = "外資近5日偏賣超台股，資金流出可能讓台幣承壓"
        pressure = "USD_TWD_UP"
    else:
        text = "外資近5日買賣超不明顯，資金流訊號中性"
        pressure = "NEUTRAL"
    importance = min(100.0, abs(reference) * 35) if zscore is not None else 35.0
    return FamilyReason("外資資金流", text, pressure, importance)


def _taiwan_equity_reason(row: dict[str, Any]) -> FamilyReason | None:
    taiex = _num(row.get("TAIEX_RETURN_5D"))
    tsmc = _num(row.get("TSMC_RETURN_5D"))
    values = [value for value in [taiex, tsmc] if value is not None]
    if not values:
        return None
    average = sum(values) / len(values)
    if average > 0:
        text = f"台股/台積電近5天偏強 {_pct(average)}，通常有利外資留在台灣"
        pressure = "USD_TWD_DOWN"
    elif average < 0:
        text = f"台股/台積電近5天偏弱 {_pct(average)}，可能增加台幣壓力"
        pressure = "USD_TWD_UP"
    else:
        text = "台股/台積電近5天變化不大，台股訊號中性"
        pressure = "NEUTRAL"
    return FamilyReason("台股與台積電", text, pressure, _score(average, 0.04))


def _asia_fx_reason(row: dict[str, Any]) -> FamilyReason | None:
    china_fx = _num(row.get("CHINA_FX_PROXY_RETURN_5D"))
    if china_fx is None:
        china_fx = _num(row.get("CNH_RETURN_5D"))
    if china_fx is None:
        china_fx = _num(row.get("CNY_RETURN_5D"))
    values = [value for value in [china_fx, _num(row.get("KRW_RETURN_5D")), _num(row.get("JPY_RETURN_5D"))] if value is not None]
    if not values:
        return None
    average = sum(values) / len(values)
    if average > 0:
        text = f"人民幣/韓元/日圓對美元近5天偏弱 {_pct(average)}，亞洲貨幣承壓"
        pressure = "USD_TWD_UP"
    elif average < 0:
        text = f"人民幣/韓元/日圓對美元近5天偏強 {_pct(average)}，亞洲貨幣壓力較低"
        pressure = "USD_TWD_DOWN"
    else:
        text = "亞洲主要貨幣近5天變化不大，區域匯率訊號中性"
        pressure = "NEUTRAL"
    return FamilyReason("亞洲貨幣", text, pressure, _score(average, 0.02))


def _risk_off_reason(row: dict[str, Any]) -> FamilyReason | None:
    vix = _num(row.get("VIX_CHANGE_5D"))
    sp500 = _num(row.get("SP500_RETURN_5D"))
    nasdaq = _num(row.get("NASDAQ_RETURN_5D"))
    if vix is None and sp500 is None and nasdaq is None:
        return None
    equity_values = [value for value in [sp500, nasdaq] if value is not None]
    equity_average = sum(equity_values) / len(equity_values) if equity_values else 0.0
    pressure_value = max(0.0, vix or 0.0) / 5 + max(0.0, -equity_average) / 0.04
    calm_value = max(0.0, -(vix or 0.0)) / 5 + max(0.0, equity_average) / 0.04
    if pressure_value > calm_value and pressure_value > 0:
        text = "VIX或美股顯示市場避險情緒升高，資金可能偏向美元"
        pressure = "USD_TWD_UP"
        importance = min(100.0, pressure_value * 50)
    elif calm_value > 0:
        text = "VIX或美股顯示市場風險情緒穩定，美元避險需求較低"
        pressure = "USD_TWD_DOWN"
        importance = min(100.0, calm_value * 50)
    else:
        text = "VIX和美股變化不大，全球風險訊號中性"
        pressure = "NEUTRAL"
        importance = 10.0
    return FamilyReason("全球風險", text, pressure, importance)


def _num(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not isfinite(parsed):
        return None
    return parsed


def _score(value: float, scale: float) -> float:
    return min(100.0, abs(value) / scale * 100)


def _pct(value: float) -> str:
    return f"{value:+.2%}"


def _bp(value: float) -> str:
    return f"{value:+.0f}bp"
