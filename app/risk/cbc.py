from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CbcInterventionRisk:
    level: str
    estimated: bool
    reasons: list[str]


def estimate_cbc_intervention_risk(row: pd.Series) -> CbcInterventionRisk:
    score = 0
    reasons: list[str] = []
    usdtwd_5d = float(row.get("USDTWD_RETURN_5D") or 0)
    usdtwd_vol = float(row.get("USDTWD_VOLATILITY_20D") or 0)
    asia_returns = [row.get("CNH_RETURN_5D"), row.get("KRW_RETURN_5D"), row.get("JPY_RETURN_5D")]
    asia_clean = [float(v) for v in asia_returns if v is not None and not pd.isna(v)]
    asia_avg = sum(asia_clean) / len(asia_clean) if asia_clean else 0
    divergence = abs(usdtwd_5d - asia_avg)

    if abs(usdtwd_5d) > 0.01:
        score += 35
        reasons.append("USD/TWD 5日變動偏大")
    if usdtwd_vol > 0.004:
        score += 25
        reasons.append("USD/TWD 20日波動偏高")
    if divergence > 0.015:
        score += 30
        reasons.append("USD/TWD 與亞洲貨幣走勢出現偏離")
    if row.get("USDTWD_BOLLINGER_POSITION_20D") is not None and not pd.isna(row.get("USDTWD_BOLLINGER_POSITION_20D")):
        pos = float(row.get("USDTWD_BOLLINGER_POSITION_20D"))
        if pos > 0.95 or pos < 0.05:
            score += 10
            reasons.append("USD/TWD 接近近期統計區間邊緣")

    if score >= 60:
        level = "HIGH"
    elif score >= 30:
        level = "MEDIUM"
    else:
        level = "LOW"
    if not reasons:
        reasons.append("未觀察到明顯異常波動或亞洲貨幣偏離")
    return CbcInterventionRisk(level=level, estimated=True, reasons=reasons)
