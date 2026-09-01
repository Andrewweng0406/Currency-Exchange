from app.line.family_reasons import family_reason_lines, family_reasons


def test_family_reasons_rank_human_readable_market_inputs():
    reasons = family_reasons(
        {
            "DXY_RETURN_5D": 0.018,
            "US2Y_CHANGE_5D": 0.12,
            "FOREIGN_FLOW_ZSCORE": -1.4,
            "TAIEX_RETURN_5D": -0.02,
            "TSMC_RETURN_5D": -0.03,
            "CNH_RETURN_5D": 0.01,
            "KRW_RETURN_5D": 0.02,
            "JPY_RETURN_5D": 0.01,
            "VIX_CHANGE_5D": 2.0,
            "SP500_RETURN_5D": -0.01,
        }
    )
    assert len(reasons) == 4
    text = "\n".join(item.text for item in reasons)
    assert "DXY" in text
    assert "美國2年債" in text
    assert any(item.category in {"外資資金流", "亞洲貨幣", "台股與台積電", "全球風險"} for item in reasons)
    assert all("PREDICTION" not in item.text for item in reasons)


def test_family_reason_lines_fallback_when_missing():
    assert family_reason_lines({}) == ["• 目前主要資料不足，先保守觀望"]
