from app.models.feature_importance import feature_category


def test_feature_category_labels_key_fx_drivers():
    assert feature_category("DXY_RETURN_5D") == "美元強弱"
    assert feature_category("US2Y_CHANGE_1D") == "美國利率"
    assert feature_category("TAIEX_RETURN_5D") == "台股與台積電"
    assert feature_category("FOREIGN_FLOW_5D") == "台灣外資資金流"
    assert feature_category("CNY_RETURN_5D") == "亞洲貨幣"
