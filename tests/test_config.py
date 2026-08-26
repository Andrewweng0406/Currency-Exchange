from app.config import risk_policy, settings


def test_config_loads():
    assert settings()["app"]["name"] == "TWD FX Monitor"
    assert "alerts" in risk_policy()
