from datetime import date

import yaml


def test_profile_yaml_shape():
    with open("config/risk_policy.yaml", encoding="utf-8") as f:
        policy = yaml.safe_load(f)
    profile = policy["exchange_recommendation"]["default_profile"]
    assert "target_usd_amount" in profile
    assert "usd_already_held" in profile
