import base64
import hashlib
import hmac
import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.database.schema import Base
from app.api.main import _session


@pytest.fixture(autouse=True)
def isolated_database(monkeypatch):
    monkeypatch.setattr("app.risk.scoring.explain_latest_prediction", lambda *args: [])
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    def session_override():
        with Session(engine) as session:
            yield session
    app.dependency_overrides[_session] = session_override
    yield engine
    app.dependency_overrides.clear()
    engine.dispose()


from app.api.main import app


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_line_webhook_returns_user_ids(monkeypatch):
    client = TestClient(app)
    monkeypatch.setenv("LINE_CHANNEL_SECRET", "test-secret")
    body = json.dumps({"events": [{"source": {"userId": "U123"}}]}).encode()
    signature = base64.b64encode(hmac.new(b"test-secret", body, hashlib.sha256).digest()).decode()
    response = client.post("/line/webhook", content=body, headers={"x-line-signature": signature})
    assert response.status_code == 200
    assert response.json()["user_ids"] == ["U123"]


def test_ai_latest_empty_database():
    client = TestClient(app)
    response = client.get("/ai/latest")
    assert response.status_code == 200
    assert response.json()["provider"] == "openai"


def test_strategy_backtest_endpoint_returns_expected_shape():
    client = TestClient(app)
    response = client.get("/backtests/strategy")
    assert response.status_code == 200
    payload = response.json()
    assert "summary" in payload
    assert "strategies" in payload


@pytest.mark.parametrize("path", ["/overview", "/features/latest"])
def test_missing_features_returns_service_unavailable(path):
    assert TestClient(app).get(path).status_code == 503


@pytest.mark.parametrize("query", ["limit=-1", "limit=0", "limit=1000000"])
def test_chart_rejects_invalid_limit(query):
    assert TestClient(app).get("/charts/usdtwd?" + query).status_code == 422


@pytest.mark.parametrize("amount", ["0", "-1", "nan", "inf"])
def test_backtest_rejects_invalid_amount(amount):
    assert TestClient(app).get("/backtests/strategy?target_usd=" + amount).status_code == 422


def test_webhook_rejects_unsigned_or_forged_request(monkeypatch):
    monkeypatch.setenv("LINE_CHANNEL_SECRET", "test-secret")
    client = TestClient(app)
    for headers in ({}, {"x-line-signature": "forged"}):
        assert client.post("/line/webhook", json={"events": []}, headers=headers).status_code == 401


def test_overview_blocks_stale_guidance_and_read_does_not_write(isolated_database):
    from datetime import datetime
    from sqlalchemy import func, select
    from app.database.schema import Feature, Prediction
    with Session(isolated_database) as session:
        session.add(Feature(observed_at_utc=datetime(2020, 1, 1), source="test", feature_set="daily_v3",
                            values_json=json.dumps({"USDTWD_CLOSE": 31.0, "DATA_COMPLETENESS": 0.0})))
        session.commit()
    response = TestClient(app).get("/overview")
    assert response.status_code == 200
    payload = response.json()
    assert payload["exchange"] is None
    assert payload["advice_available"] is False
    with Session(isolated_database) as session:
        assert session.scalar(select(func.count()).select_from(Prediction)) == 0


def test_forward_validation_endpoint_starts_empty():
    payload = TestClient(app).get("/forward-validation").json()
    assert payload["status"] == "NOT_STARTED"
    assert payload["automatic_promotion"] is False


def test_pipeline_status_is_read_only_and_does_not_claim_first_run(monkeypatch, tmp_path):
    path = tmp_path / "operations"
    monkeypatch.setattr("app.scheduler.state.STATE_DIR", path)
    response = TestClient(app).get("/pipeline/status")
    assert response.status_code == 200
    assert response.json()["status"] == "NO_RECORD"
    assert not path.exists()


def test_blocked_advice_hides_legacy_ai_on_both_public_endpoints(isolated_database):
    from datetime import datetime, timezone
    from sqlalchemy import func, select
    from app.database.schema import AIInterpretation, Feature

    with Session(isolated_database) as session:
        session.add(Feature(observed_at_utc=datetime(2024, 1, 8), source="test", feature_set="daily_v3",
                            values_json=json.dumps({"USDTWD_CLOSE": 31., "DATA_COMPLETENESS": 0.})))
        session.add(AIInterpretation(observed_at_utc=datetime.now(timezone.utc), source="openai",
                                    provider="openai", model="legacy-model",
                                    summary_zh_tw="舊模型上漲機率 99%", raw_response="legacy-private-context"))
        session.commit()
    client = TestClient(app)
    overview = client.get("/overview").json()
    assert overview["advice_available"] is False
    assert overview["ai_interpretation"] is None
    ai = client.get("/ai/latest").json()
    assert ai["enabled"] is False and ai["reason"] == "MODEL_ADVICE_BLOCKED"
    assert "99%" not in json.dumps(overview) + json.dumps(ai)
    assert "legacy-private-context" not in json.dumps(overview) + json.dumps(ai)
    with Session(isolated_database) as session:
        assert session.scalar(select(func.count()).select_from(AIInterpretation)) == 1


def test_overview_never_fills_missing_current_horizon_with_old_or_future_forecast(isolated_database):
    from datetime import datetime
    from app.database.schema import Feature, Prediction

    origin = datetime(2024, 1, 8)
    with Session(isolated_database) as session:
        session.add(Feature(observed_at_utc=origin, source="test", feature_set="daily_v3",
                            values_json=json.dumps({"USDTWD_CLOSE": 31., "DATA_COMPLETENESS": 0.})))
        for date, horizon, source in [(datetime(2024, 1, 5), "5d", "model_training_latest"),
                                      (datetime(2024, 1, 9), "20d", "model_training_latest"),
                                      (origin, "5d", "historical_research"),
                                      (origin, "1d", "model_training_latest")]:
            session.add(Prediction(observed_at_utc=date, source=source, model_version="phase4_ensemble_v3",
                                   horizon=horizon, prob_up=.6, prob_down=.4, confidence=.9))
        session.commit()
    predictions = TestClient(app).get("/overview").json()["predictions"]
    assert set(predictions) == {"1d"}
    assert predictions["1d"]["origin_utc"] == "2024-01-08T00:00:00+00:00"
    assert predictions["1d"]["confidence_method"].endswith("not_accuracy")


@pytest.mark.parametrize("observed,available", [
    ("2024-01-05 12:00", False),  # Previous close's narrative.
    ("2024-01-08 08:59", False),  # Before the official close.
    ("2024-01-08 12:00", True),
    ("2024-01-09 09:00", False),  # At the next close, origin is no longer current.
    ("2099-01-08 12:00", False),
])
def test_ai_interpretation_requires_matching_close_window(isolated_database, observed, available):
    from types import SimpleNamespace
    import pandas as pd
    from app.api.main import _current_ai_interpretation
    from app.database.schema import AIInterpretation

    with Session(isolated_database) as session:
        session.add(AIInterpretation(observed_at_utc=pd.Timestamp(observed).to_pydatetime(),
                                    source="openai", provider="openai", model="test", summary_zh_tw="研究解讀"))
        session.commit()
        result = _current_ai_interpretation(session, SimpleNamespace(blocks_model_advice=False),
                                            pd.Timestamp("2024-01-08", tz="UTC"))
        assert (result is not None) == available
        if result:
            assert result.observed_at_utc == "2024-01-08T12:00:00+00:00"
