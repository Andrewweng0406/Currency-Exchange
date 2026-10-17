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
