from fastapi.testclient import TestClient

from app.api.main import app


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_line_webhook_returns_user_ids():
    client = TestClient(app)
    response = client.post("/line/webhook", json={"events": [{"source": {"userId": "U123"}}]})
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
