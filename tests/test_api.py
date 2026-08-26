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
