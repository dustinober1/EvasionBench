from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_predict():
    payload = {
        "question": "What are your revenue expectations?",
        "answer": "We expect continued growth.",
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    j = res.json()
    assert "prediction" in j
    assert "confidence" in j
