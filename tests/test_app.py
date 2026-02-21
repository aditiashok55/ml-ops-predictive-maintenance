import pytest
import numpy as np
from unittest.mock import patch, MagicMock


@pytest.fixture
def client():
    with patch("mlflow.xgboost.load_model", return_value=MagicMock()):
        from fastapi.testclient import TestClient
        from src.inference.app import app
        return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Predictive Maintenance API" in response.json()["service"]


def test_predict_returns_correct_fields(client):
    import src.inference.app as app_module
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([85.0])
    app_module.model = mock_model

    payload = {
        "engine_id": 1, "cycle": 50,
        "op_setting_1": 0.5, "op_setting_2": 0.4, "op_setting_3": 0.3,
        "sensor_2": 0.6, "sensor_3": 0.5, "sensor_4": 0.7,
        "sensor_7": 0.4, "sensor_8": 0.8, "sensor_9": 0.6,
        "sensor_11": 0.5, "sensor_12": 0.4, "sensor_13": 0.6,
        "sensor_14": 0.7, "sensor_15": 0.3, "sensor_17": 0.5,
        "sensor_20": 0.4, "sensor_21": 0.6
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_rul" in data
    assert "warning_level" in data
    assert "latency_ms" in data
    assert data["warning_level"] == "NORMAL"