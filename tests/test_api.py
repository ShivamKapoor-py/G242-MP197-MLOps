
import pytest
from fastapi.testclient import TestClient

from src.mp197.api import app

client = TestClient(app)


VALID_INPUT = {
    "hour": 8,
    "nearby_offices": 20,
    "temp_c": 28.5,
    "rain_mm": 0.0,
    "feeder_buses": 10,
    "ticket_price_rs": 35,
    "day_type": "weekday",
    "line": "Blue Line",
}


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_valid_prediction():
    response = client.post("/predict", json=VALID_INPUT)

    assert response.status_code == 200

    result = response.json()
    assert "predicted_entries_per_hour" in result
    assert isinstance(result["predicted_entries_per_hour"], float)


@pytest.mark.parametrize(
    "field,value",
    [
        ("ticket_price_rs", 100),
        ("rain_mm", -5),
        ("hour", 25),
    ],
)
def test_invalid_inputs(field, value):
    payload = VALID_INPUT.copy()
    payload[field] = value

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_extra_field_rejected():
    payload = VALID_INPUT.copy()
    payload["unknown_field"] = 10

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_invalid_day_type():
    payload = VALID_INPUT.copy()
    payload["day_type"] = "workingday"

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
