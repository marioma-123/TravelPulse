import math

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas import ListingInput
from app.model_loader import PredictionOutOfRange, load_bundle, predict_price


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True, "model": "xgb_baseline"}


def test_predict_full_listing(client, full_listing):
    response = client.post("/predict", json=full_listing)
    assert response.status_code == 200
    body = response.json()
    assert body["city"] == "Barcelona"
    assert body["currency"] == "USD"
    assert math.isfinite(body["predicted_price"]) and body["predicted_price"] >= 0


def test_predict_minimal_listing_uses_imputation(client, minimal_listing):
    response = client.post("/predict", json=minimal_listing)
    assert response.status_code == 200


def test_predict_accepts_unseen_property_type(client, minimal_listing):
    minimal_listing["property_type"] = "Treehouse"
    assert client.post("/predict", json=minimal_listing).status_code == 200


@pytest.mark.parametrize(
    "change",
    [
        {"city": "Atlantis"},                       # مدينة غير مسموحة
        {"accommodates": 0},                        # خارج الحدود
        {"room_type": "Castle"},                    # نوع غرفة غير مسموح
        {"unexpected_field": 1},                    # حقل غير معرّف
        {"latitude": 52.5200, "longitude": 13.4050},  # إحداثيات برلين لوحدة في برشلونة
        {"last_review": "not-a-date"},              # تاريخ غلط
    ],
)
def test_predict_rejects_invalid_input(client, minimal_listing, change):
    minimal_listing.update(change)
    assert client.post("/predict", json=minimal_listing).status_code == 422


def test_predict_rejects_missing_required_field(client, minimal_listing):
    del minimal_listing["accommodates"]
    assert client.post("/predict", json=minimal_listing).status_code == 422


def test_app_fails_to_start_without_model_files(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        with TestClient(app):
            pass


class _NegativeBooster:
    """موديل مزيّف بيرجّع log سالب، عشان نجرّب حالة السعر غير المنطقي."""

    def predict(self, _data):
        return np.array([-0.5])


def test_non_positive_prediction_raises(model_dir, minimal_listing):
    bundle = load_bundle(model_dir)
    bundle.booster = _NegativeBooster()
    with pytest.raises(PredictionOutOfRange):
        predict_price(bundle, ListingInput(**minimal_listing).model_dump())


def test_non_positive_prediction_returns_422_not_zero(client, minimal_listing):
    client.app.state.bundle.booster = _NegativeBooster()
    response = client.post("/predict", json=minimal_listing)
    assert response.status_code == 422
    assert "non-positive" in response.json()["detail"]


def test_ui_is_served(client):
    r = client.get("/ui/")
    assert r.status_code == 200 and "TravelPulse" in r.text
