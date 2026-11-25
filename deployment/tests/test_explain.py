import math

import numpy as np
import pytest

from app.explain import explain_price, percent_effect, shap_contributions
from app.model_loader import load_bundle, predict_price
from app.schemas import ListingInput


@pytest.fixture
def bundle(model_dir):
    return load_bundle(model_dir)


def test_contributions_add_up_to_the_prediction(bundle, full_listing):
    """الخاصية الأساسية لـ SHAP: القيمة الأساسية + مجموع المساهمات = توقع الـ log."""
    listing = ListingInput(**full_listing).model_dump()
    _features, contribs, bias = shap_contributions(bundle, listing)
    log_price = bias + contribs.sum()
    assert np.expm1(log_price) == pytest.approx(predict_price(bundle, listing), rel=1e-4)


def test_explanation_price_matches_predict(bundle, full_listing):
    listing = ListingInput(**full_listing).model_dump()
    result = explain_price(bundle, listing)
    assert result["predicted_price"] == pytest.approx(predict_price(bundle, listing), abs=0.01)
    assert result["baseline_price"] > 0


def test_top_factors_sorted_and_limited(bundle, full_listing):
    result = explain_price(bundle, ListingInput(**full_listing).model_dump(), top_n=5)
    effects = [abs(f["effect_percent"]) for f in result["top_factors"]]
    assert len(effects) == 5
    assert effects == sorted(effects, reverse=True)


def test_one_hot_families_are_merged(bundle, full_listing):
    names = {f["name"] for f in explain_price(bundle, ListingInput(**full_listing).model_dump(), top_n=100)["top_factors"]}
    assert {"city", "room_type", "property_type", "neighbourhood"} <= names
    assert not any(n.startswith("city_") for n in names)


def test_percent_effect_examples():
    assert percent_effect(0.0) == 0.0
    assert percent_effect(math.log(1.10)) == pytest.approx(10.0)
    assert percent_effect(math.log(0.8)) == pytest.approx(-20.0)


def test_explain_endpoint(client, full_listing):
    response = client.post("/explain", json=full_listing)
    assert response.status_code == 200
    body = response.json()
    assert body["currency"] == "USD" and body["city"] == "Barcelona"
    assert body["groups"] and body["top_factors"]
    assert all(set(f) == {"name", "value", "effect_percent"} for f in body["top_factors"])
    assert body["predicted_price"] == pytest.approx(
        client.post("/predict", json=full_listing).json()["predicted_price"], abs=0.01)


def test_explain_rejects_invalid_input(client, full_listing):
    full_listing["city"] = "Atlantis"
    assert client.post("/explain", json=full_listing).status_code == 422
