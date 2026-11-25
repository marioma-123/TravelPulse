import numpy as np
import pytest

from app.interval import compute_conformal_quantiles, price_interval


def test_quantile_covers_target_rate():
    rng = np.random.default_rng(0)
    y = rng.normal(4.5, 0.6, 5000)
    pred = y + rng.normal(0, 0.3, 5000)           # خطأ عشوائي
    cal = compute_conformal_quantiles(y[:2500], pred[:2500], levels=(0.8,))
    q = cal["quantiles"]["0.8"]
    covered = (np.abs(y[2500:] - pred[2500:]) <= q).mean()
    assert covered == pytest.approx(0.8, abs=0.03)


def test_interval_contains_point_and_is_ordered():
    conf = {"quantiles": {"0.8": 0.3}}
    lo, hi = price_interval(np.log1p(100.0), conf, 0.8)
    assert 0 <= lo < 100 < hi


def test_too_few_rows_rejected():
    with pytest.raises(ValueError):
        compute_conformal_quantiles(np.zeros(10), np.zeros(10))


def test_endpoint(client, full_listing):
    r = client.post("/predict-range", json=full_listing)
    assert r.status_code == 200
    b = r.json()
    assert b["lower"] <= b["predicted_price"] <= b["upper"] and b["confidence"] == 0.8
    wide = client.post("/predict-range?confidence=0.9", json=full_listing).json()
    assert wide["upper"] - wide["lower"] > b["upper"] - b["lower"]


def test_endpoint_rejects_unknown_confidence(client, full_listing):
    assert client.post("/predict-range?confidence=0.5", json=full_listing).status_code == 422


def test_endpoint_503_without_calibration(client, full_listing):
    client.app.state.bundle.artifacts.pop("conformal")
    assert client.post("/predict-range", json=full_listing).status_code == 503
