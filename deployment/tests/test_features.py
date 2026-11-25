"""اختبارات لكل خطوة تجهيز لوحدها (بدون الموديل)."""
import json
from datetime import date

import numpy as np
import pytest

from app.features import amenity_features, build_feature_row, geo_features
from app.model_loader import load_bundle
from models.make_dummy_model import KEY_AMENITIES


@pytest.fixture
def bundle(model_dir):
    return load_bundle(model_dir)


def row(bundle, listing):
    return build_feature_row(listing, bundle.artifacts, bundle.landmarks).iloc[0]


def test_amenity_features():
    out = amenity_features(["Wifi", "Dedicated workspace", "Air conditioning"], KEY_AMENITIES)
    assert out["n_amenities"] == 3
    assert out["has_wifi"] == 1 and out["has_ac"] == 1
    assert out["has_kitchen"] == 0
    assert amenity_features([], KEY_AMENITIES)["n_amenities"] == 0


def test_geo_features_with_known_points():
    landmarks = {"Barcelona": np.array([[41.3874, 2.1686], [41.3874 + 0.0120, 2.1686]])}
    # النقطة على بُعد حوالي 0.5 كم من المعلم الأول و0.83 كم من التاني (درجة العرض الواحدة ≈ 111 كم)
    out = geo_features("Barcelona", 41.3874 + 0.0045, 2.1686, landmarks)
    assert out["dist_nearest_landmark_km"] == pytest.approx(0.5, abs=0.02)
    assert out["landmarks_1km"] == 2
    assert out["landmarks_2km"] == 2
    assert out["dist_center_km"] == pytest.approx(0.5, abs=0.02)


def test_columns_match_artifacts_order(bundle, minimal_listing):
    X = build_feature_row(minimal_listing, bundle.artifacts, bundle.landmarks)
    assert list(X.columns) == bundle.artifacts["feature_columns"]
    assert X.shape[0] == 1


def test_unknown_property_type_becomes_other(bundle, minimal_listing):
    minimal_listing["property_type"] = "Treehouse"
    r = row(bundle, minimal_listing)
    assert r["property_type_Other"] == 1
    assert r["property_type_Entire rental unit"] == 0


def test_known_categories_are_one_hot(bundle, minimal_listing):
    r = row(bundle, minimal_listing)
    assert r["city_Barcelona"] == 1 and r["city_Berlin"] == 0
    assert r["room_type_Entire home/apt"] == 1
    assert r["property_type_Entire rental unit"] == 1


def test_missing_bedrooms_uses_group_median_then_overall(bundle, minimal_listing):
    r = row(bundle, minimal_listing)  # (Entire home/apt, 4) موجودة في الجدول
    assert r["bedrooms"] == 2.0
    minimal_listing["accommodates"] = 7  # مجموعة مش موجودة -> median عام
    r = row(bundle, minimal_listing)
    assert r["bedrooms"] == bundle.artifacts["train_medians"]["bedrooms"]


def test_missing_reviews_per_month_is_zero(bundle, minimal_listing):
    assert row(bundle, minimal_listing)["reviews_per_month"] == 0.0


def test_missing_other_numeric_uses_train_median(bundle, minimal_listing):
    r = row(bundle, minimal_listing)
    assert r["availability_30"] == bundle.artifacts["train_medians"]["availability_30"]


def test_review_dates_become_days(bundle, minimal_listing):
    minimal_listing.update(
        first_review=date(2025, 9, 1), last_review=date(2025, 9, 10), as_of_date=date(2025, 9, 20)
    )
    r = row(bundle, minimal_listing)
    assert r["days_since_last_review"] == 10
    assert r["listing_age_days"] == 19


def test_default_reference_date_comes_from_artifacts(bundle, minimal_listing):
    minimal_listing["last_review"] = date(2025, 9, 10)  # reference_date = 2025-09-20
    assert row(bundle, minimal_listing)["days_since_last_review"] == 10


def test_neighbourhood_encoding_known_unknown_and_missing(bundle, minimal_listing):
    art = bundle.artifacts
    minimal_listing["neighbourhood_cleansed"] = "Gracia"
    assert row(bundle, minimal_listing)["nbh_te"] == 4.9
    minimal_listing["neighbourhood_cleansed"] = "Nowhere"
    assert row(bundle, minimal_listing)["nbh_te"] == art["nbh_global_mean"]
    minimal_listing["neighbourhood_cleansed"] = None
    assert row(bundle, minimal_listing)["nbh_te"] == art["nbh_global_mean"]


def test_has_reviews_flag(bundle, minimal_listing):
    assert row(bundle, minimal_listing)["has_reviews"] == 0
    minimal_listing["number_of_reviews"] = 3
    assert row(bundle, minimal_listing)["has_reviews"] == 1
