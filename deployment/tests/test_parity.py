"""اختبار التطابق مع الـ notebook (الأهم): نفس المدخلات لازم تدي نفس الأعمدة ونفس السعر.

بيشتغل بس لما ملف models/parity_sample.csv (من خلية التصدير) يكون موجود.
"""
import json

import numpy as np
import pandas as pd
import pytest

from app.config import get_model_dir
from app.features import build_feature_row
from app.model_loader import load_bundle, predict_price
from app.schemas import ListingInput

SAMPLE = get_model_dir() / "parity_sample.csv"
pytestmark = pytest.mark.skipif(not SAMPLE.exists(), reason="parity_sample.csv not found")

RAW_COLUMNS = [
    "city", "neighbourhood_cleansed", "latitude", "longitude", "room_type", "property_type",
    "accommodates", "bedrooms", "beds", "bathrooms", "bathroom_shared", "has_license",
    "amenities", "host_is_superhost", "host_identity_verified", "host_tenure_years",
    "calculated_host_listings_count", "availability_30", "availability_365",
    "minimum_nights_avg_ntm", "number_of_reviews", "number_of_reviews_ltm",
    "reviews_per_month", "review_scores_rating", "review_scores_location",
    "review_scores_cleanliness", "review_scores_value", "first_review", "last_review",
]


def to_payload(raw_row, reference_date):
    payload = {}
    for col in RAW_COLUMNS:
        value = raw_row[col]
        if col == "amenities":
            payload[col] = json.loads(value) if isinstance(value, str) else []
        elif pd.isna(value):
            continue  # الحقل الناقص بنسيبه فاضي (الـ API بتملاه)
        elif isinstance(value, (np.integer, np.floating)):
            payload[col] = value.item()
        else:
            payload[col] = value
    payload["as_of_date"] = reference_date
    return payload


@pytest.fixture(scope="module")
def bundle():
    return load_bundle(get_model_dir())


@pytest.fixture(scope="module")
def sample():
    return pd.read_csv(SAMPLE)


def test_features_and_prediction_match_notebook(bundle, sample):
    x_cols = [c for c in sample.columns if c.startswith("X__")]
    assert [c[3:] for c in x_cols] == bundle.artifacts["feature_columns"]
    ref = bundle.artifacts["reference_date"]

    for i, raw_row in sample.iterrows():
        listing = ListingInput(**to_payload(raw_row, ref)).model_dump()
        ours = build_feature_row(listing, bundle.artifacts, bundle.landmarks).iloc[0]
        expected = raw_row[x_cols].astype(float)
        expected.index = [c[3:] for c in x_cols]
        diff = (ours - expected).abs()
        assert diff.max() < 1e-6, f"row {i}: feature mismatch in {diff.idxmax()} ({diff.max()})"

        price = predict_price(bundle, listing)
        assert price == pytest.approx(raw_row["expected_price"], rel=1e-4), f"row {i}: price"


def test_shap_contributions_add_up_on_real_rows(bundle, sample):
    """على بيانات الـ notebook: القيمة الأساسية + مجموع مساهمات SHAP = log(1+السعر) المتوقع."""
    from app.explain import shap_contributions

    ref = bundle.artifacts["reference_date"]
    for i, raw_row in sample.head(60).iterrows():
        listing = ListingInput(**to_payload(raw_row, ref)).model_dump()
        _features, contribs, bias = shap_contributions(bundle, listing)
        assert np.expm1(bias + contribs.sum()) == pytest.approx(raw_row["expected_price"], rel=1e-3), f"row {i}"
