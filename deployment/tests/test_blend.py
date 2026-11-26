"""اختبار الخليط (XGBoost + LightGBM + CatBoost): التحميل، الأوزان، التطابق مع
ملف parity_sample_blend.csv، وهوية SHAP (القيمة الأساسية + المساهمات = التوقع).

بيشتغل بس لما ملفات models/blend/ و models/parity_sample_blend.csv موجودة.
"""
import json
import os

import numpy as np
import pandas as pd
import pytest

from app.config import get_model_dir
from app.features import build_feature_row
from app.model_loader import load_bundle, predict_price
from app.schemas import ListingInput
from test_parity import to_payload

SAMPLE = get_model_dir() / "parity_sample_blend.csv"
pytestmark = pytest.mark.skipif(
    not (SAMPLE.exists() and (get_model_dir() / "blend").exists()),
    reason="blend model files or parity_sample_blend.csv not found",
)


@pytest.fixture(scope="module")
def bundle():
    os.environ["MODEL_MODE"] = "blend"  # التثبيت جوه الـ fixture عمداً (انظر test_parity)
    return load_bundle(get_model_dir())


@pytest.fixture(scope="module")
def sample():
    return pd.read_csv(SAMPLE)


def test_blend_loaded_by_default(bundle):
    assert bundle.mode == "blend_xgb_lgbm_cat"
    weights = bundle.blend.weights
    assert set(weights) == {"xgb", "lgbm", "cat"}
    assert sum(weights.values()) == pytest.approx(1.0)
    assert all(w > 0 for w in weights.values())


def test_features_and_prediction_match_training(bundle, sample):
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


def test_blend_shap_contributions_add_up(bundle, sample):
    """القيمة الأساسية الموزونة + مجموع المساهمات الموزونة = توقع الخليط في الـ log بالظبط."""
    from app.explain import shap_contributions

    ref = bundle.artifacts["reference_date"]
    for i, raw_row in sample.head(20).iterrows():
        listing = ListingInput(**to_payload(raw_row, ref)).model_dump()
        _features, contribs, bias = shap_contributions(bundle, listing)
        assert np.expm1(bias + contribs.sum()) == pytest.approx(raw_row["expected_price"], rel=1e-3), f"row {i}"


def test_baseline_mode_still_available(sample):
    """الرجوع للموديل القديم بـ MODEL_MODE=baseline لازم يفضل شغال."""
    os.environ["MODEL_MODE"] = "baseline"
    try:
        bundle = load_bundle(get_model_dir())
        assert bundle.mode == "xgb_baseline"
        ref = bundle.artifacts["reference_date"]
        listing = ListingInput(**to_payload(sample.iloc[0], ref)).model_dump()
        assert predict_price(bundle, listing) > 0
    finally:
        os.environ["MODEL_MODE"] = "blend"
