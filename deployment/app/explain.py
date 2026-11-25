"""شرح التنبؤ: كل عامل زوّد السعر أو نقّصه بكام %.

بنستخدم قيم SHAP الجاهزة جوه XGBoost (pred_contribs=True). الموديل اتدرب على
log(السعر)، فكل مساهمة بتتجمع في الـ log، وبالتالي في السعر بتبقى "ضرب" في
exp(المساهمة). عشان كده بنعرض الأثر كنسبة مئوية: (exp(c) - 1) * 100.
"""
import numpy as np
import xgboost as xgb

from .features import build_feature_row
from .model_loader import ModelBundle, PredictionOutOfRange

# تجميع الأعمدة في مجموعات مفهومة (SHAP بيتجمع بدون مشاكل)
LOCATION = {"latitude", "longitude", "dist_center_km", "dist_nearest_landmark_km",
            "landmarks_1km", "landmarks_2km", "neighbourhood", "city"}
UNIT = {"accommodates", "bedrooms", "beds", "bathrooms", "bathroom_shared",
        "room_type", "property_type"}
HOST = {"host_is_superhost", "host_identity_verified", "host_tenure_years",
        "calculated_host_listings_count", "has_license"}
ACTIVITY = {"availability_30", "availability_365", "minimum_nights_avg_ntm",
            "number_of_reviews", "number_of_reviews_ltm", "reviews_per_month",
            "has_reviews", "days_since_last_review", "listing_age_days"}


def group_of(factor: str, amenity_columns: set) -> str:
    if factor in LOCATION:
        return "location"
    if factor in UNIT:
        return "unit_size_and_type"
    if factor == "n_amenities" or factor in amenity_columns:
        return "amenities"
    if factor in HOST:
        return "host"
    if factor in ACTIVITY:
        return "activity_and_availability"
    if factor.startswith("review_scores_"):
        return "review_scores"
    return "other"


def percent_effect(contribution: float) -> float:
    return float((np.exp(contribution) - 1.0) * 100.0)


def shap_contributions(bundle: ModelBundle, listing: dict):
    """بيرجّع (الصف، المساهمات لكل عمود، القيمة الأساسية). المجموع = توقع الـ log بالظبط."""
    features = build_feature_row(listing, bundle.artifacts, bundle.landmarks)
    raw = bundle.booster.predict(xgb.DMatrix(features), pred_contribs=True)[0]
    return features, raw[:-1], float(raw[-1])


def explain_price(bundle: ModelBundle, listing: dict, top_n: int = 8) -> dict:
    art = bundle.artifacts
    features, contribs, bias = shap_contributions(bundle, listing)
    row = features.iloc[0]
    columns = list(features.columns)

    log_price = bias + float(contribs.sum())
    price = float(np.expm1(log_price))
    if not np.isfinite(price) or price <= 0:
        raise PredictionOutOfRange(
            "The model returned a non-positive price for this input; the input is "
            "probably outside the range the model was trained on. Check the values."
        )

    # نجمّع أعمدة الـ one-hot في عامل واحد (city, room_type, property_type)
    families = {fam: [c for c in columns if c.startswith(f"{fam}_") and c not in art["num_features"]]
                for fam in art["cat_onehot"]}
    in_family = {c for cols in families.values() for c in cols}
    property_type = listing["property_type"]
    if property_type not in art["top_property_types"]:
        property_type = "Other"
    labels = {"city": listing["city"], "room_type": listing["room_type"],
              "property_type": property_type}

    factors = []  # (الاسم، القيمة، المساهمة)
    for col, c in zip(columns, contribs):
        if col in in_family:
            continue
        if col == "nbh_te":
            factors.append(("neighbourhood", listing.get("neighbourhood_cleansed") or "unknown", float(c)))
        else:
            factors.append((col, round(float(row[col]), 4), float(c)))
    index = {c: i for i, c in enumerate(columns)}
    for fam, cols in families.items():
        factors.append((fam, labels[fam], float(sum(contribs[index[c]] for c in cols))))

    amenity_columns = set(art["key_amenities"])
    groups = {}
    for name, _value, c in factors:
        g = group_of(name, amenity_columns)
        groups[g] = groups.get(g, 0.0) + c

    top = sorted(factors, key=lambda f: abs(f[2]), reverse=True)[:top_n]
    return {
        "predicted_price": round(price, 2),
        "baseline_price": round(float(np.expm1(bias)), 2),
        "groups": sorted(
            ({"name": g, "effect_percent": round(percent_effect(c), 1)} for g, c in groups.items()),
            key=lambda d: abs(d["effect_percent"]), reverse=True),
        "top_factors": [
            {"name": n, "value": v, "effect_percent": round(percent_effect(c), 1)} for n, v, c in top],
    }
