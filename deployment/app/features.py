"""بيحوّل مواصفات وحدة خام لصف الأعمدة اللي الموديل اتدرب عليه.

كل خطوة هنا منقولة من الـ notebooks (09_cleaning_listing و14_merge_final و
Airbnb_Price_Prediction). لو الـ notebook اتغير، الملف ده لازم يتغير معاه
(واختبار test_parity بيكشف أي اختلاف).
"""
from datetime import date

import numpy as np
import pandas as pd

from .config import CITY_CENTERS

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    """المسافة بالكيلومتر بين نقطتين على الكرة الأرضية (نفس دالة 14_merge_final)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def geo_features(city: str, lat: float, lon: float, landmarks: dict) -> dict:
    """المسافات: لمركز المدينة، وأقرب معلم، وعدد المعالم في 1 كم و2 كم."""
    lm = landmarks[city]  # مصفوفة (عدد المعالم × 2): latitude, longitude
    d = haversine_km(lat, lon, lm[:, 0], lm[:, 1])
    c_lat, c_lon = CITY_CENTERS[city]
    return {
        "dist_nearest_landmark_km": float(d.min()),
        "landmarks_1km": int((d <= 1).sum()),
        "landmarks_2km": int((d <= 2).sum()),
        "dist_center_km": float(haversine_km(lat, lon, c_lat, c_lon)),
    }


def amenity_features(amenities: list, key_amenities: dict) -> dict:
    """عدد المرافق + أعلام (0/1) للمرافق المهمة (wifi, kitchen, ...)."""
    lowered = [a.lower() for a in amenities]
    out = {"n_amenities": len(amenities)}
    for column, keyword in key_amenities.items():
        out[column] = int(any(keyword in a for a in lowered))
    return out


def build_feature_row(listing: dict, art: dict, landmarks: dict) -> pd.DataFrame:
    """بياخد وحدة (dict من ListingInput) ويرجّع DataFrame من صف واحد بنفس أعمدة التدريب."""
    columns = art["feature_columns"]
    medians = art["train_medians"]

    # --- التواريخ -> أيام ---
    as_of = listing.get("as_of_date") or date.fromisoformat(art["reference_date"])
    last_review, first_review = listing.get("last_review"), listing.get("first_review")
    days_since_last_review = (as_of - last_review).days if last_review else None
    listing_age_days = (as_of - first_review).days if first_review else None

    # --- القيم الخام (None = ناقصة) ---
    values = {
        "latitude": listing["latitude"],
        "longitude": listing["longitude"],
        **geo_features(listing["city"], listing["latitude"], listing["longitude"], landmarks),
        "accommodates": listing["accommodates"],
        "bedrooms": listing.get("bedrooms"),
        "beds": listing.get("beds"),
        "bathrooms": listing.get("bathrooms"),
        "bathroom_shared": int(listing.get("bathroom_shared", False)),
        "has_license": int(listing.get("has_license", False)),
        **amenity_features(listing.get("amenities") or [], art["key_amenities"]),
        "host_is_superhost": int(listing.get("host_is_superhost", False)),
        "host_identity_verified": int(listing.get("host_identity_verified", False)),
        "host_tenure_years": listing.get("host_tenure_years"),
        "calculated_host_listings_count": listing.get("calculated_host_listings_count"),
        "availability_30": listing.get("availability_30"),
        "availability_365": listing.get("availability_365"),
        "minimum_nights_avg_ntm": listing.get("minimum_nights_avg_ntm"),
        "number_of_reviews": listing.get("number_of_reviews"),
        "number_of_reviews_ltm": listing.get("number_of_reviews_ltm"),
        "reviews_per_month": listing.get("reviews_per_month"),
        "has_reviews": int((listing.get("number_of_reviews") or 0) > 0),
        "days_since_last_review": days_since_last_review,
        "listing_age_days": listing_age_days,
        "review_scores_rating": listing.get("review_scores_rating"),
        "review_scores_location": listing.get("review_scores_location"),
        "review_scores_cleanliness": listing.get("review_scores_cleanliness"),
        "review_scores_value": listing.get("review_scores_value"),
    }

    # --- ملء الناقص بنفس ترتيب الـ notebook ---
    for col in ("bedrooms", "beds"):
        if values[col] is None:
            key = f"{listing['room_type']}|{listing['accommodates']}"
            values[col] = art["group_medians"][col].get(key, medians[col])
    if values["reviews_per_month"] is None:
        values["reviews_per_month"] = 0.0
    for col in art["num_features"]:
        if values[col] is None:
            values[col] = medians[col]

    # --- ترميز الحي (target encoding): قيمة جاهزة من جدول التدريب ---
    nbh_key = f"{listing['city']} | {listing.get('neighbourhood_cleansed') or 'unknown'}"
    nbh_te = art["nbh_encoding"].get(nbh_key, art["nbh_global_mean"])

    # --- تجميع الصف: كله أصفار، وبنملا الأرقام والـ one-hot ---
    row = dict.fromkeys(columns, 0.0)
    for col in art["num_features"]:
        row[col] = float(values[col])
    row["nbh_te"] = float(nbh_te)

    property_type = listing["property_type"]
    if property_type not in art["top_property_types"]:
        property_type = "Other"
    for col, val in (
        ("city", listing["city"]),
        ("room_type", listing["room_type"]),
        ("property_type", property_type),
    ):
        name = f"{col}_{val}"
        if name in row:  # قيمة ماشفناهاش وقت التدريب = كل أعمدتها أصفار
            row[name] = 1.0

    return pd.DataFrame([row], columns=columns)