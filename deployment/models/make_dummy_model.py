"""بيبني حزمة موديل وهمية (XGBoost صغير + ملفات مساعدة) للتجربة قبل وصول الحقيقي.

    python -m models.make_dummy_model          (من جوه deployment/)
بيكتب في models/dummy/ (مش بيلمس ملفات الموديل الحقيقي)، وبعدها:
    MODEL_DIR=models/dummy uvicorn app.main:app --reload
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from app.config import ARTIFACTS_FILE, BASE_DIR, CITY_CENTERS, LANDMARKS_FILE

DUMMY_DIR = BASE_DIR / "models" / "dummy"

# نفس ترتيب num_features في notebook الموديل
KEY_AMENITIES = {
    "has_wifi": "wifi", "has_kitchen": "kitchen", "has_ac": "air conditioning",
    "has_pool": "pool", "has_parking": "parking", "has_washer": "washer",
    "has_tv": "tv", "has_heating": "heating", "has_elevator": "elevator",
}
NUM_FEATURES = [
    "latitude", "longitude", "dist_center_km", "dist_nearest_landmark_km",
    "landmarks_1km", "landmarks_2km",
    "accommodates", "bedrooms", "beds", "bathrooms", "bathroom_shared", "has_license",
    "n_amenities", *KEY_AMENITIES.keys(),
    "host_is_superhost", "host_identity_verified", "host_tenure_years",
    "calculated_host_listings_count",
    "availability_30", "availability_365", "minimum_nights_avg_ntm",
    "number_of_reviews", "number_of_reviews_ltm", "reviews_per_month",
    "has_reviews", "days_since_last_review", "listing_age_days",
    "review_scores_rating", "review_scores_location",
    "review_scores_cleanliness", "review_scores_value",
]
ROOM_TYPES = ["Entire home/apt", "Private room", "Shared room", "Hotel room"]
TOP_PROPERTY_TYPES = ["Entire rental unit", "Private room in rental unit", "Entire condo"]


def build_dummy_bundle(out_dir: Path = DUMMY_DIR, n_samples: int = 600, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cities = list(CITY_CENTERS)

    dummy_cols = (
        [f"city_{c}" for c in cities]
        + [f"room_type_{r}" for r in ROOM_TYPES]
        + [f"property_type_{p}" for p in TOP_PROPERTY_TYPES + ["Other"]]
    )
    feature_columns = NUM_FEATURES + ["nbh_te"] + dummy_cols

    X = pd.DataFrame(rng.random((n_samples, len(feature_columns))), columns=feature_columns)
    X["accommodates"] = rng.integers(1, 9, n_samples).astype(float)
    X["bedrooms"] = rng.integers(0, 5, n_samples).astype(float)
    X["nbh_te"] = rng.normal(4.8, 0.4, n_samples)
    # "log price" مصطنع: بيعتمد على عدد الضيوف والغرف والحي فقط
    y = 3.0 + 0.12 * X["accommodates"] + 0.08 * X["bedrooms"] + 0.3 * X["nbh_te"]
    booster = xgb.train({"max_depth": 3, "eta": 0.3}, xgb.DMatrix(X, label=y), num_boost_round=30)
    booster.save_model(str(out_dir / "xgb_model.json"))

    artifacts = {
        "model_file": "xgb_model.json",
        "feature_columns": feature_columns,
        "num_features": NUM_FEATURES,
        "cat_onehot": ["city", "room_type", "property_type"],
        "key_amenities": KEY_AMENITIES,
        "top_property_types": TOP_PROPERTY_TYPES,
        "train_medians": {c: float(X[c].median()) for c in NUM_FEATURES},
        "group_medians": {
            "bedrooms": {"Entire home/apt|2": 1.0, "Entire home/apt|4": 2.0},
            "beds": {"Entire home/apt|2": 1.0, "Entire home/apt|4": 3.0},
        },
        "nbh_encoding": {"Barcelona | Gracia": 4.9, "Berlin | Mitte": 4.7},
        "nbh_global_mean": 4.8,
        "reference_date": "2025-09-20",
        "conformal": {"n_calibration": 500, "quantiles": {"0.8": 0.25, "0.9": 0.4}},
    }
    (out_dir / ARTIFACTS_FILE).write_text(json.dumps(artifacts))

    rows = []
    for city, (lat, lon) in CITY_CENTERS.items():
        for _ in range(30):
            rows.append({"city": city, "latitude": lat + rng.normal(0, 0.03),
                         "longitude": lon + rng.normal(0, 0.04)})
    pd.DataFrame(rows).to_csv(out_dir / LANDMARKS_FILE, index=False)
    return out_dir


if __name__ == "__main__":
    print(f"Dummy model bundle saved to: {build_dummy_bundle()}")
