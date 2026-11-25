"""فحص سريع لكل مدينة باستخدام عيّنة الـ notebook (models/parity_sample.csv).

    python -m tools.check_by_city            (من جوه deployment/)

بيطبع: متوسط أسعار الـ notebook لكل مدينة، ونتيجة الـ API على نفس الصفوف،
وطلب جاهز (JSON) لوحدة حقيقية من كل مدينة تقدر تلصقه في /docs وتقارن.
"""
import json

import numpy as np
import pandas as pd

from app.config import get_model_dir
from app.model_loader import PredictionOutOfRange, load_bundle, predict_price
from app.schemas import ListingInput

RAW_COLUMNS = [
    "city", "neighbourhood_cleansed", "latitude", "longitude", "room_type", "property_type",
    "accommodates", "bedrooms", "beds", "bathrooms", "bathroom_shared", "has_license",
    "amenities", "host_is_superhost", "host_identity_verified", "host_tenure_years",
    "calculated_host_listings_count", "availability_30", "availability_365",
    "minimum_nights_avg_ntm", "number_of_reviews", "number_of_reviews_ltm",
    "reviews_per_month", "review_scores_rating", "review_scores_location",
    "review_scores_cleanliness", "review_scores_value", "first_review", "last_review",
]


def to_payload(row, reference_date):
    payload = {}
    for col in RAW_COLUMNS:
        value = row[col]
        if col == "amenities":
            payload[col] = json.loads(value) if isinstance(value, str) else []
        elif pd.isna(value):
            continue
        elif isinstance(value, (np.integer, np.floating)):
            payload[col] = value.item()
        else:
            payload[col] = value
    payload["as_of_date"] = reference_date
    return payload


def main():
    bundle = load_bundle(get_model_dir())
    sample = pd.read_csv(get_model_dir() / "parity_sample.csv")
    ref = bundle.artifacts["reference_date"]

    api_prices = []
    for _, row in sample.iterrows():
        try:
            api_prices.append(predict_price(bundle, ListingInput(**to_payload(row, ref)).model_dump()))
        except PredictionOutOfRange:
            api_prices.append(0.0)
    sample["api_price"] = api_prices

    print("\n== per city: notebook price vs API price on the same rows (USD) ==")
    summary = sample.groupby("city").agg(
        rows=("city", "size"),
        notebook_min=("expected_price", "min"),
        notebook_median=("expected_price", "median"),
        notebook_max=("expected_price", "max"),
        api_median=("api_price", "median"),
        api_zero_rows=("api_price", lambda s: int((s <= 0).sum())),
    )
    print(summary.round(1).to_string())

    print("\n== one real listing per city (paste into /docs -> POST /predict) ==")
    for city, group in sample.groupby("city"):
        row = group.iloc[0]
        print(f"\n--- {city}: notebook price = {row['expected_price']:.2f}, API price = {row['api_price']:.2f}")
        print(json.dumps(to_payload(row, ref), indent=2, default=str, ensure_ascii=False))


if __name__ == "__main__":
    main()
