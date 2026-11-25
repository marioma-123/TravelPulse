"""
بيقيّم كل listing عليه سعر: السعر الحقيقي مقابل توقع الموديل المنشور.
بيكتب data/processed/mispriced_listings.csv ويطبع إحصائيات حقيقية لكل مدينة.

Caveat: the deployed model was trained on all priced listings, so scores are in-sample
and the "fair" share is optimistic. Prices outside the training range (12.81-714.33)
are scored too (any price >= 1), so extreme listings will usually be flagged.

Run:  python analysis/build_mispricing.py
"""
from __future__ import annotations

import ast
from datetime import date
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

REPO_ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENT_DIR = REPO_ROOT / "deployment"
sys.path.insert(0, str(DEPLOYMENT_DIR))

from app.features import build_feature_row  # noqa: E402
from app.model_loader import load_bundle  # noqa: E402

DATA_PATH = REPO_ROOT / "data" / "processed" / "listings_final.csv"
OUT_PATH = REPO_ROOT / "data" / "processed" / "mispriced_listings.csv"
MODELS_DIR = DEPLOYMENT_DIR / "models"
PROGRESS_EVERY = 5000


LISTING_KEYS: tuple[str, ...] = (
    "city", "neighbourhood_cleansed", "latitude", "longitude", "room_type",
    "property_type", "accommodates", "bedrooms", "beds", "bathrooms",
    "bathroom_shared", "has_license", "amenities", "host_is_superhost",
    "host_identity_verified", "host_tenure_years", "calculated_host_listings_count",
    "availability_30", "availability_365", "minimum_nights_avg_ntm",
    "number_of_reviews", "number_of_reviews_ltm", "reviews_per_month",
    "review_scores_rating", "review_scores_location", "review_scores_cleanliness",
    "review_scores_value", "first_review", "last_review",
)

OUT_COLUMNS: list[str] = [
    "city", "id", "neighbourhood_cleansed", "room_type", "property_type", "accommodates",
    "latitude", "longitude", "actual_price", "predicted_price", "diff_usd",
    "diff_percent", "flag",
]


def to_listing(rec: dict[str, Any]) -> dict[str, Any]:
    """Turn one CSV record into the raw listing dict the serving pipeline expects."""
    listing = {k: rec[k] for k in LISTING_KEYS}

    raw = listing["amenities"]
    amenities: list[str] = []
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, (list, tuple, set)):
                amenities = [str(a) for a in parsed]
        except (ValueError, SyntaxError):
            amenities = []

    elif isinstance(raw, (list, tuple, set)):
        amenities = [str(a) for a in raw]
    listing["amenities"] = amenities

    # NaN numerics are passed through as-is (pipeline imputes from train_medians).
    # بنحوّلهم لـ date objects زي ما الـ API بيسلّمهم (pydantic)، مش strings.
    for key in ("first_review", "last_review"):
        v = listing[key]
        if pd.isna(v):
            listing[key] = None
        else:
            try:
                listing[key] = date.fromisoformat(str(v)[:10])
            except ValueError:
                listing[key] = None
    return listing


def flag_rows(actual: pd.Series, predicted: pd.Series) -> pd.Series:
    """underpriced if actual <= 0.8*pred, overpriced if actual >= 1.2*pred, else fair."""
    flag = pd.Series("fair", index=actual.index, dtype="object")
    flag[actual <= 0.8 * predicted] = "underpriced"
    flag[actual >= 1.2 * predicted] = "overpriced"
    return flag


def main() -> None:
    bundle = load_bundle(MODELS_DIR)
    cols: list[str] = list(bundle.artifacts["feature_columns"])

    raw = pd.read_csv(DATA_PATH, low_memory=False)
    missing = [c for c in (*LISTING_KEYS, "id", "price") if c not in raw.columns]
    if missing:
        raise KeyError(f"listings_final.csv is missing columns: {missing}")

    df = raw[raw["price"].notna() & (raw["price"] >= 1)].reset_index(drop=True)
    print(f"Priced rows (price >= 1): {len(df):,} of {len(raw):,}")

    # ---- per-row feature building, wrapped in try/except -------------------

    records = df[list(LISTING_KEYS)].to_dict("records")
    values: list[np.ndarray] = []
    kept: list[int] = []
    n_failed = 0
    first_error: str | None = None
    for i, rec in enumerate(records):
        try:
            fr = build_feature_row(to_listing(rec), bundle.artifacts, bundle.landmarks)
            values.append(fr[cols].to_numpy(dtype=float)[0])
            kept.append(i)
        except Exception as exc:  # noqa: BLE001 - skip and count
            n_failed += 1
            if first_error is None:
                first_error = f"{type(exc).__name__}: {exc}"
        if (i + 1) % PROGRESS_EVERY == 0:
            print(f"  processed {i + 1:,}/{len(records):,}", flush=True)
    if not values:
        raise RuntimeError(f"All rows failed. First error: {first_error}")

    X = pd.DataFrame(np.vstack(values), columns=cols)
    pred = np.expm1(bundle.booster.predict(xgb.DMatrix(X)))

    ok = np.isfinite(pred) & (pred > 0)
    n_bad_pred = int((~ok).sum())
    n_failed += n_bad_pred

    scored = df.iloc[kept].reset_index(drop=True)[ok].reset_index(drop=True)
    predicted = pd.Series(pred[ok], index=scored.index)
    actual = scored["price"].astype(float)

    diff = actual - predicted
    out = pd.DataFrame({

        "city": scored["city"],
        "id": scored["id"],
        "neighbourhood_cleansed": scored["neighbourhood_cleansed"],
        "room_type": scored["room_type"],
        "property_type": scored["property_type"],
        "accommodates": scored["accommodates"],
        "latitude": scored["latitude"],
        "longitude": scored["longitude"],
        "actual_price": actual.round(2),
        "predicted_price": predicted.round(2),
        "diff_usd": diff.round(2),
        "diff_percent": (diff / predicted * 100).round(1),
        "flag": flag_rows(actual, predicted),  # flag uses unrounded values
    })[OUT_COLUMNS]

    out.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH} ({len(out):,} rows)")
    print(f"Rows skipped (feature failure or non-positive/non-finite prediction): {n_failed:,}")
    if first_error:
        print(f"First feature error: {first_error}")

    # ---- real summary per city ---------------------------------------------
    counts = (
        out.groupby(["city", "flag"]).size().unstack(fill_value=0)
        .reindex(columns=["underpriced", "fair", "overpriced"], fill_value=0)
    )
    summary = counts.copy()
    summary.insert(0, "n", counts.sum(axis=1))
    summary["median_diff_percent"] = out.groupby("city")["diff_percent"].median()
    print("\n=== Per-city summary ===")
    print(summary.to_string())


    overall = out["flag"].value_counts().reindex(["underpriced", "fair", "overpriced"], fill_value=0)
    print("\n=== All cities ===")
    print(overall.to_string())
    print(f"median diff_percent: {out['diff_percent'].median():.1f}")


if __name__ == "__main__":
    main()