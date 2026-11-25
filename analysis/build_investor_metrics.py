"""
بيحسب مقاييس المستثمر لكل listing وملخص السوق لكل حي (neighbourhood).
بيكتب investor_metrics.csv و market_summary.csv في data/processed/.

Run:  python analysis/build_investor_metrics.py
Optional: --reuse-predictions  reads predicted_price from mispriced_listings.csv
          (written by build_mispricing.py) instead of re-running the model.
          Listings that file skipped are dropped here too.
"""
from __future__ import annotations

import argparse
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

PROCESSED = REPO_ROOT / "data" / "processed"
DATA_PATH = PROCESSED / "listings_final.csv"
MISPRICED_PATH = PROCESSED / "mispriced_listings.csv"
METRICS_PATH = PROCESSED / "investor_metrics.csv"

SUMMARY_PATH = PROCESSED / "market_summary.csv"
MODELS_DIR = DEPLOYMENT_DIR / "models"

MIN_LISTINGS_PER_NBH = 30
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


def predict_with_deployed_model(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Add predicted_price (USD) using the deployed booster. Returns (scored df, n_failed)."""
    bundle = load_bundle(MODELS_DIR)
    cols: list[str] = list(bundle.artifacts["feature_columns"])
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

    if first_error:
        print(f"First feature error: {first_error}")

    X = pd.DataFrame(np.vstack(values), columns=cols)
    pred = np.expm1(bundle.booster.predict(xgb.DMatrix(X)))
    ok = np.isfinite(pred) & (pred > 0)
    n_failed += int((~ok).sum())

    scored = df.iloc[kept].reset_index(drop=True)[ok].reset_index(drop=True)
    scored["predicted_price"] = pred[ok]
    return scored, n_failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-predictions", action="store_true")
    args = parser.parse_args()

    raw = pd.read_csv(DATA_PATH, low_memory=False)
    needed = [*LISTING_KEYS, "id", "price", "estimated_occupancy_l365d"]
    missing = [c for c in needed if c not in raw.columns]
    if missing:
        raise KeyError(f"listings_final.csv is missing columns: {missing}")

    # Same population as build_mispricing.py: priced listings, price >= 1.
    df = raw[raw["price"].notna() & (raw["price"] >= 1)].reset_index(drop=True)
    print(f"Priced rows (price >= 1): {len(df):,} of {len(raw):,}")

    if args.reuse_predictions:
        mp = pd.read_csv(MISPRICED_PATH, usecols=["city", "id", "predicted_price"])
        scored = df.merge(mp, on=["city", "id"], how="inner")
        n_failed = len(df) - len(scored)

        print(f"Reused predictions from {MISPRICED_PATH.name} (rounded to cents).")
    else:
        scored, n_failed = predict_with_deployed_model(df)
    print(f"Rows skipped (no valid prediction): {n_failed:,}")

    # occupancy_days is Inside Airbnb's OWN model estimate (estimated_occupancy_l365d),
    # not ours. We use it as-is; rows where it is missing get NaN revenue.
    scored["occupancy_days"] = scored["estimated_occupancy_l365d"].astype(float)
    revenue = (scored["predicted_price"] * scored["occupancy_days"]).round(0)
    scored["is_underpriced"] = scored["price"].astype(float) <= 0.8 * scored["predicted_price"]

    # ---- investor_metrics.csv (per listing) --------------------------------
    metrics = pd.DataFrame({
        "city": scored["city"],
        "id": scored["id"],
        "neighbourhood_cleansed": scored["neighbourhood_cleansed"],
        "room_type": scored["room_type"],
        "accommodates": scored["accommodates"],
        "predicted_price": scored["predicted_price"].round(2),
        "occupancy_days": scored["occupancy_days"],
        "expected_annual_revenue": revenue.astype("Int64"),
    })
    metrics.to_csv(METRICS_PATH, index=False)
    print(f"\nWrote {METRICS_PATH} ({len(metrics):,} rows)")

    # ---- market_summary.csv (city x neighbourhood, >= 30 listings) ---------
    work = pd.DataFrame({
        "city": scored["city"],
        "neighbourhood_cleansed": scored["neighbourhood_cleansed"],  # NaN neighbourhoods are excluded by groupby
        "id": scored["id"],
        "actual_price": scored["price"].astype(float),
        "predicted_price": scored["predicted_price"],

        "occupancy_days": scored["occupancy_days"],
        "annual_revenue": revenue,
        "is_underpriced": scored["is_underpriced"],
    })
    summary = (
        work.groupby(["city", "neighbourhood_cleansed"])
        .agg(
            n_listings=("id", "size"),
            median_actual_price=("actual_price", "median"),
            median_predicted_price=("predicted_price", "median"),
            median_occupancy_days=("occupancy_days", "median"),
            median_annual_revenue=("annual_revenue", "median"),
            share_underpriced=("is_underpriced", "mean"),
        )
        .reset_index()
    )
    summary = summary[summary["n_listings"] >= MIN_LISTINGS_PER_NBH].copy()
    summary["median_actual_price"] = summary["median_actual_price"].round(2)
    summary["median_predicted_price"] = summary["median_predicted_price"].round(2)
    summary["median_occupancy_days"] = summary["median_occupancy_days"].round(1)
    summary["median_annual_revenue"] = summary["median_annual_revenue"].round(0)
    summary["share_underpriced"] = summary["share_underpriced"].round(3)
    summary = summary.sort_values(["city", "n_listings"], ascending=[True, False])
    summary.to_csv(SUMMARY_PATH, index=False)
    print(f"Wrote {SUMMARY_PATH} ({len(summary):,} neighbourhoods with >= {MIN_LISTINGS_PER_NBH} listings)")

    # ---- real per-city medians ---------------------------------------------
    per_city = work.groupby("city").agg(
        n_listings=("id", "size"),
        median_actual_price=("actual_price", "median"),
        median_predicted_price=("predicted_price", "median"),
        median_occupancy_days=("occupancy_days", "median"),

        median_annual_revenue=("annual_revenue", "median"),
        share_underpriced=("is_underpriced", "mean"),
        missing_occupancy=("occupancy_days", lambda s: int(s.isna().sum())),
    )
    print("\n=== Per-city medians ===")
    print(per_city.round({
        "median_actual_price": 2, "median_predicted_price": 2,
        "median_occupancy_days": 1, "median_annual_revenue": 0, "share_underpriced": 3,
    }).to_string())


if __name__ == "__main__":
    main()