"""
بيعمل calibration residuals ويحسب الـ conformal quantiles عشان /predict-range يشتغل.
بيعيد تدريب الموديل على split host-disjoint ويكتب calibration.csv وبيضيف "conformal" في artifacts.json.

NOTE (honesty): the deployed model was trained on ALL the data, so these residuals come
from a slightly weaker refit (trained on ~68% of priced rows). The resulting intervals are
therefore conservative for the deployed model, not exact.

Run from the repo root or from deployment/:
    python deployment/tools/build_calibration.py
"""
from __future__ import annotations

import ast
from datetime import date
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

DEPLOYMENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = DEPLOYMENT_DIR.parent
sys.path.insert(0, str(DEPLOYMENT_DIR))

from app.features import build_feature_row  # noqa: E402
from app.model_loader import load_bundle  # noqa: E402


DATA_PATH = REPO_ROOT / "data" / "processed" / "listings_final.csv"
MODELS_DIR = DEPLOYMENT_DIR / "models"
ARTIFACTS_PATH = MODELS_DIR / "artifacts.json"
CALIBRATION_PATH = MODELS_DIR / "calibration.csv"

LEVELS: tuple[float, float] = (0.8, 0.9)
SEED = 42
PROGRESS_EVERY = 2000

# Raw fields that build_feature_row expects (from the project spec).
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


def build_matrix(
    df: pd.DataFrame, artifacts: dict[str, Any], landmarks: Any
) -> tuple[pd.DataFrame, int]:
    """Build serving-identical feature rows. Returns (X indexed like df, n_failed)."""
    cols: list[str] = list(artifacts["feature_columns"])
    records = df[list(LISTING_KEYS)].to_dict("records")
    index = df.index.to_list()

    values: list[np.ndarray] = []
    kept: list[int] = []
    n_failed = 0
    first_error: str | None = None

    for i, (idx, rec) in enumerate(zip(index, records), start=1):
        try:
            fr = build_feature_row(to_listing(rec), artifacts, landmarks)
            values.append(fr[cols].to_numpy(dtype=float)[0])
            kept.append(idx)

        except Exception as exc:  # noqa: BLE001 - skip and count, report below
            n_failed += 1
            if first_error is None:
                first_error = f"{type(exc).__name__}: {exc}"
        if i % PROGRESS_EVERY == 0:
            print(f"  built {i:,}/{len(records):,} feature rows", flush=True)

    if not values:
        raise RuntimeError(f"All feature rows failed. First error: {first_error}")
    if n_failed:
        print(f"  WARNING: {n_failed:,} rows failed feature building. First error: {first_error}")

    X = pd.DataFrame(np.vstack(values), columns=cols, index=kept)
    return X, n_failed


def conformal_quantile(abs_res: np.ndarray, level: float) -> float:
    """Split-conformal quantile with finite-sample correction k=min(ceil((n+1)*level), n)."""
    n = abs_res.size
    k = min(math.ceil(round((n + 1) * level, 9)), n)
    return float(np.sort(abs_res)[k - 1])


def split_half_coverage(
    abs_res: np.ndarray, groups: np.ndarray, levels: tuple[float, ...], reps: int = 20
) -> dict[float, float]:
    """Out-of-sample sanity check: quantile from one host-disjoint half, coverage on the other."""
    gss = GroupShuffleSplit(n_splits=reps, test_size=0.5, random_state=SEED)
    cover: dict[float, list[float]] = {lv: [] for lv in levels}
    for a, b in gss.split(abs_res.reshape(-1, 1), groups=groups):
        for lv in levels:
            q = conformal_quantile(abs_res[a], lv)

            cover[lv].append(float(np.mean(abs_res[b] <= q)))
    return {lv: float(np.mean(v)) for lv, v in cover.items()}


def main() -> None:
    bundle = load_bundle(MODELS_DIR)
    artifacts = bundle.artifacts
    # Range is read from artifacts.json (same values as the training notebook: 12.81-714.33)
    # instead of being hardcoded, so it cannot drift from the deployed model.
    lo, hi = (float(x) for x in artifacts["train_price_range"])

    # ---- 1. Load + filter -------------------------------------------------
    raw = pd.read_csv(DATA_PATH, low_memory=False)
    missing = [c for c in (*LISTING_KEYS, "id", "host_id", "price") if c not in raw.columns]
    if missing:
        raise KeyError(f"listings_final.csv is missing columns: {missing}")

    n_total = len(raw)
    price_null = raw["price"].isna()
    below = (~price_null) & (raw["price"] < lo)
    above = (~price_null) & (raw["price"] > hi)
    df = raw[~price_null & ~below & ~above].reset_index(drop=True)
    print(f"Rows total: {n_total:,}")
    print(f"  dropped (price null): {int(price_null.sum()):,}")
    print(f"  dropped (price < {lo}): {int(below.sum()):,}")
    print(f"  dropped (price > {hi}): {int(above.sum()):,}")
    print(f"  kept: {len(df):,}")

    # Rows without host_id become their own group (never dropped, never merged together).
    groups = df["host_id"].where(df["host_id"].notna(), -df["id"]).astype("int64")

    # ---- 2. Host-disjoint split ------------------------------------------

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_pos, test_pos = next(gss.split(df, groups=groups))
    train_ids = df.index[train_pos]
    test_ids = df.index[test_pos]
    assert set(groups.loc[train_ids]).isdisjoint(set(groups.loc[test_ids]))

    # Validation slice (15% of train hosts) for early stopping.
    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    fit_pos, val_pos = next(gss_val.split(train_ids.to_numpy(), groups=groups.loc[train_ids]))
    fit_ids = train_ids[fit_pos]
    val_ids = train_ids[val_pos]
    assert set(groups.loc[fit_ids]).isdisjoint(set(groups.loc[val_ids]))
    print(f"Split -> fit {len(fit_ids):,} | val {len(val_ids):,} | test {len(test_ids):,}")

    # ---- 3. Features (same pipeline as serving) ---------------------------
    print("Building feature rows (slow, one row at a time)...")
    X, n_failed = build_matrix(df, artifacts, bundle.landmarks)
    y_log = np.log1p(df["price"].astype(float))

    fit_ok = X.index.intersection(fit_ids)
    val_ok = X.index.intersection(val_ids)
    test_ok = X.index.intersection(test_ids)
    print(f"Usable rows -> fit {len(fit_ok):,} | val {len(val_ok):,} | test {len(test_ok):,} "
          f"(feature failures: {n_failed:,})")

    # ---- 4. Refit with the notebook's hyperparameters ---------------------
    model = xgb.XGBRegressor(
        n_estimators=3000,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,

        colsample_bytree=0.8,
        tree_method="hist",
        random_state=SEED,
        early_stopping_rounds=50,
    )
    model.fit(
        X.loc[fit_ok], y_log.loc[fit_ok],
        eval_set=[(X.loc[val_ok], y_log.loc[val_ok])],
        verbose=False,
    )
    best_iter = int(model.best_iteration)
    print(f"Refit done. best_iteration={best_iter}")

    # ---- 5. Held-out predictions + calibration.csv ------------------------
    pred_log = model.predict(X.loc[test_ok], iteration_range=(0, best_iter + 1))
    true_log = y_log.loc[test_ok].to_numpy()
    calib = pd.DataFrame({"y_true_log": true_log, "y_pred_log": pred_log})
    calib.to_csv(CALIBRATION_PATH, index=False)
    print(f"Wrote {CALIBRATION_PATH} ({len(calib):,} rows)")

    # ---- 6. Split-conformal quantiles in LOG space ------------------------
    abs_res = np.abs(true_log - pred_log)
    n = int(abs_res.size)
    quantiles = {f"{lv}": conformal_quantile(abs_res, lv) for lv in LEVELS}

    # Re-read artifacts.json from disk so every existing key is preserved untouched.
    art_raw = json.loads(ARTIFACTS_PATH.read_text(encoding="utf-8"))
    had_old = "conformal" in art_raw
    art_raw["conformal"] = {"n_calibration": n, "quantiles": quantiles}
    ARTIFACTS_PATH.write_text(
        json.dumps(art_raw, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"Wrote 'conformal' block to {ARTIFACTS_PATH}" + (" (replaced old block)" if had_old else ""))

    # ---- 7. Honest report -------------------------------------------------
    y_true_usd = np.expm1(true_log)
    y_pred_usd = np.expm1(pred_log)
    mae = mean_absolute_error(y_true_usd, y_pred_usd)
    rmse = float(np.sqrt(mean_squared_error(y_true_usd, y_pred_usd)))
    r2 = r2_score(y_true_usd, y_pred_usd)
    print("\n=== Refit model, held-out test set (USD space) ===")
    print(f"n_test={n:,}  MAE=${mae:.2f}  RMSE=${rmse:.2f}  R2={r2:.3f}")

    print("\n=== Conformal quantiles (log space) ===")
    for lv in LEVELS:
        q = quantiles[f"{lv}"]
        print(f"level {lv}: q={q:.4f}  -> multiplicative band x[{math.exp(-q):.3f}, {math.exp(q):.3f}]")

    print("\n=== Empirical coverage ===")
    for lv in LEVELS:
        q = quantiles[f"{lv}"]
        cov = float(np.mean(abs_res <= q))
        print(f"level {lv}: coverage on the calibration/held-out set itself = {cov:.3f} "
              f"(by construction >= level; NOT an independent test)")
    oos = split_half_coverage(abs_res, groups.loc[test_ok].to_numpy(), LEVELS)
    for lv in LEVELS:
        print(f"level {lv}: out-of-sample coverage (host-disjoint half/half, 20 reps) = {oos[lv]:.3f}")
    print("\nNote: the deployed model saw all data; these residuals come from a weaker refit, "
          "so the intervals are conservative for it.")


if __name__ == "__main__":
    main()