"""
بيحسب موسمية "نسبة الإتاحة" لكل مدينة وشهر من calendar_daily_city.csv.
تنبيه: دي نسبة الوحدات اللي المضيف سايبها متاحة، مش طلب فعلي ولا حجوزات.
"""
# Caveat (English): Inside Airbnb calendars contain no bookings and no prices.
# "available" is set by the host, so share_available is an availability proxy,
# NOT measured demand.

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = REPO_ROOT / "data" / "processed" / "calendar_daily_city.csv"
OUTPUT_PATH = REPO_ROOT / "data" / "processed" / "seasonality.csv"

REQUIRED_COLUMNS = {"city", "date", "n_listings", "n_available", "share_available"}
OUTPUT_COLUMNS = ["city", "month", "mean_share_available", "mean_n_available", "n_days"]


def build_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily city availability into city x calendar-month means."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"calendar_daily_city.csv is missing columns: {sorted(missing)}")

    work = df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.dropna(subset=["date", "share_available", "n_available"])
    work["month"] = work["date"].dt.month


    grouped = (
        work.groupby(["city", "month"], as_index=False)
        .agg(
            mean_share_available=("share_available", "mean"),
            mean_n_available=("n_available", "mean"),
            n_days=("date", "nunique"),
        )
        .sort_values(["city", "month"])
        .reset_index(drop=True)
    )
    grouped["mean_share_available"] = grouped["mean_share_available"].round(4)
    grouped["mean_n_available"] = grouped["mean_n_available"].round(1)
    grouped["n_days"] = grouped["n_days"].astype(int)
    return grouped[OUTPUT_COLUMNS]


def print_preview(season: pd.DataFrame) -> None:
    """Print the busiest/quietest month per city (by availability share)."""
    print("Seasonality preview (availability proxy, not measured demand)")
    for city, part in season.groupby("city"):
        highest = part.loc[part["mean_share_available"].idxmax()]
        lowest = part.loc[part["mean_share_available"].idxmin()]
        print(
            f"  {city}: highest availability month={int(highest['month'])} "
            f"({highest['mean_share_available']:.4f}), "
            f"lowest availability month={int(lowest['month'])} "
            f"({lowest['mean_share_available']:.4f})"
        )


def main() -> None:

    if not INPUT_PATH.exists():
        raise SystemExit(f"Input file not found: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)
    season = build_seasonality(df)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    season.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(season)} rows to {OUTPUT_PATH}")
    print_preview(season)


if __name__ == "__main__":
    main()