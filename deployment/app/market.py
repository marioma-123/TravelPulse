"""ملخص السوق لكل مدينة وحي من market_summary.csv، ومنه بنجيب متوسط الإشغال لـ /invest."""
from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from app.data_files import DataUnavailable, load_csv, to_records

FILENAME = "market_summary.csv"
COLUMNS = [
    "city",
    "neighbourhood_cleansed",
    "n_listings",
    "median_actual_price",
    "median_predicted_price",
    "median_occupancy_days",
    "median_annual_revenue",
    "share_underpriced",
]
NOTE = "Occupancy and revenue are estimates (Inside Airbnb occupancy model + our price model)."


def query_market(state: Any, city: Optional[str]) -> list[dict[str, Any]]:
    df = load_csv(state, FILENAME, set(COLUMNS))
    if city is not None:
        df = df[df["city"] == city]
    return to_records(df[COLUMNS])


def city_median_occupancy(state: Any, city: str) -> tuple[float, int]:
    """Median of the neighbourhood-level `median_occupancy_days` for a city.


    Returns (value, number_of_neighbourhoods_used). The file only holds per-neighbourhood
    medians, so this is a median of medians, not a true city-wide median.
    """
    df = load_csv(state, FILENAME, set(COLUMNS))
    values = pd.to_numeric(
        df.loc[df["city"] == city, "median_occupancy_days"], errors="coerce"
    ).dropna()
    if values.empty:
        raise DataUnavailable(f"No occupancy data for {city} in {FILENAME}.")
    return float(values.median()), int(len(values))