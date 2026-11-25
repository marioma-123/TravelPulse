"""
بيرجّع موسمية نسبة الإتاحة لكل مدينة وشهر من seasonality.csv.
الأرقام دي إتاحة من المضيفين، مش طلب أو حجوزات فعلية.
"""
# Caveat (English): share_available comes from host-set Inside Airbnb calendars;
# it is an availability proxy, not measured demand or bookings.

from __future__ import annotations

from typing import Any, Optional

from app.data_files import DataUnavailable, load_csv, to_records  # noqa: F401  (DataUnavailable propagates)

CITIES: tuple[str, ...] = ("Amsterdam", "Barcelona", "Berlin", "Columbus")
SEASONALITY_FILE = "seasonality.csv"
REQUIRED_COLUMNS = {"city", "month", "mean_share_available", "mean_n_available", "n_days"}
NOTE = (
    "Availability share is the share of listings marked available by hosts "
    "(Inside Airbnb calendar). It is not measured demand or bookings."
)


def get_seasonality(state: Any, city: Optional[str] = None) -> dict:
    """Return seasonality rows, optionally filtered to one city.

    Raises ValueError for an unknown city filter and lets DataUnavailable
    propagate (main.py maps it to 503).
    """
    if city is not None and city not in CITIES:
        raise ValueError(f"city must be one of {list(CITIES)}")

    df = load_csv(state, SEASONALITY_FILE, REQUIRED_COLUMNS)


    # Keep only the 4 supported cities, then apply the optional filter.
    df = df[df["city"].isin(CITIES)]
    if city is not None:
        df = df[df["city"] == city]

    df = df.sort_values(["city", "month"]).reset_index(drop=True)
    return {"rows": to_records(df), "note": NOTE}