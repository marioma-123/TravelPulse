"""بيرجّع الليستينجز اللي سعرها غلط (أرخص أو أغلى من توقّع الموديل) من mispriced_listings.csv."""
from __future__ import annotations

from typing import Any, Optional

from app.data_files import load_csv, to_records

FILENAME = "mispriced_listings.csv"
FLAGS = ("underpriced", "overpriced", "fair")
COLUMNS = [
    "city",
    "id",
    "neighbourhood_cleansed",
    "room_type",
    "property_type",
    "accommodates",
    "latitude",
    "longitude",
    "actual_price",
    "predicted_price",
    "diff_usd",
    "diff_percent",
    "flag",
]


def query_mispriced(
    state: Any, city: Optional[str], flag: Optional[str], limit: int
) -> list[dict[str, Any]]:
    """Filter by city/flag, sort by diff_percent ascending (most underpriced first), cut to `limit`."""
    df = load_csv(state, FILENAME, set(COLUMNS))
    if city is not None:

        df = df[df["city"] == city]
    if flag is not None:
        df = df[df["flag"] == flag]
    df = df.sort_values("diff_percent", ascending=True, kind="stable").head(limit)
    return to_records(df[COLUMNS])