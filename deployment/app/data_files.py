"""بيحمّل ملفات الـ CSV بتاعة الداتا lazy ويخزّنها في app.state، ولو الملف ناقص بيرمي DataUnavailable."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


class DataUnavailable(RuntimeError):
    """Raised when a required data file is missing, unreadable or malformed."""


def get_data_dir() -> Path:
    env = os.environ.get("DATA_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / "data"


def load_csv(state: Any, filename: str, required_columns: set[str]) -> pd.DataFrame:
    """Load <DATA_DIR>/<filename> once and cache it on `state` (keyed by full path)."""
    path = get_data_dir() / filename
    cache: dict[str, pd.DataFrame] | None = getattr(state, "csv_cache", None)
    if cache is None:
        cache = {}
        state.csv_cache = cache

    key = str(path)
    if key in cache:

        return cache[key]

    if not path.is_file():
        raise DataUnavailable(
            f"{filename} is not available on this server; "
            "this endpoint stays disabled until the file is generated."
        )
    try:
        df = pd.read_csv(path)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError):
        raise DataUnavailable(f"{filename} exists but could not be read.") from None

    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise DataUnavailable(f"{filename} is missing expected columns: {', '.join(missing)}.")

    cache[key] = df
    return df


def to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """DataFrame -> list of plain-Python dicts (NaN becomes None)."""
    return json.loads(df.to_json(orient="records"))