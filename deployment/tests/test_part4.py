"""
اختبارات الجزء الرابع: موسمية الإتاحة + تحويل الوصف الحر لـ ListingInput.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from app import data_files, parse_listing, seasonality
from app.advisor import AdvisorError
from app.data_files import DataUnavailable

VALID_JSON = (
    '{"city":"Berlin","latitude":52.52,"longitude":13.405,'
    '"room_type":"Entire home/apt","property_type":"Entire rental unit",'
    '"accommodates":2,"amenities":["Wifi","Kitchen"]}'
)


@pytest.fixture()
def state():
    # NOTE: adjust if tests/test_new_endpoints.py builds `state` differently.
    return SimpleNamespace()


@pytest.fixture()
def data_dir(tmp_path, monkeypatch):
    # Same monkeypatched DATA_DIR pattern as tests/test_new_endpoints.py.
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    return tmp_path


def _write_seasonality(folder) -> None:
    # Tiny synthetic fixture, test data only.
    pd.DataFrame(
        [
            {"city": "Berlin", "month": 1, "mean_share_available": 0.5, "mean_n_available": 100.0, "n_days": 31},
            {"city": "Berlin", "month": 2, "mean_share_available": 0.6, "mean_n_available": 120.0, "n_days": 28},
            {"city": "Columbus", "month": 1, "mean_share_available": 0.4, "mean_n_available": 80.0, "n_days": 31},
        ]
    ).to_csv(folder / "seasonality.csv", index=False)


# ---------- get_seasonality ----------

def test_seasonality_all_rows_and_note(state, data_dir):
    _write_seasonality(data_dir)
    out = seasonality.get_seasonality(state, None)
    assert len(out["rows"]) == 3
    assert {r["city"] for r in out["rows"]} == {"Berlin", "Columbus"}
    assert out["note"] == (
        "Availability share is the share of listings marked available by hosts "
        "(Inside Airbnb calendar). It is not measured demand or bookings."
    )


def test_seasonality_city_filter(state, data_dir):
    _write_seasonality(data_dir)
    out = seasonality.get_seasonality(state, "Berlin")
    assert [r["month"] for r in out["rows"]] == [1, 2]
    assert all(r["city"] == "Berlin" for r in out["rows"])



def test_seasonality_unknown_city_rejected(state, data_dir):
    _write_seasonality(data_dir)
    with pytest.raises(ValueError):
        seasonality.get_seasonality(state, "Paris")


def test_seasonality_missing_file(state, data_dir):
    with pytest.raises(DataUnavailable):
        seasonality.get_seasonality(state, None)


# ---------- parse_listing_text ----------

def test_parse_listing_no_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(AdvisorError, match="GEMINI_API_KEY is not set"):
        parse_listing.parse_listing_text("studio in Berlin for two")


def test_parse_listing_success(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(parse_listing, "call_gemini", lambda *a, **k: VALID_JSON)
    out = parse_listing.parse_listing_text("شقة كاملة في برلين لشخصين")
    listing = out["listing"]
    assert listing["city"] == "Berlin"
    assert listing["latitude"] == 52.52
    assert listing["longitude"] == 13.405
    assert listing["room_type"] == "Entire home/apt"
    assert listing["property_type"] == "Entire rental unit"
    assert listing["accommodates"] == 2

    assert listing["amenities"] == ["Wifi", "Kitchen"]
    # exclude_none=True: unstated optional fields must not appear as None.
    assert None not in listing.values()


def test_parse_listing_strips_markdown_fences(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    fenced = f"```json\n{VALID_JSON}\n```"
    monkeypatch.setattr(parse_listing, "call_gemini", lambda *a, **k: fenced)
    assert parse_listing.parse_listing_text("x")["listing"]["city"] == "Berlin"


def test_parse_listing_garbage(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(parse_listing, "call_gemini", lambda *a, **k: "not json")
    with pytest.raises(AdvisorError, match="Could not parse the model's answer as JSON"):
        parse_listing.parse_listing_text("x")


def test_parse_listing_invalid_listing(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    bad = '{"city":"Paris","room_type":"Entire home/apt","property_type":"Flat","accommodates":2,"latitude":1,"longitude":2}'
    monkeypatch.setattr(parse_listing, "call_gemini", lambda *a, **k: bad)
    with pytest.raises(AdvisorError, match="not a valid listing"):
        parse_listing.parse_listing_text("x")

def test_parse_listing_missing_coords_uses_city_center(monkeypatch):
    from app.config import CITY_CENTERS

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    no_coords = (
        '{"city":"Berlin","room_type":"Entire home/apt",'
        '"property_type":"Entire rental unit","accommodates":4}'
    )
    monkeypatch.setattr(parse_listing, "call_gemini", lambda *a, **k: no_coords)
    listing = parse_listing.parse_listing_text("x")["listing"]
    assert (listing["latitude"], listing["longitude"]) == CITY_CENTERS["Berlin"]


def test_parse_listing_missing_coords_and_city(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    bad = '{"room_type":"Entire home/apt","property_type":"Flat","accommodates":2}'
    monkeypatch.setattr(parse_listing, "call_gemini", lambda *a, **k: bad)
    with pytest.raises(AdvisorError, match="Could not find a city"):
        parse_listing.parse_listing_text("x")
