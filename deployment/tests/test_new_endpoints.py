"""تيستات الـ endpoints الجديدة: /mispricing و /market و /invest و /advisor."""
from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from app import advisor as advisor_module
from app.invest import build_investment
from app.main import app
from app.model_loader import predict_price
from app.schemas import ListingInput

MISPRICED_CSV = """city,id,neighbourhood_cleansed,room_type,property_type,accommodates,latitude,longitude,actual_price,predicted_price,diff_usd,diff_percent,flag
Barcelona,1,Eixample,Entire home/apt,Entire rental unit,4,41.39,2.16,80.0,120.0,-40.0,-33.3,underpriced
Barcelona,2,Gracia,Private room,Private room in rental unit,2,41.40,2.15,100.0,90.0,10.0,11.1,overpriced
Berlin,3,Mitte,Entire home/apt,Entire rental unit,3,52.52,13.40,60.0,100.0,-40.0,-40.0,underpriced
Berlin,4,Pankow,Private room,Private room in rental unit,1,52.55,13.41,50.0,50.0,0.0,0.0,fair
Amsterdam,5,Centrum,Entire home/apt,Entire rental unit,2,52.37,4.89,300.0,150.0,150.0,100.0,overpriced
"""

MARKET_CSV = """city,neighbourhood_cleansed,n_listings,median_actual_price,median_predicted_price,median_occupancy_days,median_annual_revenue,share_underpriced
Barcelona,Eixample,100,120.0,118.0,200,24000.0,0.4
Barcelona,Gracia,50,90.0,95.0,100,9000.0,0.3
Barcelona,Sants,10,70.0,72.0,150,10500.0,0.2
Berlin,Mitte,80,100.0,98.0,120,12000.0,0.5
"""

MARKET_NOTE = "Occupancy and revenue are estimates (Inside Airbnb occupancy model + our price model)."

LISTING = {
    "city": "Barcelona",

    "latitude": 41.39,
    "longitude": 2.16,
    "room_type": "Entire home/apt",
    "property_type": "Entire rental unit",
    "accommodates": 4,
}
FAKE_KEY = "test-key-should-never-leak-123"


@pytest.fixture(scope="module")
def api():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def data_dir(tmp_path, monkeypatch):
    (tmp_path / "mispriced_listings.csv").write_text(MISPRICED_CSV, encoding="utf-8")
    (tmp_path / "market_summary.csv").write_text(MARKET_CSV, encoding="utf-8")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture()
def empty_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    return tmp_path


def _expected_price(client: TestClient, listing: dict) -> float:
    bundle = client.app.state.bundle
    return float(predict_price(bundle, ListingInput(**listing).model_dump()))



# ---------------------------------------------------------------- /mispricing


def test_mispricing_sorted_most_underpriced_first(api, data_dir):
    r = api.get("/mispricing")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 5
    assert [x["id"] for x in body["listings"]] == [3, 1, 4, 2, 5]
    assert [x["diff_percent"] for x in body["listings"]] == [-40.0, -33.3, 0.0, 11.1, 100.0]


def test_mispricing_filter_by_city(api, data_dir):
    body = api.get("/mispricing", params={"city": "Barcelona"}).json()
    assert [x["id"] for x in body["listings"]] == [1, 2]
    assert body["count"] == 2


def test_mispricing_filter_by_flag(api, data_dir):
    body = api.get("/mispricing", params={"flag": "underpriced"}).json()
    assert [x["id"] for x in body["listings"]] == [3, 1]


def test_mispricing_filter_city_and_flag(api, data_dir):
    body = api.get("/mispricing", params={"city": "Berlin", "flag": "underpriced"}).json()
    assert [x["id"] for x in body["listings"]] == [3]


def test_mispricing_limit(api, data_dir):
    body = api.get("/mispricing", params={"limit": 2}).json()

    assert body["count"] == 2
    assert [x["id"] for x in body["listings"]] == [3, 1]


@pytest.mark.parametrize(
    "params",
    [{"city": "Paris"}, {"flag": "cheap"}, {"limit": 0}, {"limit": 101}],
)
def test_mispricing_validation_errors(api, data_dir, params):
    assert api.get("/mispricing", params=params).status_code == 422


def test_mispricing_503_when_file_missing(api, empty_data_dir):
    r = api.get("/mispricing")
    assert r.status_code == 503
    assert "mispriced_listings.csv" in r.json()["detail"]


def test_mispricing_503_when_columns_missing(api, empty_data_dir):
    (empty_data_dir / "mispriced_listings.csv").write_text("city,id\nBerlin,1\n", encoding="utf-8")
    r = api.get("/mispricing")
    assert r.status_code == 503
    assert "missing expected columns" in r.json()["detail"]


# -------------------------------------------------------------------- /market


def test_market_all_rows_with_note(api, data_dir):
    r = api.get("/market")
    assert r.status_code == 200
    body = r.json()

    assert body["count"] == 4
    assert body["note"] == MARKET_NOTE
    assert body["rows"][0]["neighbourhood_cleansed"] == "Eixample"


def test_market_filter_by_city(api, data_dir):
    body = api.get("/market", params={"city": "Barcelona"}).json()
    assert body["count"] == 3
    assert {row["city"] for row in body["rows"]} == {"Barcelona"}
    assert body["note"] == MARKET_NOTE


def test_market_invalid_city(api, data_dir):
    assert api.get("/market", params={"city": "Paris"}).status_code == 422


def test_market_503_when_file_missing(api, empty_data_dir):
    r = api.get("/market")
    assert r.status_code == 503
    assert "market_summary.csv" in r.json()["detail"]


# --------------------------------------------------------------------- /invest


def test_build_investment_pure_math():
    out = build_investment(
        price=100.0, occupancy_days=200, purchase_price_usd=400000, annual_costs_usd=4000
    )
    assert out == {
        "annual_revenue_usd": 20000.0,
        "gross_yield_pct": 5.0,

        "net_revenue_usd": 16000.0,
        "net_yield_pct": 4.0,
    }


def test_invest_with_occupancy_override_matches_model_price(api, empty_data_dir):
    # Override given -> market_summary.csv is not needed (data dir is empty on purpose).
    price = _expected_price(api, LISTING)
    payload = {
        **LISTING,
        "purchase_price_usd": 400000,
        "annual_costs_usd": 5000,
        "occupancy_days": 200,
    }
    r = api.post("/invest", json=payload)
    assert r.status_code == 200
    body = r.json()

    revenue = price * 200
    assert body["predicted_price"] == pytest.approx(round(price, 2), abs=0.01)
    assert body["occupancy_days"] == 200
    assert body["purchase_price_usd"] == 400000
    assert body["annual_costs_usd"] == 5000
    assert body["annual_revenue_usd"] == pytest.approx(round(revenue, 2), abs=0.01)
    assert body["gross_yield_pct"] == pytest.approx(round(100 * revenue / 400000, 1), abs=0.05)
    assert body["net_revenue_usd"] == pytest.approx(round(revenue - 5000, 2), abs=0.01)
    assert body["net_yield_pct"] == pytest.approx(
        round(100 * (revenue - 5000) / 400000, 1), abs=0.05
    )
    assert any("supplied by the caller (override)" in a for a in body["assumptions"])



def test_invest_default_costs_zero_net_equals_gross(api, empty_data_dir):
    payload = {**LISTING, "purchase_price_usd": 300000, "occupancy_days": 180}
    body = api.post("/invest", json=payload).json()
    assert body["annual_costs_usd"] == 0
    assert body["net_revenue_usd"] == body["annual_revenue_usd"]
    assert body["net_yield_pct"] == body["gross_yield_pct"]
    assert any("assumed 0 USD" in a for a in body["assumptions"])


def test_invest_zero_occupancy_override_is_respected(api, empty_data_dir):
    payload = {**LISTING, "purchase_price_usd": 300000, "occupancy_days": 0}
    r = api.post("/invest", json=payload)
    assert r.status_code == 200
    assert r.json()["annual_revenue_usd"] == 0
    assert r.json()["gross_yield_pct"] == 0


def test_invest_uses_city_median_occupancy_from_market_summary(api, data_dir):
    price = _expected_price(api, LISTING)
    r = api.post("/invest", json={**LISTING, "purchase_price_usd": 400000})
    assert r.status_code == 200
    body = r.json()
    # Barcelona neighbourhood medians: 200, 100, 150 -> median 150
    assert body["occupancy_days"] == 150
    assert body["annual_revenue_usd"] == pytest.approx(round(price * 150, 2), abs=0.01)
    occupancy_notes = [a for a in body["assumptions"] if a.startswith("occupancy:")]
    assert len(occupancy_notes) == 1
    assert "Barcelona" in occupancy_notes[0]
    assert "market_summary.csv" in occupancy_notes[0]
    assert "Inside Airbnb estimate" in occupancy_notes[0]



def test_invest_503_without_override_when_csv_missing(api, empty_data_dir):
    r = api.post("/invest", json={**LISTING, "purchase_price_usd": 400000})
    assert r.status_code == 503
    assert "market_summary.csv" in r.json()["detail"]


def test_invest_503_when_city_has_no_market_rows(api, data_dir):
    columbus = {**LISTING, "city": "Columbus", "latitude": 39.96, "longitude": -82.99}
    r = api.post("/invest", json={**columbus, "purchase_price_usd": 250000})
    assert r.status_code == 503
    assert "Columbus" in r.json()["detail"]


@pytest.mark.parametrize(
    "extra",
    [
        {"purchase_price_usd": 0},
        {"purchase_price_usd": -5},
        {"purchase_price_usd": 400000, "annual_costs_usd": -1},
        {"purchase_price_usd": 400000, "occupancy_days": 366},
        {"purchase_price_usd": 400000, "occupancy_days": -1},
        {},  # purchase_price_usd missing
    ],
)
def test_invest_validation_errors(api, empty_data_dir, extra):
    assert api.post("/invest", json={**LISTING, **extra}).status_code == 422


# -------------------------------------------------------------------- /advisor


def _gemini_ok(text: str = "The price is driven mainly by location.") -> httpx.Response:

    return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": text}]}}]})


def _install_fake_post(monkeypatch, response=None, exc=None):
    calls: list[dict] = []

    def fake_post(url, **kwargs):
        calls.append({"url": url, **kwargs})
        if exc is not None:
            raise exc
        return response

    monkeypatch.setattr(advisor_module.httpx, "post", fake_post)
    return calls


def test_advisor_503_when_key_missing_and_nothing_computed(api, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    def boom(*args, **kwargs):
        raise AssertionError("must not run before the API key check")

    monkeypatch.setattr(advisor_module, "predict_price", boom)
    monkeypatch.setattr(advisor_module, "explain_price", boom)
    monkeypatch.setattr(advisor_module.httpx, "post", boom)

    r = api.post("/advisor", json={"listing": LISTING})
    assert r.status_code == 503
    assert r.json() == {"detail": "Advisor is disabled: GEMINI_API_KEY is not set"}


def test_advisor_success_sends_honesty_instruction_and_real_price(api, monkeypatch):

    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    calls = _install_fake_post(monkeypatch, response=_gemini_ok())
    expected = _expected_price(api, LISTING)

    r = api.post("/advisor", json={"listing": LISTING, "question": "Why this price?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "The price is driven mainly by location."
    assert body["used_key"] is True
    assert body["predicted_price"] == pytest.approx(round(expected, 2), abs=0.01)
    assert FAKE_KEY not in r.text

    assert len(calls) == 1
    call = calls[0]
    assert call["url"] == (
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent"
    )
    assert call["headers"]["x-goog-api-key"] == FAKE_KEY
    assert call["timeout"] == 60.0

    sent = call["json"]
    system_text = sent["system_instruction"]["parts"][0]["text"]
    assert system_text == advisor_module.SYSTEM_INSTRUCTION
    assert "Use ONLY the numbers provided below" in system_text
    assert "Never invent prices, revenues, occupancy or market stats" in system_text

    user_text = sent["contents"][0]["parts"][0]["text"]
    assert f"{expected:.2f}" in user_text
    assert "Why this price?" in user_text
    assert "top_factors" in user_text



def test_advisor_uses_model_from_env(api, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test-model")
    calls = _install_fake_post(monkeypatch, response=_gemini_ok())
    assert api.post("/advisor", json={"listing": LISTING}).status_code == 200
    assert calls[0]["url"].endswith("/models/gemini-test-model:generateContent")


def test_advisor_maps_gemini_http_error_to_502(api, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    _install_fake_post(monkeypatch, response=httpx.Response(429, json={"error": "quota"}))
    r = api.post("/advisor", json={"listing": LISTING})
    assert r.status_code == 502
    assert r.json()["detail"] == "Gemini returned HTTP 429"
    assert FAKE_KEY not in r.text


def test_advisor_maps_timeout_to_502(api, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    _install_fake_post(monkeypatch, exc=httpx.TimeoutException("slow"))
    r = api.post("/advisor", json={"listing": LISTING})
    assert r.status_code == 502
    assert FAKE_KEY not in r.text


def test_advisor_maps_empty_candidates_to_502(api, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    _install_fake_post(monkeypatch, response=httpx.Response(200, json={"candidates": []}))
    assert api.post("/advisor", json={"listing": LISTING}).status_code == 502



def test_advisor_validation_error(api, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    assert api.post("/advisor", json={"listing": {"city": "Paris"}}).status_code == 422