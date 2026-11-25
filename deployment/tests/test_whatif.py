import pytest

from app.whatif import apply_changes


def test_apply_changes_adds_and_removes_without_mutating():
    original = {"amenities": ["Wifi", "Kitchen"], "bedrooms": 1}
    new = apply_changes(original, {"add_amenities": ["Washer", "wifi"], "remove_amenities": ["kitchen"], "bedrooms": 2})
    assert new["amenities"] == ["Wifi", "Washer"] and new["bedrooms"] == 2
    assert original == {"amenities": ["Wifi", "Kitchen"], "bedrooms": 1}


def test_whatif_endpoint(client, full_listing):
    body = {"listing": full_listing, "scenarios": [
        {"name": "more guests", "changes": {"accommodates": 6}},
        {"name": "no change", "changes": {}},
    ]}
    r = client.post("/whatif", json=body)
    assert r.status_code == 200
    out = r.json()
    base = client.post("/predict", json=full_listing).json()["predicted_price"]
    assert out["base_price"] == pytest.approx(base, abs=0.01)
    assert out["scenarios"][1]["change_usd"] == 0
    assert out["scenarios"][0]["change_usd"] > 0     # الموديل الوهمي بيزوّد السعر مع الضيوف


@pytest.mark.parametrize("bad", [{"city": "Berlin"}, {"accommodates": 0}, {"unknown": 1}])
def test_whatif_rejects_bad_changes(client, full_listing, bad):
    body = {"listing": full_listing, "scenarios": [{"name": "x", "changes": bad}]}
    assert client.post("/whatif", json=body).status_code == 422


def test_whatif_requires_scenarios(client, full_listing):
    assert client.post("/whatif", json={"listing": full_listing, "scenarios": []}).status_code == 422


def test_improve_only_returns_gains_sorted(client, minimal_listing):
    r = client.post("/improve", json=minimal_listing)
    assert r.status_code == 200
    gains = [s["change_usd"] for s in r.json()["suggestions"]]
    assert all(g > 0 for g in gains) and gains == sorted(gains, reverse=True)
    assert len(gains) <= 5


def test_improve_skips_what_already_exists(client, full_listing):
    names = {s["name"] for s in client.post("/improve", json=full_listing).json()["suggestions"]}
    assert "Become a Superhost" not in names and "Add wifi" not in names
