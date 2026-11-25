import pytest
from fastapi.testclient import TestClient

from app.main import app
from models.make_dummy_model import build_dummy_bundle


@pytest.fixture
def model_dir(tmp_path):
    """حزمة موديل وهمية في مجلد مؤقت (مش بنلمس models/ الحقيقي)."""
    return build_dummy_bundle(tmp_path)


@pytest.fixture
def client(model_dir, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    # with بيشغّل lifespan (تحميل الموديل) زي السيرفر الحقيقي
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def minimal_listing():
    """الحقول الإجبارية فقط."""
    return {
        "city": "Barcelona",
        "latitude": 41.3951,
        "longitude": 2.1650,
        "room_type": "Entire home/apt",
        "property_type": "Entire rental unit",
        "accommodates": 4,
    }


@pytest.fixture
def full_listing(minimal_listing):
    return {
        **minimal_listing,
        "neighbourhood_cleansed": "Gracia",
        "bedrooms": 2, "beds": 3, "bathrooms": 1,
        "amenities": ["Wifi", "Kitchen", "Air conditioning"],
        "host_is_superhost": True, "host_identity_verified": True,
        "host_tenure_years": 5.5, "calculated_host_listings_count": 2,
        "availability_30": 10, "availability_365": 200,
        "minimum_nights_avg_ntm": 3, "number_of_reviews": 40,
        "number_of_reviews_ltm": 12, "reviews_per_month": 1.5,
        "review_scores_rating": 4.8, "review_scores_location": 4.9,
        "review_scores_cleanliness": 4.7, "review_scores_value": 4.6,
        "first_review": "2020-01-10", "last_review": "2025-08-01",
    }
