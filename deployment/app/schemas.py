"""شكل البيانات الداخلة والخارجة من الـ API (العقد بين المستخدم والخدمة)."""
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import CITY_CENTERS, MAX_DEGREES_FROM_CENTER

City = Literal["Amsterdam", "Barcelona", "Berlin", "Columbus"]
RoomType = Literal["Entire home/apt", "Private room", "Shared room", "Hotel room"]
FlagFilter = Literal["underpriced", "overpriced", "fair"]


class ListingInput(BaseModel):
    """وحدة شغالة بالفعل (ليها تاريخ). الحقول الاختيارية لو اتسابت فاضية بنملاها
    بنفس طريقة الـ notebook (median من بيانات التدريب)، ودقة التنبؤ بتقل."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "city": "Barcelona",
                "neighbourhood_cleansed": "la Dreta de l'Eixample",
                "latitude": 41.3951,
                "longitude": 2.1650,
                "room_type": "Entire home/apt",
                "property_type": "Entire rental unit",
                "accommodates": 4,
                "bedrooms": 2,
                "beds": 3,
                "bathrooms": 1,
                "amenities": ["Wifi", "Kitchen", "Air conditioning", "Washer", "TV"],

                "host_is_superhost": True,
                "host_identity_verified": True,
                "host_tenure_years": 6.5,
                "calculated_host_listings_count": 3,
                "availability_30": 12,
                "availability_365": 210,
                "minimum_nights_avg_ntm": 3,
                "number_of_reviews": 85,
                "number_of_reviews_ltm": 20,
                "reviews_per_month": 1.8,
                "review_scores_rating": 4.8,
                "review_scores_location": 4.9,
                "review_scores_cleanliness": 4.7,
                "review_scores_value": 4.6,
                "first_review": "2019-04-02",
                "last_review": "2025-08-21",
            }
        },
    )

    # ---- إجباري ----
    city: City
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    room_type: RoomType
    property_type: str = Field(min_length=1, max_length=100)
    accommodates: int = Field(ge=1, le=50)

    # ---- اختياري: المكان والمواصفات ----
    neighbourhood_cleansed: str | None = Field(default=None, max_length=200)
    bedrooms: float | None = Field(default=None, ge=0, le=50)
    beds: float | None = Field(default=None, ge=0, le=100)

    bathrooms: float | None = Field(default=None, ge=0, le=50)
    bathroom_shared: bool = False
    has_license: bool = False
    amenities: list[str] = Field(default_factory=list, max_length=500)

    # ---- اختياري: المضيف ----
    host_is_superhost: bool = False
    host_identity_verified: bool = False
    host_tenure_years: float | None = Field(default=None, ge=0, le=60)
    calculated_host_listings_count: int | None = Field(default=None, ge=0, le=10000)

    # ---- اختياري: التوفر والنشاط ----
    availability_30: int | None = Field(default=None, ge=0, le=31)
    availability_365: int | None = Field(default=None, ge=0, le=366)
    minimum_nights_avg_ntm: float | None = Field(default=None, ge=0, le=5000)
    number_of_reviews: int | None = Field(default=None, ge=0)
    number_of_reviews_ltm: int | None = Field(default=None, ge=0)
    reviews_per_month: float | None = Field(default=None, ge=0, le=500)
    first_review: date | None = None
    last_review: date | None = None
    # التاريخ اللي بنحسب منه "عدد الأيام". لو فاضي بنستخدم تاريخ بيانات التدريب.
    as_of_date: date | None = None

    # ---- اختياري: التقييمات (مقياس من 0 لـ 5) ----
    review_scores_rating: float | None = Field(default=None, ge=0, le=5)
    review_scores_location: float | None = Field(default=None, ge=0, le=5)
    review_scores_cleanliness: float | None = Field(default=None, ge=0, le=5)
    review_scores_value: float | None = Field(default=None, ge=0, le=5)

    @model_validator(mode="after")
    def coordinates_match_city(self):
        c_lat, c_lon = CITY_CENTERS[self.city]

        if (
            abs(self.latitude - c_lat) > MAX_DEGREES_FROM_CENTER
            or abs(self.longitude - c_lon) > MAX_DEGREES_FROM_CENTER
        ):
            raise ValueError(
                f"latitude/longitude are too far from the centre of {self.city}"
            )
        return self


class PredictionOutput(BaseModel):
    city: City
    predicted_price: float
    currency: Literal["USD"] = "USD"  # كل الأسعار في بيانات المشروع بالدولار


class HealthOutput(BaseModel):
    status: str
    model_loaded: bool


class GroupEffect(BaseModel):
    name: str
    effect_percent: float  # الأثر على السعر بالنسبة المئوية (موجب = زوّد، سالب = نقّص)


class FactorEffect(GroupEffect):
    value: float | str  # قيمة العامل في الوحدة (رقم أو نص زي اسم المدينة)


class ExplanationOutput(BaseModel):
    city: City

    predicted_price: float
    currency: Literal["USD"] = "USD"
    baseline_price: float  # سعر "الوحدة المتوسطة" اللي الموديل بيبدأ منها
    groups: list[GroupEffect]
    top_factors: list[FactorEffect]


class RangeOutput(BaseModel):
    city: City
    predicted_price: float
    lower: float
    upper: float
    confidence: float  # نسبة التغطية المستهدفة، مثلاً 0.8
    currency: Literal["USD"] = "USD"


class WhatIfChanges(BaseModel):
    """التغييرات المسموحة بس (قائمة بيضاء): مواصفات تقدر تتغيّر فعلاً."""
    model_config = ConfigDict(extra="forbid")

    accommodates: int | None = Field(default=None, ge=1, le=50)
    bedrooms: float | None = Field(default=None, ge=0, le=50)
    beds: float | None = Field(default=None, ge=0, le=100)
    bathrooms: float | None = Field(default=None, ge=0, le=50)
    bathroom_shared: bool | None = None
    host_is_superhost: bool | None = None
    host_identity_verified: bool | None = None
    has_license: bool | None = None
    availability_365: int | None = Field(default=None, ge=0, le=366)
    minimum_nights_avg_ntm: float | None = Field(default=None, ge=0, le=5000)
    add_amenities: list[str] | None = Field(default=None, max_length=20)
    remove_amenities: list[str] | None = Field(default=None, max_length=20)



class Scenario(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    changes: WhatIfChanges


class WhatIfRequest(BaseModel):
    listing: ListingInput
    scenarios: list[Scenario] = Field(min_length=1, max_length=10)


class ScenarioResult(BaseModel):
    name: str
    predicted_price: float | None = None
    change_usd: float | None = None
    change_percent: float | None = None
    error: str | None = None


class WhatIfOutput(BaseModel):
    base_price: float
    scenarios: list[ScenarioResult]
    currency: Literal["USD"] = "USD"
    note: str = "Model-estimated association, not a guaranteed effect of making the change."


class Suggestion(BaseModel):
    name: str
    predicted_price: float
    change_usd: float
    change_percent: float



class ImproveOutput(BaseModel):
    base_price: float
    suggestions: list[Suggestion]
    currency: Literal["USD"] = "USD"
    note: str = "Model-estimated association, not a guaranteed effect of making the change."


# ======================= الإضافات الجديدة =======================


class MispricedListing(BaseModel):
    """صف من mispriced_listings.csv (السعر الفعلي مقابل توقّع الموديل)."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "city": "Barcelona",
                "id": 123456,
                "neighbourhood_cleansed": "Eixample",
                "room_type": "Entire home/apt",
                "property_type": "Entire rental unit",
                "accommodates": 4,
                "latitude": 41.39,
                "longitude": 2.16,
                "actual_price": 80.0,
                "predicted_price": 120.0,
                "diff_usd": -40.0,
                "diff_percent": -33.3,
                "flag": "underpriced",
            }

        }
    )

    city: str
    id: int | str
    neighbourhood_cleansed: str | None = None
    room_type: str | None = None
    property_type: str | None = None
    accommodates: int | None = None
    latitude: float | None = None
    longitude: float | None = None
    actual_price: float | None = None
    predicted_price: float | None = None
    diff_usd: float | None = None
    diff_percent: float | None = None
    flag: str | None = None


class MispricingResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "count": 1,
                "listings": [
                    {
                        "city": "Barcelona",
                        "id": 123456,
                        "neighbourhood_cleansed": "Eixample",
                        "room_type": "Entire home/apt",
                        "property_type": "Entire rental unit",
                        "accommodates": 4,
                        "latitude": 41.39,

                        "longitude": 2.16,
                        "actual_price": 80.0,
                        "predicted_price": 120.0,
                        "diff_usd": -40.0,
                        "diff_percent": -33.3,
                        "flag": "underpriced",
                    }
                ],
            }
        }
    )

    count: int
    listings: list[MispricedListing]


class MarketRow(BaseModel):
    """صف من market_summary.csv (ملخص حي واحد)."""

    city: str
    neighbourhood_cleansed: str | None = None
    n_listings: int | None = None
    median_actual_price: float | None = None
    median_predicted_price: float | None = None
    median_occupancy_days: float | None = None
    median_annual_revenue: float | None = None
    share_underpriced: float | None = None


class MarketResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={

            "example": {
                "count": 1,
                "rows": [
                    {
                        "city": "Barcelona",
                        "neighbourhood_cleansed": "Eixample",
                        "n_listings": 100,
                        "median_actual_price": 120.0,
                        "median_predicted_price": 118.0,
                        "median_occupancy_days": 200.0,
                        "median_annual_revenue": 24000.0,
                        "share_underpriced": 0.4,
                    }
                ],
                "note": "Occupancy and revenue are estimates (Inside Airbnb occupancy model + our price model).",
            }
        }
    )

    count: int
    rows: list[MarketRow]
    note: str


class InvestRequest(ListingInput):
    """نفس مدخلات ListingInput + سعر الشراء وافتراضات التكاليف والإشغال."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "city": "Barcelona",
                "latitude": 41.3951,

                "longitude": 2.1650,
                "room_type": "Entire home/apt",
                "property_type": "Entire rental unit",
                "accommodates": 4,
                "bedrooms": 2,
                "bathrooms": 1,
                "purchase_price_usd": 400000,
                "annual_costs_usd": 5000,
                "occupancy_days": 200,
            }
        }
    )

    purchase_price_usd: float = Field(gt=0)
    annual_costs_usd: float = Field(default=0, ge=0)
    # لو فاضي بنستخدم متوسط الإشغال للمدينة من market_summary.csv
    occupancy_days: float | None = Field(default=None, ge=0, le=365)


class InvestResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "city": "Barcelona",
                "predicted_price": 100.0,
                "occupancy_days": 200.0,
                "purchase_price_usd": 400000.0,
                "annual_costs_usd": 4000.0,
                "annual_revenue_usd": 20000.0,
                "gross_yield_pct": 5.0,
                "net_revenue_usd": 16000.0,
                "net_yield_pct": 4.0,

                "assumptions": [
                    "price: nightly price predicted by the TravelPulse model (USD), not an observed price",
                    "occupancy: 200 days/year supplied by the caller (override)",
                ],
                "currency": "USD",
            }
        }
    )

    city: City
    predicted_price: float
    occupancy_days: float
    purchase_price_usd: float
    annual_costs_usd: float
    annual_revenue_usd: float
    gross_yield_pct: float
    net_revenue_usd: float
    net_yield_pct: float
    assumptions: list[str]
    currency: Literal["USD"] = "USD"


class AdvisorRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "listing": {
                    "city": "Barcelona",
                    "latitude": 41.3951,
                    "longitude": 2.1650,
                    "room_type": "Entire home/apt",
                    "property_type": "Entire rental unit",

                    "accommodates": 4,
                },
                "question": "Why is the predicted price what it is?",
            }
        }
    )

    listing: ListingInput
    question: str | None = Field(default=None, max_length=1000)


class AdvisorResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "answer": "The price is driven mainly by location and the number of guests it sleeps.",
                "predicted_price": 126.0,
                "used_key": True,
            }
        }
    )

    answer: str
    predicted_price: float
    used_key: bool

class SeasonalityRow(BaseModel):
    """شهر واحد في مدينة: متوسط نسبة الإتاحة وعدد الأيام."""

    city: City
    month: int = Field(ge=1, le=12)
    mean_share_available: float = Field(ge=0, le=1)
    mean_n_available: float
    n_days: int = Field(ge=1)


class SeasonalityResponse(BaseModel):
    rows: list[SeasonalityRow]
    note: str


class ParseListingRequest(BaseModel):
    """وصف حر للوحدة (عربي أو إنجليزي) يتحوّل لـ ListingInput."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"text": "شقة غرفتين في برشلونة جنب Sagrada Familia فيها مطبخ و Wifi"}
        }
    )

    text: str = Field(min_length=3, max_length=2000)


class ParseListingResponse(BaseModel):
    listing: dict
