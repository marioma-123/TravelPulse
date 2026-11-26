"""شكل البيانات الداخلة والخارجة من الـ API (العقد بين المستخدم والخدمة)."""
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import CITY_CENTERS, MAX_DEGREES_FROM_CENTER

City = Literal["Amsterdam", "Barcelona", "Berlin", "Columbus"]
RoomType = Literal["Entire home/apt", "Private room", "Shared room", "Hotel room"]


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
    model: str = "xgb_baseline"  # الموديل الشغال: blend_xgb_lgbm_cat أو xgb_baseline


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
