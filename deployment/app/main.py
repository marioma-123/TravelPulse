from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="TravelPulse API")


class ListingInput(BaseModel):
    city: str
    room_type: str
    property_type: str
    accommodates: int = Field(ge=1, le=16)
    bedrooms: float = Field(ge=0, le=10)
    beds: float = Field(ge=0, le=20)
    bathrooms: float = Field(ge=0, le=10)
    minimum_nights: int = Field(ge=1, le=365)
    host_is_superhost: bool = False


def dummy_predict(listing: ListingInput) -> float:
    # موديل وهمي: معادلة بسيطة، هنبدّلها بالموديل الحقيقي بعدين
    return 30 + 18 * listing.accommodates + 12 * listing.bedrooms


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(listing: ListingInput):
    price = dummy_predict(listing)
    return {"city": listing.city, "predicted_price": round(price, 2)}
