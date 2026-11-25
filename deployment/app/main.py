"""نقطة دخول الـ API: بتعرّف الـ endpoints وبتحمّل الموديل مرة واحدة عند التشغيل."""
from contextlib import asynccontextmanager

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.staticfiles import StaticFiles

from .config import get_model_dir
from .model_loader import PredictionOutOfRange, load_bundle, predict_price
from .explain import explain_price
import numpy as np

from .advisor import AdvisorError, ask_advisor, get_api_key
from .data_files import DataUnavailable
from .interval import level_key, price_interval
from .invest import run_investment
from .market import NOTE as MARKET_NOTE
from .market import query_market
from .mispricing import query_mispriced
from .parse_listing import parse_listing_text
from .seasonality import get_seasonality
from .schemas import (AdvisorRequest, AdvisorResponse, City, ExplanationOutput, FlagFilter,
                      HealthOutput, ImproveOutput, InvestRequest, InvestResponse,
                      ListingInput, MarketResponse, MispricingResponse, ParseListingRequest,
                      ParseListingResponse, PredictionOutput, RangeOutput,
                      SeasonalityResponse, WhatIfOutput, WhatIfRequest)
from .whatif import run_scenarios, suggest_improvements


@asynccontextmanager
async def lifespan(app: FastAPI):
    # مرة واحدة وقت التشغيل. لو ملف ناقص أو غير متطابق، السيرفر مش بيشتغل.
    app.state.bundle = load_bundle(get_model_dir())
    yield



app = FastAPI(title="TravelPulse API", version="0.2.0", lifespan=lifespan)


@app.get("/health", response_model=HealthOutput)
def health(request: Request):
    return HealthOutput(
        status="ok",
        model_loaded=getattr(request.app.state, "bundle", None) is not None,
    )


@app.post("/predict", response_model=PredictionOutput)
def predict(listing: ListingInput, request: Request):
    try:
        price = predict_price(request.app.state.bundle, listing.model_dump())
    except PredictionOutOfRange as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return PredictionOutput(city=listing.city, predicted_price=round(price, 2))


@app.post("/explain", response_model=ExplanationOutput)
def explain(listing: ListingInput, request: Request):
    """نفس مدخلات /predict، لكن بيرجّع كمان إيه اللي زوّد السعر وإيه اللي نقّصه."""
    try:
        result = explain_price(request.app.state.bundle, listing.model_dump())
    except PredictionOutOfRange as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return ExplanationOutput(city=listing.city, **result)



@app.post("/predict-range", response_model=RangeOutput)
def predict_range(listing: ListingInput, request: Request, confidence: float = 0.8):
    """السعر المتوقع + نطاق (conformal). confidence لازم يكون من المستويات المحسوبة."""
    bundle = request.app.state.bundle
    conformal = bundle.artifacts.get("conformal")
    if not conformal:
        raise HTTPException(status_code=503, detail="Calibration data not available: run tools/add_conformal.py")
    if level_key(confidence) not in conformal["quantiles"]:
        raise HTTPException(status_code=422, detail=f"confidence must be one of {sorted(conformal['quantiles'])}")
    try:
        price = predict_price(bundle, listing.model_dump())
    except PredictionOutOfRange as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    lower, upper = price_interval(float(np.log1p(price)), conformal, confidence)
    return RangeOutput(city=listing.city, predicted_price=round(price, 2),
                       lower=round(lower, 2), upper=round(upper, 2), confidence=confidence)


@app.post("/whatif", response_model=WhatIfOutput)
def whatif(req: WhatIfRequest, request: Request):
    """بيجرّب كذا سيناريو تغيير على نفس الوحدة ويقارن بالسعر الأساسي."""
    scenarios = [{"name": s.name, "changes": s.changes.model_dump(exclude_none=True)} for s in req.scenarios]
    try:
        return run_scenarios(request.app.state.bundle, req.listing.model_dump(), scenarios)
    except PredictionOutOfRange as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.post("/improve", response_model=ImproveOutput)
def improve(listing: ListingInput, request: Request):
    """بيجرّب تحسينات جاهزة (مرفق ناقص، Superhost...) ويرتّبها حسب أكبر زيادة متوقعة."""
    try:

        return suggest_improvements(request.app.state.bundle, listing.model_dump())
    except PredictionOutOfRange as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.get("/mispricing", response_model=MispricingResponse)
def mispricing(
    request: Request,
    city: City | None = None,
    flag: FlagFilter | None = None,
    limit: int = Query(20, ge=1, le=100),
):
    """الوحدات اللي سعرها بعيد عن توقّع الموديل، مرتبة من الأرخص نسبياً للأغلى."""
    try:
        rows = query_mispriced(request.app.state, city, flag, limit)
    except DataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return {"count": len(rows), "listings": rows}


@app.get("/market", response_model=MarketResponse)
def market(request: Request, city: City | None = None):
    """ملخص السوق لكل حي (واختياري لمدينة واحدة)، من market_summary.csv."""
    try:
        rows = query_market(request.app.state, city)
    except DataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return {"count": len(rows), "rows": rows, "note": MARKET_NOTE}


@app.post("/invest", response_model=InvestResponse)
def invest(req: InvestRequest, request: Request):

    """العائد الإجمالي والصافي من السعر المتوقع وافتراض الإشغال، وكل الافتراضات مكتوبة."""
    listing = req.model_dump(
        exclude={"purchase_price_usd", "annual_costs_usd", "occupancy_days"}
    )
    try:
        return run_investment(
            request.app.state,
            listing,
            req.city,
            req.purchase_price_usd,
            req.annual_costs_usd,
            req.occupancy_days,
        )
    except PredictionOutOfRange as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except DataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/advisor", response_model=AdvisorResponse)
def advisor(req: AdvisorRequest, request: Request):
    """إجابة بلغة بسيطة من Gemini، ومقيّدة بأرقام الموديل بس."""
    api_key = get_api_key()
    if api_key is None:
        raise HTTPException(
            status_code=503, detail="Advisor is disabled: GEMINI_API_KEY is not set"
        )
    try:
        return ask_advisor(
            api_key, request.app.state.bundle, req.listing.model_dump(), req.question
        )
    except PredictionOutOfRange as exc:

        raise HTTPException(status_code=422, detail=str(exc))
    except AdvisorError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.get("/seasonality", response_model=SeasonalityResponse)
def seasonality(request: Request, city: City | None = None):
    """متوسط نسبة الإتاحة لكل مدينة وشهر (proxy للإتاحة، مش طلب مقيس)."""
    try:
        return get_seasonality(request.app.state, city)
    except DataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/parse-listing", response_model=ParseListingResponse)
def parse_listing_endpoint(req: ParseListingRequest):
    """بيحوّل وصف حر لـ ListingInput مُتحقق منه عن طريق Gemini (503 لو مفيش مفتاح)."""
    try:
        return parse_listing_text(req.text)
    except AdvisorError as exc:
        msg = str(exc)
        status = 503 if "GEMINI_API_KEY" in msg else 502
        raise HTTPException(status_code=status, detail=msg)


# الواجهة (نفس الـ origin، فمفيش CORS): http://localhost:8000/ui/
WEB_DIR = Path(__file__).resolve().parent.parent / "web"
if WEB_DIR.exists():
    app.mount("/ui", StaticFiles(directory=WEB_DIR, html=True), name="ui")