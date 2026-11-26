"""نقطة دخول الـ API: بتعرّف الـ endpoints وبتحمّل الموديل مرة واحدة عند التشغيل."""
from contextlib import asynccontextmanager

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles

from .config import get_model_dir
from .model_loader import PredictionOutOfRange, load_bundle, predict_price
from .explain import explain_price
import numpy as np

from .interval import level_key, price_interval
from .schemas import (ExplanationOutput, HealthOutput, ImproveOutput, ListingInput,
                      PredictionOutput, RangeOutput, WhatIfOutput, WhatIfRequest)
from .whatif import run_scenarios, suggest_improvements


@asynccontextmanager
async def lifespan(app: FastAPI):
    # مرة واحدة وقت التشغيل. لو ملف ناقص أو غير متطابق، السيرفر مش بيشتغل.
    app.state.bundle = load_bundle(get_model_dir())
    yield


app = FastAPI(title="TravelPulse API", version="0.3.0", lifespan=lifespan)


@app.get("/health", response_model=HealthOutput)
def health(request: Request):
    bundle = getattr(request.app.state, "bundle", None)
    return HealthOutput(
        status="ok",
        model_loaded=bundle is not None,
        model=bundle.mode if bundle is not None else "none",
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


# الواجهة (نفس الـ origin، فمفيش CORS): http://localhost:8000/ui/
WEB_DIR = Path(__file__).resolve().parent.parent / "web"
if WEB_DIR.exists():
    app.mount("/ui", StaticFiles(directory=WEB_DIR, html=True), name="ui")
