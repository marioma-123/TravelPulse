"""تحميل ملفات الموديل مرة واحدة، وتنفيذ التنبؤ."""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from .config import ARTIFACTS_FILE, CITY_CENTERS, LANDMARKS_FILE
from .features import build_feature_row


class PredictionOutOfRange(ValueError):
    """الموديل رجّع سعر صفر أو سالب: المدخلات غالباً خارج اللي اتدرب عليه."""


@dataclass
class ModelBundle:
    booster: xgb.Booster   # موديل XGBoost نفسه
    artifacts: dict        # الأدوات المحفوظة من التدريب (medians, ترميز الأحياء, ترتيب الأعمدة...)
    landmarks: dict        # لكل مدينة: مصفوفة إحداثيات المعالم


def load_bundle(model_dir: Path) -> ModelBundle:
    artifacts_path = model_dir / ARTIFACTS_FILE
    if not artifacts_path.exists():
        raise FileNotFoundError(
            f"{artifacts_path} not found. Copy the files exported from the modeling "
            "notebook into the model folder, or run `python -m models.make_dummy_model` "
            "and start the API with MODEL_DIR=models/dummy."
        )
    artifacts = json.loads(artifacts_path.read_text())

    model_path = model_dir / artifacts["model_file"]
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    booster = xgb.Booster()
    booster.load_model(str(model_path))

    landmarks_path = model_dir / LANDMARKS_FILE
    if not landmarks_path.exists():
        raise FileNotFoundError(f"Landmarks file not found: {landmarks_path}")
    lm = pd.read_csv(landmarks_path, usecols=["city", "latitude", "longitude"])
    landmarks = {
        city: group[["latitude", "longitude"]].to_numpy(dtype=float)
        for city, group in lm.groupby("city")
    }
    missing = [c for c in CITY_CENTERS if c not in landmarks]
    if missing:
        raise ValueError(f"No landmarks for cities: {missing}")

    # أمان: لو أسماء/ترتيب الأعمدة في الموديل مختلف عن الملف، نرفض من البداية
    if booster.feature_names != artifacts["feature_columns"]:
        raise ValueError("Model feature names do not match artifacts feature_columns")

    return ModelBundle(booster=booster, artifacts=artifacts, landmarks=landmarks)


def predict_price(bundle: ModelBundle, listing: dict) -> float:
    """السعر المتوقع بالدولار لوحدة واحدة."""
    features = build_feature_row(listing, bundle.artifacts, bundle.landmarks)
    log_price = float(bundle.booster.predict(xgb.DMatrix(features))[0])
    # الموديل اتدرب على log1p(price) فبنرجّع للدولار بـ expm1.
    price = float(np.expm1(log_price))
    if not np.isfinite(price) or price <= 0:
        # مش بنرجّع 0 كأنه سعر حقيقي: ده معناه إن الموديل فشل مع المدخلات دي.
        raise PredictionOutOfRange(
            "The model returned a non-positive price for this input; the input is "
            "probably outside the range the model was trained on. Check the values."
        )
    return price
