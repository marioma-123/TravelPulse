"""تحميل ملفات الموديل مرة واحدة، وتنفيذ التنبؤ.

الوضع الافتراضي بقى "الخليط" (blend): XGBoost + LightGBM + CatBoost بأوزان
اتعلّمت من تنبؤات out-of-fold. الموديل القديم (XGBoost لوحده) لسه موجود
احتياطي، ويرجع يشتغل لو MODEL_MODE=baseline أو لو ملفات الخليط مش موجودة.
"""
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from .config import ARTIFACTS_FILE, CITY_CENTERS, LANDMARKS_FILE
from .features import build_feature_row

logger = logging.getLogger(__name__)


class PredictionOutOfRange(ValueError):
    """الموديل رجّع سعر صفر أو سالب: المدخلات غالباً خارج اللي اتدرب عليه."""


@dataclass
class BlendModels:
    xgb_booster: xgb.Booster   # موديل XGBoost من الخليط
    lgbm_booster: object       # LgbmTrees (بدون مكتبة lightgbm)
    cat_model: object          # CatTrees (بدون مكتبة catboost)
    weights: dict              # {"xgb": w1, "lgbm": w2, "cat": w3} - مجموعهم 1


@dataclass
class ModelBundle:
    booster: xgb.Booster   # موديل XGBoost الأساسي (احتياطي + وضع baseline)
    artifacts: dict        # الأدوات المحفوظة من التدريب (medians, ترميز الأحياء, ترتيب الأعمدة...)
    landmarks: dict        # لكل مدينة: مصفوفة إحداثيات المعالم
    blend: BlendModels | None = None   # الخليط لو موجود ومفعّل
    mode: str = "xgb_baseline"         # "blend_xgb_lgbm_cat" أو "xgb_baseline"


def _load_blend(model_dir: Path, artifacts: dict, columns: list[str]) -> BlendModels:
    """بيحمّل موديلات الخليط الثلاثة ويتأكد إن أعمدتها نفس أعمدة التطبيق."""
    from .cat_numpy import CatTrees
    from .lgbm_numpy import LgbmTrees

    spec = artifacts["blend"]
    files = spec["files"]

    xgb_booster = xgb.Booster()
    xgb_booster.load_model(str(model_dir / files["xgb"]))
    if xgb_booster.feature_names != columns:
        raise ValueError("Blend XGBoost feature names do not match artifacts feature_columns")

    lgbm_booster = LgbmTrees(model_dir / files["lgbm_trees"])
    # LightGBM بيغيّر المسافات لـ _ في أسماء الأعمدة جوه الملف، فبنقارن بعد التطبيع
    sanitized = [c.replace(" ", "_") for c in columns]
    if lgbm_booster.feature_names != sanitized:
        raise ValueError("Blend LightGBM feature names do not match artifacts feature_columns")

    cat_model = CatTrees(model_dir / files["cat_trees"])
    if cat_model.feature_names != columns:
        raise ValueError("Blend CatBoost feature names do not match artifacts feature_columns")

    weights = {k: float(v) for k, v in spec["weights"].items()}
    if abs(sum(weights.values()) - 1.0) > 1e-6:
        raise ValueError(f"Blend weights must sum to 1, got {sum(weights.values())}")
    return BlendModels(xgb_booster=xgb_booster, lgbm_booster=lgbm_booster,
                       cat_model=cat_model, weights=weights)


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

    # الخليط: افتراضي شغال طالما موجود في artifacts، ويتعطّل بـ MODEL_MODE=baseline
    blend, mode = None, "xgb_baseline"
    if "blend" in artifacts and os.getenv("MODEL_MODE", "blend").lower() != "baseline":
        try:
            blend = _load_blend(model_dir, artifacts, artifacts["feature_columns"])
            mode = "blend_xgb_lgbm_cat"
        except Exception as exc:  # أي مشكلة في ملفات/مكتبات الخليط: نكمّل بالموديل القديم بدل ما التطبيق يقع
            logger.warning("Blend failed to load (%s: %s); falling back to the baseline XGBoost model.",
                           type(exc).__name__, exc)
            blend, mode = None, "xgb_baseline"

    return ModelBundle(booster=booster, artifacts=artifacts, landmarks=landmarks,
                       blend=blend, mode=mode)


def blend_log_price(blend: BlendModels, features: pd.DataFrame) -> float:
    """متوسط موزون لتوقعات الموديلات الثلاثة في الـ log (نفس معادلة التدريب)."""
    row = features.iloc[0].to_numpy(dtype=float).reshape(1, -1)
    p_xgb = float(blend.xgb_booster.predict(xgb.DMatrix(row, feature_names=list(features.columns)))[0])
    p_lgbm = blend.lgbm_booster.predict_row(row)
    p_cat = blend.cat_model.predict_row(row)
    w = blend.weights
    return w["xgb"] * p_xgb + w["lgbm"] * p_lgbm + w["cat"] * p_cat


def _check_price(price: float) -> float:
    if not np.isfinite(price) or price <= 0:
        # مش بنرجّع 0 كأنه سعر حقيقي: ده معناه إن الموديل فشل مع المدخلات دي.
        raise PredictionOutOfRange(
            "The model returned a non-positive price for this input; the input is "
            "probably outside the range the model was trained on. Check the values."
        )
    return price


def predict_price(bundle: ModelBundle, listing: dict) -> float:
    """السعر المتوقع بالدولار لوحدة واحدة."""
    features = build_feature_row(listing, bundle.artifacts, bundle.landmarks)
    if bundle.blend is not None:
        log_price = blend_log_price(bundle.blend, features)
    else:
        log_price = float(bundle.booster.predict(xgb.DMatrix(features))[0])
    # الموديل اتدرب على log1p(price) فبنرجّع للدولار بـ expm1.
    return _check_price(float(np.expm1(log_price)))
