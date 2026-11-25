"""نطاق السعر (split conformal prediction).

الفكرة: على مجموعة معايرة (validation) محدش اتدرّب عليها، بنحسب الخطأ المطلق في الـ log
لكل وحدة. الربع الأعلى (مثلاً 80%) من الأخطاء دي = q. بعدها أي توقع جديد:
    النطاق = [expm1(log_pred - q), expm1(log_pred + q)]
وبيغطي السعر الحقيقي بنسبة ~80% من الحالات (بشرط إن الوحدات الجديدة شبه بيانات المعايرة).
القيم q بتتحسب مرة واحدة في الـ notebook وتتحفظ في artifacts.json تحت المفتاح "conformal".
"""
import math

import numpy as np


def level_key(level: float) -> str:
    return f"{level:g}"


def compute_conformal_quantiles(y_true_log, y_pred_log, levels=(0.8, 0.9)) -> dict:
    """بيحسب q لكل مستوى ثقة (بتصحيح العيّنة المحدودة: ceil((n+1)*level))."""
    residuals = np.sort(np.abs(np.asarray(y_true_log, float) - np.asarray(y_pred_log, float)))
    n = len(residuals)
    if n < 50:
        raise ValueError(f"need at least 50 calibration rows, got {n}")
    out = {}
    for level in levels:
        k = min(math.ceil((n + 1) * level), n)
        out[level_key(level)] = float(residuals[k - 1])
    return {"n_calibration": n, "quantiles": out}


def price_interval(log_pred: float, conformal: dict, level: float) -> tuple[float, float]:
    q = conformal["quantiles"][level_key(level)]
    lower = max(float(np.expm1(log_pred - q)), 0.0)
    upper = float(np.expm1(log_pred + q))
    return lower, upper
