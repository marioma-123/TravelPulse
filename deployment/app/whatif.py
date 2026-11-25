"""What-if: بنغيّر في مواصفات الوحدة ونشوف السعر المتوقع اتغيّر كام.

مهم: ده تغيّر في "توقع الموديل" بناءً على أنماط في البيانات (ارتباط)، مش وعد إن السعر
الفعلي هيزيد بنفس المقدار لو المضيف عمل التغيير ده فعلاً.
"""
from .features import amenity_features
from .model_loader import ModelBundle, PredictionOutOfRange, predict_price


def apply_changes(listing: dict, changes: dict) -> dict:
    new = dict(listing)
    new["amenities"] = list(listing.get("amenities") or [])
    for field, value in changes.items():
        if field == "add_amenities":
            have = {a.lower() for a in new["amenities"]}
            new["amenities"] += [a for a in value if a.lower() not in have]
        elif field == "remove_amenities":
            drop = [v.lower() for v in value]
            new["amenities"] = [a for a in new["amenities"] if not any(d in a.lower() for d in drop)]
        else:
            new[field] = value
    return new


def run_scenarios(bundle: ModelBundle, listing: dict, scenarios: list[dict]) -> dict:
    base = predict_price(bundle, listing)  # لو فشل، الخطأ بيطلع للـ endpoint (422)
    results = []
    for sc in scenarios:
        try:
            price = predict_price(bundle, apply_changes(listing, sc["changes"]))
            results.append({
                "name": sc["name"], "predicted_price": round(price, 2),
                "change_usd": round(price - base, 2),
                "change_percent": round((price / base - 1) * 100, 1),
            })
        except PredictionOutOfRange as exc:
            results.append({"name": sc["name"], "error": str(exc)})
    return {"base_price": round(base, 2), "scenarios": results}


def improvement_candidates(bundle: ModelBundle, listing: dict) -> list[dict]:
    """التحسينات اللي ممكن تتجرّب: مرفق ناقص، أو Superhost، أو هوية موثّقة."""
    key_amenities = bundle.artifacts["key_amenities"]
    present = amenity_features(listing.get("amenities") or [], key_amenities)
    candidates = [
        {"name": f"Add {kw}", "changes": {"add_amenities": [kw]}}
        for col, kw in key_amenities.items() if not present[col]
    ]
    if not listing.get("host_is_superhost"):
        candidates.append({"name": "Become a Superhost", "changes": {"host_is_superhost": True}})
    if not listing.get("host_identity_verified"):
        candidates.append({"name": "Verify host identity", "changes": {"host_identity_verified": True}})
    return candidates


def suggest_improvements(bundle: ModelBundle, listing: dict, top_n: int = 5) -> dict:
    out = run_scenarios(bundle, listing, improvement_candidates(bundle, listing))
    gains = [s for s in out["scenarios"] if s.get("change_usd", 0) > 0]
    gains.sort(key=lambda s: s["change_usd"], reverse=True)
    return {"base_price": out["base_price"], "suggestions": gains[:top_n]}
