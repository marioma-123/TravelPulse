"""حسابات العائد الاستثماري: السعر المتوقع × الإشغال ÷ سعر الشراء، مع كل الافتراضات مكتوبة."""
from __future__ import annotations

from typing import Any, Optional

from app.market import city_median_occupancy
from app.model_loader import predict_price


def build_investment(
    price: float,
    occupancy_days: float,
    purchase_price_usd: float,
    annual_costs_usd: float = 0.0,
) -> dict[str, float]:
    """Pure math. Money rounded to 2 decimals, percentages to 1."""
    revenue = price * occupancy_days
    net_revenue = revenue - annual_costs_usd
    return {
        "annual_revenue_usd": round(revenue, 2),
        "gross_yield_pct": round(100 * revenue / purchase_price_usd, 1),
        "net_revenue_usd": round(net_revenue, 2),
        "net_yield_pct": round(100 * net_revenue / purchase_price_usd, 1),
    }


def resolve_occupancy(state: Any, city: str, override: Optional[float]) -> tuple[float, str]:
    """Return (occupancy_days, assumption_text). Raises DataUnavailable if no override and no data."""
    if override is not None:
        return float(override), f"occupancy: {override:g} days/year supplied by the caller (override)"
    value, n_neighbourhoods = city_median_occupancy(state, city)
    return value, (

        f"occupancy: median of the neighbourhood medians for {city} "
        f"({n_neighbourhoods} neighbourhoods) from market_summary.csv (Inside Airbnb estimate)"
    )


def run_investment(
    state: Any,
    listing: dict[str, Any],
    city: str,
    purchase_price_usd: float,
    annual_costs_usd: float,
    occupancy_override: Optional[float],
) -> dict[str, Any]:
    """Predict the nightly price, resolve occupancy, compute yields and list the assumptions."""
    price = float(predict_price(state.bundle, listing))
    occupancy, occupancy_note = resolve_occupancy(state, city, occupancy_override)
    money = build_investment(price, occupancy, purchase_price_usd, annual_costs_usd)

    if annual_costs_usd > 0:
        costs_note = f"annual costs: {annual_costs_usd:,.2f} USD supplied by the caller"
    else:
        costs_note = "annual costs: not provided, assumed 0 USD (net figures equal gross figures)"

    assumptions = [
        "price: nightly price predicted by the TravelPulse model (USD), not an observed price",
        occupancy_note,
        "annual revenue = predicted nightly price x occupancy days; "
        "seasonality, platform fees and taxes are not modelled",
        f"purchase price: {purchase_price_usd:,.2f} USD supplied by the caller",
        costs_note,
    ]
    return {

        "city": city,
        "predicted_price": round(price, 2),
        "occupancy_days": occupancy,
        "purchase_price_usd": round(purchase_price_usd, 2),
        "annual_costs_usd": round(annual_costs_usd, 2),
        **money,
        "assumptions": assumptions,
    }