"""المستشار: بياخد توقّع الموديل وتفسير SHAP ويبعتهم لـ Gemini عشان يجاوب بدون ما يخترع أرقام."""
from __future__ import annotations

import json
import os
from typing import Any, Optional
from urllib.parse import quote

import httpx

from app.explain import explain_price
from app.model_loader import predict_price

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_MODEL = "gemini-3-flash-preview"
TIMEOUT_SECONDS = 60.0
DEFAULT_QUESTION = "Explain this predicted price in plain language and what drives it."

SYSTEM_INSTRUCTION = (
    "You are an assistant for an Airbnb price tool. "
    "Use ONLY the numbers provided below. "
    "Never invent prices, revenues, occupancy or market stats. "
    "If asked something the data can't answer, say so."
)


class AdvisorError(RuntimeError):
    """Gemini call failed; the message is short and safe to return to the client."""


def get_api_key() -> Optional[str]:
    key = os.environ.get("GEMINI_API_KEY", "").strip()

    return key or None


def get_model() -> str:
    return os.environ.get("GEMINI_MODEL", "").strip() or DEFAULT_MODEL


def build_user_message(
    listing: dict[str, Any],
    predicted_price: float,
    explanation: dict[str, Any],
    question: Optional[str],
) -> str:
    data = {
        "listing_as_submitted": {k: v for k, v in listing.items() if v is not None},
        "predicted_price_usd": round(predicted_price, 2),
        "baseline_price_usd": explanation.get("baseline_price"),
        "groups": explanation.get("groups"),
        "top_factors": explanation.get("top_factors"),
    }
    asked = question.strip() if question and question.strip() else DEFAULT_QUESTION
    return (
        f"Predicted nightly price (USD): {predicted_price:.2f}\n\n"
        f"Data:\n{json.dumps(data, default=str, ensure_ascii=False)}\n\n"
        f"Question: {asked}"
    )


def build_request_body(user_message: str, model: str = "", thinking_level: str = "low") -> dict[str, Any]:
    body = {
        "system_instruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
        "contents": [{"role": "user", "parts": [{"text": user_message}]}],

        "generationConfig": {"temperature": 0.2},
    }
    if "gemini-3" in model:
        body["generationConfig"]["thinkingConfig"] = {"thinkingLevel": thinking_level}
    return body


def call_gemini(api_key: str, model: str, user_message: str, thinking_level: str = "low") -> str:
    """POST to Gemini REST. Never logs or echoes the key; errors are mapped to AdvisorError."""
    url = GEMINI_URL.format(model=quote(model, safe="-._"))
    try:
        response = httpx.post(
            url,
            headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
            json=build_request_body(user_message, model, thinking_level),
            timeout=TIMEOUT_SECONDS,
        )
    except httpx.TimeoutException:
        raise AdvisorError("Gemini request timed out") from None
    except httpx.HTTPError:
        raise AdvisorError("Could not reach Gemini") from None

    if response.status_code != 200:
        raise AdvisorError(f"Gemini returned HTTP {response.status_code}")

    try:
        payload = response.json()
        parts = payload["candidates"][0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts).strip()
    except (ValueError, KeyError, IndexError, TypeError, AttributeError):
        raise AdvisorError("Gemini returned no usable answer") from None

    if not text:
        raise AdvisorError("Gemini returned an empty answer")
    return text



def ask_advisor(
    api_key: str,
    bundle: Any,
    listing: dict[str, Any],
    question: Optional[str],
) -> dict[str, Any]:
    predicted = float(predict_price(bundle, listing))
    explanation = explain_price(bundle, listing)
    message = build_user_message(listing, predicted, explanation, question)
    answer = call_gemini(api_key, get_model(), message)
    return {"answer": answer, "predicted_price": round(predicted, 2), "used_key": True}