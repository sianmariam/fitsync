"""Gemini is used for what LLMs are good at — naming foods and recalling nutrition facts —
NOT for measuring portion size. Its own gram guess is kept only as a baseline to compare against.

Set GEMINI_API_KEY (and optionally GEMINI_MODEL) in the environment or enter the key in the app sidebar.
"""
from __future__ import annotations

import json
import os

import cv2

from .core import Det

PROMPT = """You are a food recognition assistant. Look at this photo of food (a coin is on the table for scale).
Return ONLY a JSON list, one object per distinct food item (ignore the coin, plate, cutlery):
[{"name": short lowercase food name using underscores (e.g. "rice", "dosa", "chicken_curry", "apple"),
  "box_2d": [ymin, xmin, ymax, xmax] normalised to 0-1000,
  "density_g_cm3": typical density of this food as served,
  "kcal_100g": kcal per 100 g, "protein_100g": g, "carbs_100g": g, "fat_100g": g,
  "gemini_grams": your best guess of this item's weight in grams}]"""


def default_model():
    return os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


def analyze(bgr, api_key: str | None = None, model: str | None = None):
    from google import genai
    from google.genai import types
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("No Gemini API key")
    client = genai.Client(api_key=key)
    ok, jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    resp = client.models.generate_content(
        model=model or default_model(),
        contents=[types.Part.from_bytes(data=jpg.tobytes(), mime_type="image/jpeg"), PROMPT],
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.0),
    )
    items = json.loads(resp.text)
    if isinstance(items, dict):
        items = items.get("items", [items])
    return items


def to_dets(items, shape) -> list[Det]:
    h, w = shape[:2]
    out = []
    for it in items:
        b = it.get("box_2d")
        if not b or len(b) != 4:
            continue
        y1, x1, y2, x2 = b
        out.append(Det(it["name"], (x1 / 1000 * w, y1 / 1000 * h, x2 / 1000 * w, y2 / 1000 * h), 0.5))
    return out
