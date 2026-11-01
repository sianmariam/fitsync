"""Gemini-only baseline: ask Gemini for the weight of each food in held-out ECUSTFD test images and
compare with the measured ground truth — the same images our pipeline is scored on.

Usage:
  export GEMINI_API_KEY=...        (Windows: set GEMINI_API_KEY=...)
  python scripts/eval_gemini_baseline.py --n 60

Writes results/gemini_baseline.csv and prints Gemini's MAPE next to ours.
Free-tier keys are rate limited, so the script sleeps between calls (--sleep).
"""
import argparse, sys, time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from foodvol import gemini  # noqa

PROMPT = ("This photo shows {foods} on a plate. A coin of 25 mm diameter is on the table for scale. "
          "Estimate the weight in grams of each listed food. Return ONLY JSON: {{\"<food>\": grams, ...}}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--sleep", type=float, default=6.0)
    a = ap.parse_args()
    from google import genai
    from google.genai import types
    import json, os
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    pred = pd.read_csv(ROOT / "results" / "test_predictions.csv")
    imgs = pred.image.drop_duplicates().sample(min(a.n, pred.image.nunique()), random_state=0)
    rows = []
    for i, im in enumerate(imgs):
        sub = pred[pred.image == im]
        foods = [f.replace("_", " ").replace("qiwi", "kiwi") for f in sub.food]
        bgr = cv2.imread(str(ROOT / "data" / "ECUSTFD" / "JPEGImages" / f"{im}.JPG"))
        ok, jpg = cv2.imencode(".jpg", bgr)
        try:
            r = client.models.generate_content(
                model=gemini.default_model(),
                contents=[types.Part.from_bytes(data=jpg.tobytes(), mime_type="image/jpeg"),
                          PROMPT.format(foods=" and ".join(foods))],
                config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.0))
            g = {k.lower().replace(" ", "_").replace("kiwi", "qiwi"): float(v) for k, v in json.loads(r.text).items()}
        except Exception as e:  # noqa
            print("skip", im, e)
            time.sleep(a.sleep)
            continue
        for _, row in sub.iterrows():
            if row.food in g:
                rows.append(dict(image=im, food=row.food, gt_weight=row.gt_weight, gemini_g=g[row.food]))
        print(f"{i+1}/{len(imgs)} {im} {g}")
        time.sleep(a.sleep)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "gemini_baseline.csv", index=False)
    ape = (df.gemini_g - df.gt_weight).abs() / df.gt_weight * 100
    print(f"\nGemini-only mass MAPE on {len(df)} items: {ape.mean():.1f}%  (within ±20%: {(ape<=20).mean()*100:.1f}%)")
    # same images, our method (mass = volume x measured density)
    import json as _j
    dens = _j.load(open(ROOT / "models" / "densities.json"))
    ours = pred.merge(df[["image", "food"]], on=["image", "food"])
    om = ours["OURS Depth + ML"] * ours.food.map(dens)
    oape = (om - ours.gt_weight).abs() / ours.gt_weight * 100
    print(f"Ours (same images)  mass MAPE: {oape.mean():.1f}%  (within ±20%: {(oape<=20).mean()*100:.1f}%)")


if __name__ == "__main__":
    main()
