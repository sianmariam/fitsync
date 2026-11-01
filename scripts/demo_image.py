"""Run the full pipeline on one image and save a 4-panel figure (for slides / report).

Usage: python scripts/demo_image.py path/to/photo.jpg [--coin 2.5] [--out results/demo.png]
"""
import argparse, sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from foodvol.core import FoodVolumePipeline  # noqa


def colorize(x):
    lo, hi = np.percentile(x, 1), np.percentile(x, 99.5)
    v = np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)
    return cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


def label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (20, 20, 20), -1)
    cv2.putText(out, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--coin", type=float, default=2.5)
    ap.add_argument("--out", default=str(ROOT / "results" / "demo.png"))
    a = ap.parse_args()
    bgr = cv2.imread(a.image)
    pipe = FoodVolumePipeline()
    res = pipe.run(bgr, coin_diam_cm=a.coin)
    if res["error"]:
        print(res["error"]); return
    ov = bgr.copy()
    for it, m in zip(res["items"], res["masks"]):
        layer = ov.copy(); layer[m] = (113, 204, 46)
        ov = cv2.addWeighted(layer, 0.45, ov, 0.55, 0)
        x1, y1 = int(it.box[0]), int(it.box[1])
        cv2.rectangle(ov, (x1, y1), (int(it.box[2]), int(it.box[3])), (113, 204, 46), 2)
    if res.get("coin"):
        cb = [int(v) for v in res["coin"].box]
        cv2.rectangle(ov, (cb[0], cb[1]), (cb[2], cb[3]), (0, 215, 255), 2)
    if res.get("approx"):
        print("note: no coin found - scale assumed from a 35 cm camera distance")
    hg = colorize(res["hgt"]); hg[res["hgt"] <= 0] = (bgr[res["hgt"] <= 0] * 0.35).astype(np.uint8)
    panels = [label(bgr, "Input photo"), label(ov, "YOLOv8 detection + segmentation"),
              label(colorize(res["disp"]), "Depth Anything V2"), label(hg, "Height map above table")]
    top, bot = np.hstack(panels[:2]), np.hstack(panels[2:])
    fig = np.vstack([top, bot])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(a.out, fig)
    for it in res["items"]:
        print(f"{it.name:12s} vol={it.volume_cm3:6.0f} cm3 (raw depth {it.volume_depth_raw_cm3:6.0f})  "
              f"mass={it.mass_g:6.0f} g  kcal={it.kcal:5.0f}")
    print("saved", a.out)


if __name__ == "__main__":
    main()
