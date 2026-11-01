"""Run segmentation + Depth Anything V2 + coin calibration on every ECUSTFD image and save
geometric features joined with the measured ground-truth volume / weight.

Uses the dataset's annotated boxes (to measure volume-estimation error on its own, independent of detection).
Depth maps are cached in data/depth_cache/ so the script can be re-run cheaply.

Usage: python scripts/build_features.py [--views T S] [--limit N]
"""
import argparse, json, re, sys, time
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from foodvol.core import (DepthEstimator, calibrate_from_coin, height_map, item_features, segment_in_box)  # noqa

SRC = ROOT / "data" / "ECUSTFD"
ITEM_RE = re.compile(r"^([a-z_]+\d{3})([ST])\((\d+)\)")
BAD = {"mix002T(2)", "mix005S(4)"}


def load_gt():
    x = pd.read_excel(SRC / "density.xls", sheet_name=None)
    df = pd.concat(x.values(), ignore_index=True)
    df.columns = ["id", "type", "volume_cm3", "weight_g"]  # 'volume(mm^3)' column is actually mL (cm^3)
    return {(r.id, r.type): (float(r.volume_cm3), float(r.weight_g)) for r in df.itertuples()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", nargs="+", default=["T"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    gt = load_gt()
    split = json.load(open(ROOT / "data" / "item_split.json"))
    cache = ROOT / "data" / "depth_cache"
    cache.mkdir(exist_ok=True)
    depth = DepthEstimator(threads=a.threads)
    rows, t0 = [], time.time()
    files = sorted((SRC / "Annotations").glob("*.xml"))
    files = [f for f in files if (m := ITEM_RE.match(f.stem)) and m.group(2) in a.views and f.stem not in BAD]
    if a.limit:
        files = files[:a.limit]
    for i, xf in enumerate(files):
        m = ITEM_RE.match(xf.stem)
        item, view = m.group(1), m.group(2)
        r = ET.parse(xf).getroot()
        objs = [(o.find("name").text.strip(), tuple(float(o.find("bndbox/" + k).text)
                 for k in ("xmin", "ymin", "xmax", "ymax"))) for o in r.findall("object")]
        coin = next((b for n, b in objs if n == "coin"), None)
        foods = [(n, b) for n, b in objs if n != "coin"]
        if coin is None or not foods:
            continue
        img = cv2.imread(str(SRC / "JPEGImages" / (xf.stem + ".JPG")))
        if img is None:
            continue
        cp = cache / (xf.stem + ".npy")
        if cp.exists():
            disp = np.load(cp).astype(np.float32)
        else:
            disp = depth(img)
            np.save(cp, disp.astype(np.float16))
        calib = calibrate_from_coin(coin, img.shape[1])
        for name, box in foods:
            if (item, name) not in gt:
                continue
            mask = segment_in_box(img, box)
            hgt, rel = height_map(disp, mask, coin, calib)
            f = item_features(name, mask, hgt, rel, calib, box, view)
            vol, wt = gt[(item, name)]
            rows.append(dict(image=xf.stem, item=item, split=split.get(item, "train"),
                             gt_volume_cm3=vol, gt_weight_g=wt, **asdict(f)))
        if i % 50 == 0:
            el = time.time() - t0
            print(f"{i+1}/{len(files)}  {el/60:.1f} min  eta {el/(i+1)*(len(files)-i-1)/60:.1f} min", flush=True)
    df = pd.DataFrame(rows)
    out = ROOT / "data" / f"features_{''.join(a.views)}.csv"
    df.to_csv(out, index=False)
    print("saved", out, df.shape)


if __name__ == "__main__":
    main()
