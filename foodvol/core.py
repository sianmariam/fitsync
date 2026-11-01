"""Depth-aware food volume & calorie estimation — core pipeline.

Stages
  1. Detection      : YOLOv8 (fine-tuned on ECUSTFD) finds food items + reference coin
  2. Segmentation   : GrabCut refines each food box into a pixel mask
  3. Depth          : Depth Anything V2 (ViT-S, ONNX) predicts relative inverse depth
  4. Calibration    : coin of known diameter -> pixels/cm and camera distance;
                      robust plane fit on the table/plate around the food -> reference surface
  5. Volume         : integrate height above the reference surface over the food mask
  6. ML correction  : gradient-boosting model maps geometric features -> final volume
  7. Nutrition      : mass = volume x density ; kcal = mass x kcal/g (food table)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

# ----------------------------------------------------------------------------- depth

class DepthEstimator:
    """Depth Anything V2 small, exported to ONNX. Output = relative inverse depth (disparity)."""

    MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    STD = np.array([0.229, 0.224, 0.225], np.float32)

    def __init__(self, path: str | Path = MODELS / "da2_vits.onnx", threads: int = 0, size: int = 518):
        import onnxruntime as ort
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.size = size

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x = cv2.resize(rgb, (self.size, self.size), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
        x = ((x - self.MEAN) / self.STD).transpose(2, 0, 1)[None]
        d = self.sess.run(None, {self.inp: x})[0][0]
        d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
        return d.astype(np.float32)


# ----------------------------------------------------------------------------- detection

@dataclass
class Det:
    name: str
    box: tuple  # x1, y1, x2, y2 (pixels)
    conf: float = 1.0


def norm_name(n: str) -> str:
    return n.strip().lower().replace(" ", "_").replace("-", "_")


class Detector:
    """Several YOLO models run together: the ECUSTFD model supplies the reference coin (+19 foods), the
    32-dish Indian model (YOLO11n) and the idli/dosa/chutney model add Indian dishes."""

    def __init__(self, weights=MODELS / "food_coin_yolov8n.pt", conf: float = 0.25,
                 extra=((MODELS / "indian_food_yolo11n.pt", 512, 0.4, 0),
                        (MODELS / "indian_food_yolov8.pt", 640, 0.35, 0),
                        (MODELS / "idli_dosa_yolov8n.pt", 512, 0.75, 2))):
        from ultralytics import YOLO
        # (model, image size, confidence, priority): lower priority wins when boxes overlap
        self.models = [(YOLO(str(weights)), 416, conf, 1)]
        for w, sz, c, pr in extra:
            if Path(w).exists():
                self.models.append((YOLO(str(w)), sz, c, pr))
        self.conf = conf

    @property
    def n_models(self):
        return len(self.models)

    def __call__(self, bgr: np.ndarray) -> list[Det]:
        out, prio = [], {}
        for k, (model, sz, conf, pr) in enumerate(self.models):
            r = model.predict(bgr, conf=conf, verbose=False, imgsz=sz)[0]
            for b, c, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                name = norm_name(r.names[int(c)])
                if k == 0 and name != "coin" and s < 0.4:  # main model: low-confidence foods are usually look-alikes
                    continue
                out.append(Det(name, tuple(float(v) for v in b), float(s)))
                prio[id(out[-1])] = pr
        # when an Indian-dish model fires, weak fruit/snack guesses from the main model are usually look-alikes
        if any(prio[id(d)] != 1 for d in out):
            out = [d for d in out if prio[id(d)] != 1 or d.name == "coin" or d.conf >= 0.85]
        # keep best coin + cross-model de-duplication of foods (priority, then confidence)
        coins = sorted([d for d in out if d.name == "coin"], key=lambda d: -d.conf)[:1]
        foods = []
        for d in sorted([d for d in out if d.name != "coin"], key=lambda d: (prio[id(d)], -d.conf)):
            if all(iou(d.box, f.box) < 0.5 and overlap_small(d.box, f.box) < 0.6 for f in foods) \
                    and (not coins or iou(d.box, coins[0].box) < 0.5):
                foods.append(d)
        return coins + foods


def overlap_small(a, b):
    """Intersection over the smaller box: catches a plate-sized box drawn around an already-detected food."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / small if small > 0 else 0.0


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


# ----------------------------------------------------------------------------- segmentation

def segment_in_box(bgr: np.ndarray, box, iters: int = 4) -> np.ndarray:
    """GrabCut initialised with the detection box. Falls back to an inscribed ellipse."""
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    ell = np.zeros((h, w), np.uint8)
    cv2.ellipse(ell, ((x1 + x2) // 2, (y1 + y2) // 2), (max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2)), 0, 0, 360, 1, -1)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return ell.astype(bool)
    # work on a padded crop for speed
    pad = int(0.25 * max(x2 - x1, y2 - y1))
    cx1, cy1, cx2, cy2 = max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad)
    crop = bgr[cy1:cy2, cx1:cx2]
    mask = np.zeros(crop.shape[:2], np.uint8)
    rect = (x1 - cx1, y1 - cy1, x2 - x1, y2 - y1)
    try:
        bg, fg = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(crop, mask, rect, bg, fg, iters, cv2.GC_INIT_WITH_RECT)
        m = np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    except cv2.error:
        return ell.astype(bool)
    # keep largest component, fill holes
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:
        m = (lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    full = np.zeros((h, w), np.uint8)
    full[cy1:cy2, cx1:cx2] = m
    box_area = (x2 - x1) * (y2 - y1)
    # sanity check: GrabCut sometimes collapses or floods
    if full.sum() < 0.25 * box_area or full.sum() > 1.05 * box_area:
        return ell.astype(bool)
    return full.astype(bool)


# ----------------------------------------------------------------------------- geometry

@dataclass
class Calib:
    px_per_cm: float
    cam_dist_cm: float
    f_px: float


def calibrate_from_coin(coin_box, img_w: int, coin_diam_cm: float = 2.5, hfov_deg: float = 66.0) -> Calib:
    """Coin seen roughly from above -> its major axis gives the diameter in pixels.
    Pinhole model: Z = f * D_real / D_px  with f from a typical phone horizontal FOV."""
    w = coin_box[2] - coin_box[0]
    h = coin_box[3] - coin_box[1]
    d_px = max(w, h)
    px_per_cm = d_px / coin_diam_cm
    f_px = (img_w / 2) / math.tan(math.radians(hfov_deg) / 2)
    return Calib(px_per_cm, f_px * coin_diam_cm / d_px, f_px)


def fit_reference_plane(disp: np.ndarray, region: np.ndarray):
    """Robust least-squares plane  d = a*x + b*y + c  over reference pixels (table / plate)."""
    ys, xs = np.nonzero(region)
    if len(xs) < 50:
        return None
    if len(xs) > 20000:
        idx = np.random.default_rng(0).choice(len(xs), 20000, replace=False)
        xs, ys = xs[idx], ys[idx]
    z = disp[ys, xs]
    A = np.c_[xs, ys, np.ones_like(xs)].astype(np.float64)
    keep = np.ones(len(z), bool)
    coef = None
    for _ in range(3):  # iteratively drop outliers (other objects, plate rim, shadows)
        coef, *_ = np.linalg.lstsq(A[keep], z[keep], rcond=None)
        r = z - A @ coef
        s = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep]))) + 1e-6
        keep = np.abs(r) < 2.5 * s
    return coef


def height_map(disp, food_mask, coin_box, calib: Calib, ring_frac: float = 0.6):
    """Height (cm) of each food pixel above the local reference surface."""
    h, w = disp.shape
    ys, xs = np.nonzero(food_mask)
    x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
    pad = int(ring_frac * max(x2 - x1, y2 - y1))
    region = np.zeros_like(food_mask)
    region[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)] = True
    grown = cv2.dilate(food_mask.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    region &= ~grown
    if coin_box is not None:  # coin lies on the table: good reference pixels, keep them
        cx1, cy1, cx2, cy2 = [int(v) for v in coin_box]
        region[max(0, cy1):cy2, max(0, cx1):cx2] = True
    coef = fit_reference_plane(disp, region)
    if coef is None:
        base = np.full_like(disp, np.median(disp[~grown]))
    else:
        yy, xx = np.mgrid[0:h, 0:w]
        base = (coef[0] * xx + coef[1] * yy + coef[2]).astype(np.float32)
    # disparity ~ 1/Z  ->  Z = Z0 * base/disp ;  height = Z0 - Z
    d = np.maximum(disp, 1e-6)
    b = np.maximum(base, 1e-6)
    hgt = calib.cam_dist_cm * (1.0 - b / d)
    hgt = np.clip(hgt, 0, None) * food_mask
    # also a scale-free relative height (disparity excess), used as an ML feature
    rel = np.clip((d - b) / b, 0, None) * food_mask
    return hgt.astype(np.float32), rel.astype(np.float32)


@dataclass
class ItemFeatures:
    name: str
    area_cm2: float
    major_cm: float
    minor_cm: float
    h_mean_cm: float
    h_p90_cm: float
    h_max_cm: float
    rel_mean: float
    rel_p90: float
    vol_depth_cm3: float      # direct depth integration
    vol_ellipsoid_cm3: float  # shape prior: ellipsoid from 2D axes (what area-only methods do)
    coin_px: float
    box_fill: float
    view: str = "T"

    def vector(self):
        return [self.area_cm2, self.major_cm, self.minor_cm, self.h_mean_cm, self.h_p90_cm, self.h_max_cm,
                self.rel_mean, self.rel_p90, self.vol_depth_cm3, self.vol_ellipsoid_cm3, self.box_fill,
                math.log1p(self.vol_depth_cm3), math.log1p(self.vol_ellipsoid_cm3),
                self.area_cm2 * self.h_p90_cm]


FEATURE_NAMES = ["area_cm2", "major_cm", "minor_cm", "h_mean_cm", "h_p90_cm", "h_max_cm", "rel_mean", "rel_p90",
                 "vol_depth_cm3", "vol_ellipsoid_cm3", "box_fill", "log_vol_depth", "log_vol_ellipsoid", "area_x_h90"]


def item_features(name, mask, hgt, rel, calib: Calib, box, view="T") -> ItemFeatures:
    px_area = 1.0 / calib.px_per_cm ** 2
    area = float(mask.sum()) * px_area
    pts = np.column_stack(np.nonzero(mask)[::-1]).astype(np.float32)
    if len(pts) >= 5:
        (_, _), (a, b), _ = cv2.minAreaRect(pts)
        major, minor = max(a, b) / calib.px_per_cm, min(a, b) / calib.px_per_cm
    else:
        major = minor = math.sqrt(area)
    hv = hgt[mask]
    rv = rel[mask]
    # ellipsoid with semi-axes (major/2, minor/2, minor/2): the classic 2D shape prior
    v_ell = 4 / 3 * math.pi * (major / 2) * (minor / 2) * (minor / 2)
    bw, bh = box[2] - box[0], box[3] - box[1]
    return ItemFeatures(
        name=name, area_cm2=area, major_cm=major, minor_cm=minor,
        h_mean_cm=float(hv.mean()) if hv.size else 0.0,
        h_p90_cm=float(np.percentile(hv, 90)) if hv.size else 0.0,
        h_max_cm=float(hv.max()) if hv.size else 0.0,
        rel_mean=float(rv.mean()) if rv.size else 0.0,
        rel_p90=float(np.percentile(rv, 90)) if rv.size else 0.0,
        vol_depth_cm3=float(hgt.sum() * px_area),
        vol_ellipsoid_cm3=v_ell,
        coin_px=calib.px_per_cm * 2.5,
        box_fill=float(mask.sum() / max(1.0, bw * bh)),
        view=view,
    )


# ----------------------------------------------------------------------------- end-to-end

@dataclass
class ItemResult:
    name: str
    conf: float
    box: tuple
    volume_cm3: float
    volume_depth_raw_cm3: float
    mass_g: float
    kcal: float
    protein_g: float
    carbs_g: float
    fat_g: float
    features: dict = field(default_factory=dict)


class FoodVolumePipeline:
    def __init__(self, detector_weights=MODELS / "food_coin_yolov8n.pt", depth_model=MODELS / "da2_vits.onnx",
                 regressor=MODELS / "volume_regressor.joblib", coin_diam_cm: float = 2.5):
        from .nutrition import NutritionDB
        self.detector = Detector(detector_weights) if Path(detector_weights).exists() else None
        self.depth = DepthEstimator(depth_model)
        self.coin_diam_cm = coin_diam_cm
        self.reg = None
        if Path(regressor).exists():
            import joblib
            self.reg = joblib.load(regressor)
        self.db = NutritionDB()

    def predict_volume(self, f: ItemFeatures) -> float:
        if self.reg is None:
            return f.vol_depth_cm3
        return float(self.reg.predict(f))

    def run(self, bgr, extra_dets: list[Det] | None = None, coin_diam_cm=None, hfov_deg=66.0,
            allow_no_coin: bool = True, assumed_dist_cm: float = 35.0):
        coin_diam_cm = coin_diam_cm or self.coin_diam_cm
        dets = self.detector(bgr) if self.detector else []
        if extra_dets:  # e.g. boxes proposed by Gemini for foods the detector doesn't know
            for d in extra_dets:
                if all(iou(d.box, e.box) < 0.5 for e in dets if e.name != "coin"):
                    dets.append(d)
        coin = next((d for d in dets if d.name == "coin"), None)
        foods = [d for d in dets if d.name != "coin"]
        disp = self.depth(bgr)
        approx = False
        if coin is None:
            if not allow_no_coin or not foods:
                return dict(error="No reference coin found. Place a coin next to the food and retake the photo.",
                            dets=dets, disp=disp, items=[], masks=[], hgt=None)
            # fallback: assume the phone was held ~35 cm above the food (pinhole model) -> approximate scale
            f_px = (bgr.shape[1] / 2) / math.tan(math.radians(hfov_deg) / 2)
            calib = Calib(f_px / assumed_dist_cm, assumed_dist_cm, f_px)
            approx = True
        else:
            calib = calibrate_from_coin(coin.box, bgr.shape[1], coin_diam_cm, hfov_deg)
        items, masks = [], []
        hsum = np.zeros(disp.shape, np.float32)
        for d in foods:
            m = segment_in_box(bgr, d.box)
            hgt, rel = height_map(disp, m, coin.box if coin else None, calib)
            f = item_features(d.name, m, hgt, rel, calib, d.box)
            vol = self.predict_volume(f)
            n = self.db.nutrients(d.name, vol)
            items.append(ItemResult(d.name, d.conf, d.box, vol, f.vol_depth_cm3, **n, features=asdict(f)))
            masks.append(m)
            hsum = np.maximum(hsum, hgt)
        return dict(error=None, dets=dets, coin=coin, calib=calib, disp=disp, hgt=hsum, items=items, masks=masks,
                    approx=approx)
