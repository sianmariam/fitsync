"""Train the ML volume-correction model and compare it against baselines on held-out food items.

Methods compared (all predict volume; mass = volume x density, kcal = mass x kcal/g):
  B0  Typical-portion guess  : average volume of that food type in the training set.
                               (This is effectively what an LLM like Gemini does when it 'estimates portion size'
                                from appearance — it recalls a typical serving.)
  B1  2D area + shape prior  : ellipsoid from the segmented outline + coin scale (area-based methods in literature)
  B2  Raw monocular depth    : integrate Depth Anything V2 height map over the mask (single global scale fitted)
  M1  Depth + ML (food-agnostic): gradient boosting on depth + geometry features, never told the food name
  OURS Depth + ML + food prior : same, plus log(typical volume of the recognised food) as a feature —
                                 recognition gives the prior, measurement adjusts it to the actual portion

Evaluation is item-level: the test foods are physical items never seen in training.
We report both the fixed test split and 5-fold GroupKFold cross-validation over all items.

Usage: python scripts/train_and_evaluate.py
"""
import json, sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from foodvol.core import FEATURE_NAMES  # noqa
from foodvol.nutrition import NutritionDB  # noqa
from foodvol.regressor import VolumeRegressor  # noqa

OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
METHODS = ["B0 Typical portion", "B1 2D area + shape", "B2 Raw depth", "M1 Depth + ML (agnostic)",
           "OURS Depth + ML + prior"]


def feats(df):
    d = df.copy()
    d["log_vol_depth"] = np.log1p(d.vol_depth_cm3)
    d["log_vol_ellipsoid"] = np.log1p(d.vol_ellipsoid_cm3)
    d["area_x_h90"] = d.area_cm2 * d.h_p90_cm
    return d[FEATURE_NAMES].to_numpy(np.float64)


def predict_all(tr, te):
    P = {}
    cls_mean = tr.groupby("name").gt_volume_cm3.mean()
    P[METHODS[0]] = te.name.map(cls_mean).fillna(tr.gt_volume_cm3.mean()).to_numpy()
    s1 = np.median(tr.gt_volume_cm3 / tr.vol_ellipsoid_cm3.clip(lower=1e-3))
    P[METHODS[1]] = te.vol_ellipsoid_cm3.to_numpy() * s1
    s2 = np.median(tr.gt_volume_cm3 / tr.vol_depth_cm3.clip(lower=1e-3))
    P[METHODS[2]] = te.vol_depth_cm3.to_numpy() * s2
    reg = VolumeRegressor().fit(feats(tr), tr.gt_volume_cm3.to_numpy())
    P[METHODS[3]] = reg.predict_matrix(feats(te))
    regp = VolumeRegressor(priors=cls_mean.to_dict()).fit(feats(tr), tr.gt_volume_cm3.to_numpy(), list(tr.name))
    P[METHODS[4]] = regp.predict_matrix(feats(te), list(te.name))
    return P, regp, dict(shape_scale=float(s1), depth_scale=float(s2))


def within_food_spearman(df, pred):
    """Does the method rank bigger portions of the SAME food as bigger? (item-level, foods with >=4 items)"""
    from scipy.stats import spearmanr
    d = pd.DataFrame({"name": df.name.values, "item": df.item.values, "gt": df.gt_volume_cm3.values, "p": pred})
    g = d.groupby(["name", "item"]).agg(gt=("gt", "first"), p=("p", "mean")).reset_index()
    rs = []
    for n, x in g.groupby("name"):
        if len(x) >= 4 and x["gt"].nunique() > 1 and x["p"].nunique() > 1:
            rs.append(spearmanr(x["gt"], x["p"]).correlation)
        elif len(x) >= 4 and x["gt"].nunique() > 1:
            rs.append(0.0)
    return float(np.nanmean(rs)) if rs else float("nan")


def metrics(pred, gt):
    ape = np.abs(pred - gt) / gt * 100
    return dict(MAPE=float(ape.mean()), MedAPE=float(np.median(ape)), within20=float((ape <= 20).mean() * 100),
                MAE=float(np.abs(pred - gt).mean()))


def main():
    df = pd.read_csv(ROOT / "data" / "features_T.csv")
    df = df[(df.area_cm2 > 0.5) & np.isfinite(df.vol_depth_cm3)].reset_index(drop=True)
    db = NutritionDB()
    # measured densities from training items only (g/cm^3)
    trv = df[df.split != "test"]
    dens = (trv.groupby("name").gt_weight_g.sum() / trv.groupby("name").gt_volume_cm3.sum()).to_dict()
    kcal_g = {k: db.lookup(k)["kcal_100g"] / 100 for k in df.name.unique()}

    def to_mass(names, vol):
        return vol * names.map(dens).to_numpy()

    report = {"n_images": int(len(df)), "n_items": int(df.item.nunique()), "n_food_types": int(df.name.nunique())}

    # ---- fixed held-out split
    tr, te = df[df.split != "test"], df[df.split == "test"].reset_index(drop=True)
    P, reg, scales = predict_all(tr, te)
    res = {}
    for m, v in P.items():
        mass = to_mass(te.name, v)
        kc = mass * te.name.map(kcal_g).to_numpy()
        gt_kc = te.gt_weight_g.to_numpy() * te.name.map(kcal_g).to_numpy()
        res[m] = dict(volume=metrics(v, te.gt_volume_cm3.to_numpy()),
                      mass=metrics(mass, te.gt_weight_g.to_numpy()),
                      kcal=metrics(kc, gt_kc))
    report["test_split"] = dict(n_images=int(len(te)), n_items=int(te.item.nunique()), results=res)
    pd.DataFrame({"image": te.image, "food": te.name, "gt_volume": te.gt_volume_cm3, "gt_weight": te.gt_weight_g,
                  **{m: P[m] for m in METHODS}}).to_csv(OUT / "test_predictions.csv", index=False)

    # ---- 5-fold GroupKFold CV over all items
    from sklearn.model_selection import GroupKFold
    cv_pred = {m: np.zeros(len(df)) for m in METHODS}
    for tri, tei in GroupKFold(5).split(df, groups=df.item):
        Pk, _, _ = predict_all(df.iloc[tri], df.iloc[tei].reset_index(drop=True))
        for m in METHODS:
            cv_pred[m][tei] = Pk[m]
    cv = {}
    for m in METHODS:
        mass = to_mass(df.name, cv_pred[m])
        cv[m] = dict(volume=metrics(cv_pred[m], df.gt_volume_cm3.to_numpy()),
                     mass=metrics(mass, df.gt_weight_g.to_numpy()),
                     within_food_spearman=within_food_spearman(df, cv_pred[m]))
    report["cv5"] = cv
    # foods whose portion size really varies (coefficient of variation of item volume >= 0.25)
    iv = df.groupby(["name", "item"]).gt_volume_cm3.first()
    cvar = (iv.groupby("name").std() / iv.groupby("name").mean())
    varying = sorted(cvar[cvar >= 0.25].index)
    sel = df.name.isin(varying).to_numpy()
    report["size_varying_foods"] = dict(foods=varying, n_images=int(sel.sum()), cv5={
        m: metrics(to_mass(df.name[sel], cv_pred[m][sel]), df.gt_weight_g.to_numpy()[sel]) for m in METHODS})

    # per-food breakdown (CV, mass MAPE) for OURS vs typical portion
    per = []
    for name, g in df.groupby("name"):
        idx = g.index.to_numpy()
        row = dict(food=name, n=len(idx))
        for m in METHODS:
            mass = cv_pred[m][idx] * dens[name] if name in dens else cv_pred[m][idx]
            row[m] = float((np.abs(mass - g.gt_weight_g) / g.gt_weight_g * 100).mean())
        per.append(row)
    pd.DataFrame(per).round(1).to_csv(OUT / "per_food_cv_mass_mape.csv", index=False)

    # permutation importance of the final model on test split
    from sklearn.inspection import permutation_importance
    pi = permutation_importance(reg.agnostic, feats(te), np.log(te.gt_volume_cm3), n_repeats=10, random_state=0)
    report["feature_importance"] = dict(sorted(zip(FEATURE_NAMES, pi.importances_mean.round(4).tolist()),
                                               key=lambda kv: -kv[1]))
    report["scales"] = scales

    # ---- final model on ALL data for the app
    priors = df.groupby("name").gt_volume_cm3.mean().round(1).to_dict()
    final = VolumeRegressor(priors=priors).fit(feats(df), df.gt_volume_cm3.to_numpy(), list(df.name))
    joblib.dump(final, ROOT / "models" / "volume_regressor.joblib")
    full_dens = (df.groupby("name").gt_weight_g.sum() / df.groupby("name").gt_volume_cm3.sum()).round(3).to_dict()
    json.dump(full_dens, open(ROOT / "models" / "densities.json", "w"), indent=1)
    json.dump(report, open(OUT / "metrics.json", "w"), indent=1)

    def show(title, r, keys=("volume", "mass", "kcal")):
        print(f"\n== {title}")
        print(f"{'method':28s}" + "".join(f"{k+' MAPE':>13s}{k+' ≤20%':>12s}" for k in keys if k in next(iter(r.values()))))
        for m, v in r.items():
            print(f"{m:28s}" + "".join(f"{v[k]['MAPE']:13.1f}{v[k]['within20']:12.1f}" for k in keys if k in v))
    print(json.dumps({k: report[k] for k in ("n_images", "n_items", "n_food_types")}))
    show(f"Held-out test items ({len(te)} images, {te.item.nunique()} items)", res)
    show("5-fold GroupKFold CV", cv)
    print("\nwithin-food rank correlation (CV):", {m: round(cv[m]["within_food_spearman"], 3) for m in METHODS})
    sv = report["size_varying_foods"]
    print("size-varying foods", sv["foods"], sv["n_images"], {m: round(v["MAPE"], 1) for m, v in sv["cv5"].items()})
    print("\nfeature importance:", list(report["feature_importance"].items())[:6])


if __name__ == "__main__":
    main()
