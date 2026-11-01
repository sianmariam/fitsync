"""Train YOLOv8 on the IndianFoodNet dataset (Roboflow export, "YOLOv8" format) and compare with the
published IndianFoodNet baseline (YOLOv8, mAP@50 = 0.807, mAP@50-95 = 0.628, 5 epochs).

Usage (GPU strongly recommended — use the Colab notebook in colab/ if you have no GPU):
  python scripts/train_indian_detector.py --zip IndianFoodNet.v1i.yolov8.zip --model yolov8s.pt --epochs 60

Outputs:
  models/indian_food_yolov8.pt          -> picked up automatically by the app
  results/indian_detector_metrics.json  -> used by scripts/add_indian_slide.py
"""
import argparse, json, shutil, zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
PAPER = {"model": "YOLOv8 (IndianFoodNet paper, 5 epochs)", "precision": 0.775, "recall": 0.719,
         "map50": 0.807, "map5095": 0.628}


def prepare(zip_path: Path, out: Path) -> Path:
    if zip_path.suffix == ".zip":
        out.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(out)
        root = out
    else:
        root = zip_path
    yml = next(root.rglob("data.yaml"))
    base = yml.parent
    d = yaml.safe_load(open(yml))
    for split, folder in (("train", "train"), ("val", "valid"), ("test", "test")):
        p = base / folder / "images"
        if p.exists():
            d[split] = str(p.resolve())
        elif split == "test":
            d.pop("test", None)
    d.pop("path", None)
    fixed = base / "data_fixed.yaml"
    yaml.safe_dump(d, open(fixed, "w"), sort_keys=False)
    print("classes:", d["names"])
    return fixed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help="Roboflow YOLOv8 export (.zip) or extracted folder")
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    from ultralytics import YOLO
    data = prepare(Path(a.zip), ROOT / "data" / "indianfoodnet")
    m = YOLO(a.model)
    m.train(data=str(data), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device, patience=20,
            cos_lr=True, close_mosaic=10, mixup=0.1, project=str(ROOT / "runs"), name="indian_det", exist_ok=True)
    best = Path(m.trainer.best)
    d = yaml.safe_load(open(data))
    split = "test" if "test" in d else "val"
    v = YOLO(str(best)).val(data=str(data), split=split, imgsz=a.imgsz, device=a.device)
    ours = dict(model=f"{Path(a.model).stem} ({a.epochs} epochs, ours)", precision=float(v.box.mp),
                recall=float(v.box.mr), map50=float(v.box.map50), map5095=float(v.box.map))
    per_class = {d["names"][i] if isinstance(d["names"], list) else d["names"][i]: float(x)
                 for i, x in zip(v.box.ap_class_index, v.box.ap50)}
    res = dict(dataset="IndianFoodNet (Roboflow)", split=split, n_classes=len(d["names"]), paper=PAPER, ours=ours,
               per_class_ap50=per_class)
    (ROOT / "results").mkdir(exist_ok=True)
    json.dump(res, open(ROOT / "results" / "indian_detector_metrics.json", "w"), indent=1)
    shutil.copy(best, ROOT / "models" / "indian_food_yolov8.pt")
    print(f"\nPaper YOLOv8 : mAP50={PAPER['map50']:.3f}  mAP50-95={PAPER['map5095']:.3f}")
    print(f"Ours         : mAP50={ours['map50']:.3f}  mAP50-95={ours['map5095']:.3f}  ({split} split)")


if __name__ == "__main__":
    main()
