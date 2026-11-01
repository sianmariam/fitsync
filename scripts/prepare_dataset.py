"""Convert ECUSTFD (Pascal VOC boxes) to YOLO format with an item-level train/val/test split.

Split is done by FOOD ITEM (e.g. apple001), not by image, so the same physical
food never appears in both train and test (no data leakage).

Usage:  python scripts/prepare_dataset.py --src data/ECUSTFD --out data/yolo
"""
import argparse, json, random, re, shutil
import xml.etree.ElementTree as ET
from pathlib import Path

CLASSES = ["coin", "apple", "banana", "bread", "bun", "doughnut", "egg",
           "fired_dough_twist", "grape", "lemon", "litchi", "mango", "mooncake",
           "orange", "peach", "pear", "plum", "qiwi", "sachima", "tomato"]
ITEM_RE = re.compile(r"^([a-z_]+\d{3})([ST])\((\d+)\)")
BAD = {"mix002T(2)", "mix005S(4)"}  # no calibration coin (see dataset README)


def item_id(stem):
    m = ITEM_RE.match(stem)
    return m.group(1) if m else None


def split_items(src, seed=42, frac=(0.7, 0.1, 0.2)):
    items = sorted({item_id(p.stem) for p in (src / "Annotations").glob("*.xml")} - {None})
    rng = random.Random(seed)
    # stratify by food type so every class is in every split
    by_type = {}
    for it in items:
        by_type.setdefault(re.sub(r"\d+$", "", it), []).append(it)
    split = {}
    for t, its in by_type.items():
        rng.shuffle(its)
        n = len(its)
        n_te = max(1, round(n * frac[2]))
        n_va = max(1, round(n * frac[1])) if n > 3 else 0
        for i, it in enumerate(its):
            split[it] = "test" if i < n_te else ("val" if i < n_te + n_va else "train")
    return split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/ECUSTFD")
    ap.add_argument("--out", default="data/yolo")
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    split = split_items(src)
    (out.parent).mkdir(parents=True, exist_ok=True)
    json.dump(split, open(out.parent / "item_split.json", "w"), indent=1)
    for s in ("train", "val", "test"):
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)
    n = 0
    for xf in sorted((src / "Annotations").glob("*.xml")):
        if xf.stem in BAD:
            continue
        it = item_id(xf.stem)
        if it is None:
            continue
        s = split[it]
        r = ET.parse(xf).getroot()
        W = float(r.find("size/width").text); H = float(r.find("size/height").text)
        lines = []
        for o in r.findall("object"):
            name = o.find("name").text.strip()
            if name not in CLASSES:
                continue
            b = o.find("bndbox")
            x1, y1, x2, y2 = (float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
            lines.append(f"{CLASSES.index(name)} {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}")
        img = src / "JPEGImages" / (xf.stem + ".JPG")
        if not img.exists():
            continue
        dst = out / "images" / s / (xf.stem + ".jpg")
        if not dst.exists():
            try:
                dst.symlink_to(img.resolve())
            except OSError:
                shutil.copy(img, dst)
        (out / "labels" / s / (xf.stem + ".txt")).write_text("\n".join(lines))
        n += 1
    (out / "ecustfd.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
    print("images:", n, {s: sum(1 for v in split.values() if v == s) for s in ("train", "val", "test")}, "items")


if __name__ == "__main__":
    main()
