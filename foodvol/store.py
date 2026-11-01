"""Tiny JSON store so meals, workouts and weight persist between app sessions (data/user_data.json)."""
from __future__ import annotations

import base64
import datetime as dt
import json
import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PATH = ROOT / "data" / "user_data.json"

DEFAULT = dict(profile=dict(name="Sian", age=21, sex="Female", height=160, weight=55, activity="Lightly active",
                            goal="Maintain", days=4, level="Beginner", equip="None (home)"),
               meals=[], workouts=[], weights=[])


def today() -> str:
    return dt.date.today().isoformat()


def _path(user=None) -> Path:
    if user:
        return ROOT / "data" / "users" / f"{user}.json"
    return PATH


def load(user: str | None = None, name: str | None = None) -> dict:
    path = _path(user)
    if path.exists():
        try:
            d = json.load(open(path))
            for k, v in DEFAULT.items():
                d.setdefault(k, v if not isinstance(v, dict) else dict(v))
            for k, v in DEFAULT["profile"].items():
                d["profile"].setdefault(k, v)
            d["_user"] = user
            return d
        except Exception:  # noqa
            pass
    d = json.loads(json.dumps(DEFAULT))
    if user:
        d["profile"]["name"] = name or user
    d["_user"] = user
    return d


def save(d: dict):
    path = _path(d.get("_user"))
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(d, open(path, "w"), indent=1)


def thumb(bgr, size=160) -> str:
    h, w = bgr.shape[:2]
    s = size / max(h, w)
    small = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf.tobytes()).decode()


def add_meal(d, meal_type, items, bgr=None, date=None):
    d["meals"].append(dict(date=date or today(), time=dt.datetime.now().strftime("%H:%M"), meal=meal_type,
                           items=items, thumb=thumb(bgr) if bgr is not None else None,
                           kcal=sum(i["kcal"] for i in items), protein=sum(i["protein"] for i in items),
                           carbs=sum(i["carbs"] for i in items), fat=sum(i["fat"] for i in items)))
    save(d)


def add_workout(d, name, minutes, kcal, date=None):
    d["workouts"].append(dict(date=date or today(), name=name, minutes=minutes, kcal=kcal))
    save(d)


def remove_workout(d, name, date=None):
    date = date or today()
    d["workouts"] = [w for w in d["workouts"] if not (w["date"] == date and w["name"] == name)]
    save(d)


def log_weight(d, kg, date=None):
    date = date or today()
    d["weights"] = [w for w in d["weights"] if w["date"] != date] + [dict(date=date, kg=kg)]
    d["weights"].sort(key=lambda w: w["date"])
    save(d)


def seed_demo_week(d, kcal_target, weight):
    """Fill the previous 6 days with sample entries (marked demo=True) so the dashboard can be shown."""
    rng = random.Random(7)
    menu = {"Breakfast": [("idli", 3 * 40, 130, 4, 27, .5), ("dosa", 120, 168, 3.9, 29, 3.7), ("poha", 180, 180, 3, 30, 5),
                          ("banana", 120, 89, 1.1, 22.8, .3), ("egg", 55, 143, 12.6, .7, 9.5)],
            "Lunch": [("white_rice", 200, 130, 2.7, 28.2, .3), ("dal", 180, 116, 7, 16, 2.5),
                      ("rajma_curry", 180, 120, 6, 15, 4), ("chapati", 80, 297, 9.8, 46, 7.5),
                      ("palak_paneer", 150, 160, 7.5, 6, 12)],
            "Snack": [("apple", 220, 52, .3, 13.8, .2), ("samosa", 90, 308, 5, 32, 18), ("chai", 150, 45, 1.5, 6.5, 1.5),
                      ("orange", 180, 47, .9, 11.8, .1)],
            "Dinner": [("chapati", 80, 297, 9.8, 46, 7.5), ("chicken_curry", 200, 150, 13, 4, 9),
                       ("biryani", 250, 170, 6.5, 24, 5.5), ("dal", 150, 116, 7, 16, 2.5)]}
    d["meals"] = [m for m in d["meals"] if not m.get("demo")]
    d["workouts"] = [w for w in d["workouts"] if not w.get("demo")]
    d["weights"] = [w for w in d["weights"] if not w.get("demo")]
    base = dt.date.today()
    for back in range(6, 0, -1):
        day = (base - dt.timedelta(days=back)).isoformat()
        scale = kcal_target / 1500 * rng.uniform(0.92, 1.12)
        for meal, opts in menu.items():
            picks = rng.sample(opts, 2 if meal != "Snack" else 1)
            items = []
            for n, g, kc, p, c, f in picks:
                g = g * scale * rng.uniform(0.8, 1.2)
                items.append(dict(food=n, grams=round(g), kcal=round(g * kc / 100), protein=round(g * p / 100, 1),
                                  carbs=round(g * c / 100, 1), fat=round(g * f / 100, 1)))
            d["meals"].append(dict(date=day, time={"Breakfast": "08:30", "Lunch": "13:15", "Snack": "17:00",
                                                   "Dinner": "20:30"}[meal], meal=meal, items=items, thumb=None,
                                   kcal=sum(i["kcal"] for i in items), protein=sum(i["protein"] for i in items),
                                   carbs=sum(i["carbs"] for i in items), fat=sum(i["fat"] for i in items), demo=True))
        if rng.random() < 0.7:
            mins = rng.choice([25, 30, 40, 45])
            d["workouts"].append(dict(date=day, name=rng.choice(["Workout plan session", "Brisk walking", "Jogging"]),
                                      minutes=mins, kcal=round(mins * rng.uniform(5, 8)), demo=True))
        d["weights"].append(dict(date=day, kg=round(weight + back * 0.05 + rng.uniform(-.12, .12), 1), demo=True))
    d["weights"].sort(key=lambda w: w["date"])
    save(d)


def clear_demo(d):
    for k in ("meals", "workouts", "weights"):
        d[k] = [x for x in d[k] if not x.get("demo")]
    save(d)
