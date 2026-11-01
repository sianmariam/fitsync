"""Food density + nutrient table.

density_g_cm3 : for ECUSTFD foods, computed from the dataset's measured weight/volume (see scripts/build_features.py
                which overwrites these with exact dataset means into models/densities.json).
nutrients     : approximate values per 100 g (USDA FoodData Central / IFCT 2017 style). Before publishing results,
                cross-check the Indian items against the Indian Food Composition Tables (IFCT 2017, NIN Hyderabad).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "nutrition_table.csv"
DENS_JSON = ROOT / "models" / "densities.json"


class NutritionDB:
    def __init__(self, path: Path = CSV):
        self.rows = {}
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                key = r["food"].strip().lower()
                self.rows[key] = {k: (float(v) if k != "food" and k != "source" else v) for k, v in r.items()}
        if DENS_JSON.exists():  # measured densities from the dataset override the table
            for k, v in json.load(open(DENS_JSON)).items():
                if k in self.rows:
                    self.rows[k]["density_g_cm3"] = v

    def lookup(self, name: str):
        k = name.strip().lower().replace(" ", "_").replace("-", "_")
        if k in self.rows:
            return self.rows[k]
        squash = k.replace("_", "")  # "AlooGobi" -> "aloogobi" == "aloo_gobi"
        for key in self.rows:
            if key.replace("_", "") == squash:
                return self.rows[key]
        for key in sorted(self.rows, key=len, reverse=True):  # loose match ("plain dosa" -> "dosa")
            if len(key) >= 3 and (key in k or k in key):
                return self.rows[key]
        return None

    def add(self, name, density, kcal, protein, carbs, fat, source="gemini"):
        self.rows[name.strip().lower().replace(" ", "_")] = dict(food=name, density_g_cm3=density, kcal_100g=kcal,
                                                                  protein_100g=protein, carbs_100g=carbs,
                                                                  fat_100g=fat, source=source)

    def nutrients(self, name: str, volume_cm3: float):
        r = self.lookup(name) or dict(density_g_cm3=0.8, kcal_100g=150, protein_100g=5, carbs_100g=20, fat_100g=5)
        mass = volume_cm3 * r["density_g_cm3"]
        f = mass / 100.0
        return dict(mass_g=mass, kcal=f * r["kcal_100g"], protein_g=f * r["protein_100g"],
                    carbs_g=f * r["carbs_100g"], fat_g=f * r["fat_100g"])
