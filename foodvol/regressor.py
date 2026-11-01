"""Learned volume-correction models: geometric/depth features -> volume (cm^3).

Two variants
  * class-agnostic  : never sees the food name, so it works for foods outside the training set
                      (e.g. Indian dishes named by Gemini).
  * with class prior: also gets log(typical volume of that food) as a feature — recognition gives a prior,
                      measurement adjusts it. Used when the food is one the model has statistics for.
Both learn to correct systematic errors of monocular depth (scale/shift ambiguity, hidden underside,
segmentation leaks)."""
from __future__ import annotations

import math

import numpy as np

from .core import FEATURE_NAMES, ItemFeatures


def make_model():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.04, max_leaf_nodes=15,
                                         min_samples_leaf=20, l2_regularization=1.0, random_state=0)


class VolumeRegressor:
    def __init__(self, priors: dict | None = None):
        self.priors = priors          # food name -> typical volume (cm^3); None = class-agnostic
        self.model = None
        self.agnostic = None          # fallback for unknown foods when priors are used

    def _X(self, X, names):
        if self.priors is None:
            return X
        p = np.array([math.log(self.priors.get(n, np.nan)) if n in self.priors else np.nan for n in names])
        return np.c_[X, p]

    def fit(self, X: np.ndarray, y_volume: np.ndarray, names=None):
        y = np.log(y_volume)          # log target -> optimises relative error
        self.model = make_model().fit(self._X(X, names), y)
        if self.priors is not None:
            self.agnostic = make_model().fit(X, y)
        return self

    def predict_matrix(self, X: np.ndarray, names=None) -> np.ndarray:
        if self.priors is None:
            return np.exp(self.model.predict(X))
        out = np.exp(self.model.predict(self._X(X, names)))
        unknown = np.array([n not in self.priors for n in names])
        if unknown.any():
            out[unknown] = np.exp(self.agnostic.predict(X[unknown]))
        return out

    def predict(self, f: ItemFeatures) -> float:
        X = np.array([f.vector()], dtype=np.float64)
        return float(self.predict_matrix(X, [f.name])[0])
