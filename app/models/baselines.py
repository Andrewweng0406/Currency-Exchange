from __future__ import annotations

import numpy as np


class ConstantProbabilityModel:
    def __init__(self, probability: float = 0.5, learn_rate: bool = False):
        self.probability = probability
        self.learn_rate = learn_rate

    def fit(self, x, y):
        if self.learn_rate:
            self.probability = float(np.mean(y)) if len(y) else 0.5
        return self

    def predict_proba(self, x):
        p = np.full(len(x), self.probability, dtype=float)
        return np.column_stack((1 - p, p))
