"""1-NN classifier wrapper."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from . import Classifier


class KNNClassifier(Classifier):
    """1-nearest-neighbor classifier."""

    def __init__(self, n_neighbors: int = 1) -> None:
        self.n_neighbors = n_neighbors
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def fit(self, Z: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        self.model.fit(Z, y)
        return self

    def predict(self, Z: np.ndarray) -> np.ndarray:
        return self.model.predict(Z)

    def predict_proba(self, Z: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(Z)
