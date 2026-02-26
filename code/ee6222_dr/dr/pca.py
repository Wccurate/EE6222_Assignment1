"""PCA dimensionality reduction wrapper."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from . import DRMethod


class PCADR(DRMethod):
    """PCA wrapper implementing DRMethod API."""

    def __init__(self, n_components: int, whiten: bool = False) -> None:
        self.n_components = n_components
        self.whiten = whiten
        self.model: PCA | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "PCADR":
        self.model = PCA(n_components=self.n_components, whiten=self.whiten)
        self.model.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("PCA model is not fitted")
        return self.model.transform(X)

    @property
    def components_(self) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("PCA model is not fitted")
        return self.model.components_
