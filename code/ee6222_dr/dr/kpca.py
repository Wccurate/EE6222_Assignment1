"""Kernel PCA dimensionality reduction wrapper."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import KernelPCA

from . import DRMethod


class KernelPCADR(DRMethod):
    """RBF KernelPCA wrapper implementing DRMethod API."""

    def __init__(self, n_components: int, gamma: float = 1e-3) -> None:
        self.n_components = n_components
        self.gamma = gamma
        self.model: KernelPCA | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "KernelPCADR":
        self.model = KernelPCA(
            n_components=self.n_components,
            kernel="rbf",
            gamma=self.gamma,
            fit_inverse_transform=False,
        )
        self.model.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("KernelPCA model is not fitted")
        return self.model.transform(X)
