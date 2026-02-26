"""ICA dimensionality reduction wrapper."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import FastICA

from . import DRMethod


class ICADR(DRMethod):
    """FastICA wrapper implementing DRMethod API."""

    def __init__(
        self,
        n_components: int,
        fun: str = "logcosh",
        max_iter: int = 500,
        tol: float = 1e-4,
        random_state: int = 0,
    ) -> None:
        self.n_components = n_components
        self.fun = fun
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.model: FastICA | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "ICADR":
        self.model = FastICA(
            n_components=self.n_components,
            fun=self.fun,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            whiten="unit-variance",
        )
        self.model.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("ICA model is not fitted")
        return self.model.transform(X)
