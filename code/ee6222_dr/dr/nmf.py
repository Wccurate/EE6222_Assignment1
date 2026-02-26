"""NMF dimensionality reduction wrapper."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import NMF

from . import DRMethod


class NMFDR(DRMethod):
    """NMF wrapper implementing DRMethod API."""

    def __init__(
        self,
        n_components: int,
        init: str = "nndsvd",
        solver: str = "cd",
        max_iter: int = 300,
        random_state: int = 0,
    ) -> None:
        self.n_components = n_components
        self.init = init
        self.solver = solver
        self.max_iter = max_iter
        self.random_state = random_state
        self.model: NMF | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "NMFDR":
        self.model = NMF(
            n_components=self.n_components,
            init=self.init,
            solver=self.solver,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self.model.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("NMF model is not fitted")
        return self.model.transform(X)

    @property
    def components_(self) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("NMF model is not fitted")
        return self.model.components_
