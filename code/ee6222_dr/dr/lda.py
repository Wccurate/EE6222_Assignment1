"""LDA dimensionality reduction wrapper."""

from __future__ import annotations

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from . import DRMethod


class LDADR(DRMethod):
    """LDA wrapper implementing DRMethod API."""

    def __init__(self, n_components: int, shrinkage: str | float | None = None) -> None:
        self.n_components = n_components
        self.shrinkage = shrinkage
        self.model: LinearDiscriminantAnalysis | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "LDADR":
        if y is None:
            raise ValueError("LDA requires labels y")

        if self.shrinkage in (None, "none"):
            self.model = LinearDiscriminantAnalysis(
                n_components=self.n_components,
                solver="svd",
            )
        else:
            shrinkage = self.shrinkage
            if isinstance(shrinkage, str) and shrinkage.lower() == "auto":
                shrinkage = "auto"
            self.model = LinearDiscriminantAnalysis(
                n_components=self.n_components,
                solver="lsqr",
                shrinkage=shrinkage,
            )
        self.model.fit(X, y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("LDA model is not fitted")
        return self.model.transform(X)
