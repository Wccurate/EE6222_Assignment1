"""PCA to LDA two-stage dimensionality reduction wrapper."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from . import DRMethod


class PCALDADR(DRMethod):
    """Two-stage PCA->LDA DR wrapper."""

    def __init__(
        self,
        n_components: int,
        pca_components: int = 100,
        shrinkage: str | float | None = None,
    ) -> None:
        self.n_components = n_components
        self.pca_components = pca_components
        self.shrinkage = shrinkage
        self.pca: PCA | None = None
        self.lda: LinearDiscriminantAnalysis | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "PCALDADR":
        if y is None:
            raise ValueError("PCA->LDA requires labels y")

        pca_dim = min(self.pca_components, X.shape[1], X.shape[0] - 1)
        if pca_dim <= 0:
            raise ValueError("Invalid PCA pre-dimension in PCA->LDA")

        self.pca = PCA(n_components=pca_dim)
        Z = self.pca.fit_transform(X)

        if self.shrinkage in (None, "none"):
            self.lda = LinearDiscriminantAnalysis(n_components=self.n_components, solver="svd")
        else:
            shrinkage = self.shrinkage
            if isinstance(shrinkage, str) and shrinkage.lower() == "auto":
                shrinkage = "auto"
            self.lda = LinearDiscriminantAnalysis(
                n_components=self.n_components,
                solver="lsqr",
                shrinkage=shrinkage,
            )

        self.lda.fit(Z, y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.pca is None or self.lda is None:
            raise RuntimeError("PCA->LDA model is not fitted")
        Z = self.pca.transform(X)
        return self.lda.transform(Z)
