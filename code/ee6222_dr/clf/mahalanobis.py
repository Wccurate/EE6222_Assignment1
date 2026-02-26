"""Regularized Mahalanobis distance classifier."""

from __future__ import annotations

import numpy as np

from . import Classifier


class MahalanobisClassifier(Classifier):
    """Classify by minimum Mahalanobis distance to class means."""

    def __init__(self, reg: float = 1e-3, shrinkage: float = 0.1) -> None:
        self.reg = reg
        self.shrinkage = shrinkage
        self.classes_: np.ndarray | None = None
        self.class_means_: dict[int, np.ndarray] | None = None
        self.inv_cov_: np.ndarray | None = None

    def fit(self, Z: np.ndarray, y: np.ndarray) -> "MahalanobisClassifier":
        classes = np.unique(y)
        means = {int(cls): Z[y == cls].mean(axis=0) for cls in classes}

        cov = np.cov(Z, rowvar=False)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]], dtype=np.float64)

        diag_cov = np.diag(np.diag(cov))
        cov = (1.0 - self.shrinkage) * cov + self.shrinkage * diag_cov
        cov = cov + self.reg * np.eye(cov.shape[0])

        inv_cov = np.linalg.pinv(cov)

        self.classes_ = classes
        self.class_means_ = means
        self.inv_cov_ = inv_cov
        return self

    def _distance_matrix(self, Z: np.ndarray) -> np.ndarray:
        if self.classes_ is None or self.class_means_ is None or self.inv_cov_ is None:
            raise RuntimeError("MahalanobisClassifier is not fitted")

        dists = np.zeros((Z.shape[0], len(self.classes_)), dtype=np.float64)
        for i, cls in enumerate(self.classes_):
            diff = Z - self.class_means_[int(cls)]
            dists[:, i] = np.einsum("bi,ij,bj->b", diff, self.inv_cov_, diff)
        return dists

    def predict(self, Z: np.ndarray) -> np.ndarray:
        dists = self._distance_matrix(Z)
        idx = np.argmin(dists, axis=1)
        assert self.classes_ is not None
        return self.classes_[idx]

    def predict_proba(self, Z: np.ndarray) -> np.ndarray:
        dists = self._distance_matrix(Z)
        logits = -dists
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        return probs
