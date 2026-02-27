"""Logistic regression classifier wrapper."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from . import Classifier


class LogisticClassifier(Classifier):
    """Multiclass logistic regression in reduced space."""

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 3000,
        tol: float = 1e-3,
        solver: str = "lbfgs",
        n_jobs: int = -1,
        random_state: int = 0,
    ) -> None:
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            tol=tol,
            solver=solver,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    def fit(self, Z: np.ndarray, y: np.ndarray) -> "LogisticClassifier":
        self.model.fit(Z, y)
        return self

    def predict(self, Z: np.ndarray) -> np.ndarray:
        return self.model.predict(Z)

    def predict_proba(self, Z: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(Z)
