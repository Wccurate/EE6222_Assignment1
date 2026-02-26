"""Dimensionality reduction method interfaces and exports."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class DRMethod(ABC):
    """Common API for all dimensionality reduction methods."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "DRMethod":
        raise NotImplementedError

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)
