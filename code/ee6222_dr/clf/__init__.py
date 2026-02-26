"""Classifier interface and exports."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Classifier(ABC):
    """Common API for classifiers in reduced space."""

    @abstractmethod
    def fit(self, Z: np.ndarray, y: np.ndarray) -> "Classifier":
        raise NotImplementedError

    @abstractmethod
    def predict(self, Z: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict_proba(self, Z: np.ndarray) -> np.ndarray:
        raise NotImplementedError("predict_proba is optional for this classifier")
