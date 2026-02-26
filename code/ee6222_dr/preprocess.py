"""Preprocessing utilities with strict train-only fitting."""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

MINMAX_METHODS = {"nmf", "ae", "vae"}


def get_preprocessor(method_name: str):
    """Return the proper scaler for a method."""
    if method_name in MINMAX_METHODS:
        return MinMaxScaler(feature_range=(0.0, 1.0))
    return StandardScaler()


def fit_transform_train_test(
    method_name: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
):
    """Fit scaler on train only and transform both train/test."""
    scaler = get_preprocessor(method_name)
    X_train_t = scaler.fit_transform(X_train)
    X_test_t = scaler.transform(X_test)
    return scaler, X_train_t, X_test_t
