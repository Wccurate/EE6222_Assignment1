"""Metrics and summary helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


def compute_accuracy_and_error(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Return (accuracy, error_rate)."""
    acc = float(accuracy_score(y_true, y_pred))
    return acc, float(1.0 - acc)


def summarize_best_results(df: pd.DataFrame) -> pd.DataFrame:
    """Best accuracy row for each dataset/method/classifier."""
    if df.empty:
        return df
    idx = (
        df.groupby(["dataset", "method", "classifier"])["accuracy"]
        .idxmax()
        .dropna()
        .astype(int)
    )
    cols = [
        "dataset",
        "method",
        "classifier",
        "d",
        "accuracy",
        "error_rate",
        "seed",
    ]
    return df.loc[idx, cols].sort_values(["dataset", "classifier", "method"]).reset_index(drop=True)


def aggregate_curves(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mean/std metrics over seeds for plotting."""
    grouped = (
        df.groupby(["dataset", "method", "classifier", "d"], as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_error=("error_rate", "mean"),
            std_error=("error_rate", "std"),
        )
        .fillna(0.0)
    )
    return grouped
