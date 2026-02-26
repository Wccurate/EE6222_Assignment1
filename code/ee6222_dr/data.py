"""Dataset loading and split utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.datasets import fetch_olivetti_faces


@dataclass
class DatasetSplit:
    """Container for train/test arrays and metadata."""

    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    image_shape: tuple[int, int]


def stratified_subsample(
    X: np.ndarray,
    y: np.ndarray,
    n_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified subsampling preserving class proportions."""
    if n_samples <= 0 or n_samples >= len(y):
        return X, y

    rng = np.random.default_rng(seed)
    unique_classes = np.unique(y)

    per_class = {cls: np.where(y == cls)[0] for cls in unique_classes}
    counts = {cls: len(indices) for cls, indices in per_class.items()}

    # Allocate at least one per class, then distribute remainder proportionally.
    allocation = {cls: 1 for cls in unique_classes}
    remaining = n_samples - len(unique_classes)
    if remaining > 0:
        total = len(y)
        for cls in unique_classes:
            extra = int(remaining * counts[cls] / total)
            allocation[cls] += extra

        # Fix any rounding mismatch.
        allocated = sum(allocation.values())
        while allocated < n_samples:
            cls = unique_classes[int(rng.integers(0, len(unique_classes)))]
            if allocation[cls] < counts[cls]:
                allocation[cls] += 1
                allocated += 1
        while allocated > n_samples:
            cls = unique_classes[int(rng.integers(0, len(unique_classes)))]
            if allocation[cls] > 1:
                allocation[cls] -= 1
                allocated -= 1

    selected: list[int] = []
    for cls in unique_classes:
        indices = per_class[cls].copy()
        rng.shuffle(indices)
        take = min(allocation[cls], len(indices))
        selected.extend(indices[:take].tolist())

    selected = np.array(selected, dtype=int)
    rng.shuffle(selected)
    return X[selected], y[selected]


def load_fashion_mnist(
    seed: int,
    data_root: str | Path,
    train_limit: int | None = None,
    test_limit: int | None = None,
) -> DatasetSplit:
    """Load Fashion-MNIST from torchvision using official split."""
    try:
        import torchvision
    except Exception as exc:
        raise ImportError("torchvision is required for Fashion-MNIST") from exc

    data_root = Path(data_root)
    train_ds = torchvision.datasets.FashionMNIST(root=data_root, train=True, download=True)
    test_ds = torchvision.datasets.FashionMNIST(root=data_root, train=False, download=True)

    X_train = train_ds.data.numpy().reshape(-1, 28 * 28).astype(np.float32) / 255.0
    y_train = train_ds.targets.numpy().astype(np.int64)
    X_test = test_ds.data.numpy().reshape(-1, 28 * 28).astype(np.float32) / 255.0
    y_test = test_ds.targets.numpy().astype(np.int64)

    if train_limit is not None:
        X_train, y_train = stratified_subsample(X_train, y_train, train_limit, seed)
    if test_limit is not None:
        X_test, y_test = stratified_subsample(X_test, y_test, test_limit, seed)

    return DatasetSplit(
        name="fashion_mnist",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        image_shape=(28, 28),
    )


def _split_olivetti_per_class(
    X: np.ndarray,
    y: np.ndarray,
    train_per_class: int,
    test_per_class: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create per-class train/test split for Olivetti."""
    if train_per_class + test_per_class > 10:
        raise ValueError("Olivetti has 10 images per class; train_per_class + test_per_class must be <= 10")

    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    test_idx: list[int] = []

    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0].copy()
        rng.shuffle(cls_idx)
        train_idx.extend(cls_idx[:train_per_class].tolist())
        test_idx.extend(cls_idx[train_per_class : train_per_class + test_per_class].tolist())

    train_idx = np.array(train_idx, dtype=int)
    test_idx = np.array(test_idx, dtype=int)

    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def load_olivetti(
    seed: int,
    train_per_class: int = 5,
    test_per_class: int = 5,
) -> DatasetSplit:
    """Load Olivetti faces and split per identity."""
    dataset = fetch_olivetti_faces(shuffle=False, download_if_missing=True)
    X = dataset.data.astype(np.float32)
    y = dataset.target.astype(np.int64)

    X_train, y_train, X_test, y_test = _split_olivetti_per_class(
        X=X,
        y=y,
        train_per_class=train_per_class,
        test_per_class=test_per_class,
        seed=seed,
    )

    return DatasetSplit(
        name="olivetti",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        image_shape=(64, 64),
    )


def load_dataset(dataset: str, seed: int, cfg: dict[str, Any]) -> DatasetSplit:
    """Dispatch dataset loading according to config."""
    quick_subsample = cfg.get("quick_subsample", {})

    if dataset == "fashion_mnist":
        train_limit = quick_subsample.get("fashion_mnist_train")
        test_limit = quick_subsample.get("fashion_mnist_test")
        data_root = cfg.get("data_root", "./data")
        return load_fashion_mnist(
            seed=seed,
            data_root=data_root,
            train_limit=train_limit,
            test_limit=test_limit,
        )

    if dataset == "olivetti":
        train_pc = int(quick_subsample.get("olivetti_train_per_class", 5))
        test_pc = int(quick_subsample.get("olivetti_test_per_class", 5))
        return load_olivetti(seed=seed, train_per_class=train_pc, test_per_class=test_pc)

    raise ValueError(f"Unsupported dataset: {dataset}")
