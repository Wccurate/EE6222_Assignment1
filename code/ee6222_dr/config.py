"""Configuration loading and validation."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

KNOWN_DATASETS = {"fashion_mnist", "olivetti"}
KNOWN_METHODS = {"pca", "lda", "pca_lda", "kpca", "nmf", "ica", "ae", "vae"}
KNOWN_CLASSIFIERS = {"knn", "mahalanobis", "logistic"}
DATASET_NUM_CLASSES = {"fashion_mnist": 10, "olivetti": 40}


def load_config(path: str | Path) -> dict[str, Any]:
    """Load JSON config and apply defaults."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    return apply_defaults(cfg)


def apply_mode_overrides(cfg: dict[str, Any], mode: str) -> dict[str, Any]:
    """Apply optional quick/full override sections without mutating input."""
    out = deepcopy(cfg)
    key = f"{mode}_overrides"
    if key in out:
        out = deep_update(out, out[key])
    out["mode"] = mode
    return out


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Recursively update dictionaries."""
    result = deepcopy(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_update(result[k], v)
        else:
            result[k] = deepcopy(v)
    return result


def apply_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    """Fill missing optional config fields with defaults."""
    out = deepcopy(cfg)
    out.setdefault("experiment_name", "ee6222_dr")
    out.setdefault("datasets", ["fashion_mnist", "olivetti"])
    out.setdefault("seeds", [0])
    out.setdefault("methods", sorted(KNOWN_METHODS))
    out.setdefault("classifiers", ["knn", "mahalanobis", "logistic"])
    out.setdefault(
        "d_grids",
        {
            "fashion_mnist": [2, 4, 8, 16, 32, 64, 128, 256],
            "olivetti": [2, 5, 10, 20, 30, 39, 60, 80, 100, 150, 200],
        },
    )
    out.setdefault("method_grids", {})
    out.setdefault("method_d_overrides", {})
    out.setdefault("method_max_dims", {})
    out.setdefault("cv_folds", 3)
    out.setdefault("n_jobs", -1)
    out.setdefault(
        "training",
        {
            "batch_size": 256,
            "ae_epochs": 10,
            "vae_epochs": 10,
            "lr": 1e-3,
            "weight_decay": 1e-5,
            "beta": 1.0,
        },
    )
    out.setdefault("quick_subsample", {})
    out.setdefault("paired_test", False)

    # Ensure every method has at least one candidate configuration.
    for method in out["methods"]:
        out["method_grids"].setdefault(method, [{}])

    return out


def validate_config(cfg: dict[str, Any]) -> None:
    """Validate config structure and key constraints."""
    datasets = cfg.get("datasets")
    methods = cfg.get("methods")
    classifiers = cfg.get("classifiers")
    seeds = cfg.get("seeds")
    d_grids = cfg.get("d_grids")

    if not isinstance(datasets, list) or not datasets:
        raise ValueError("`datasets` must be a non-empty list")
    unknown_datasets = set(datasets) - KNOWN_DATASETS
    if unknown_datasets:
        raise ValueError(f"Unknown datasets: {sorted(unknown_datasets)}")

    if not isinstance(methods, list) or not methods:
        raise ValueError("`methods` must be a non-empty list")
    unknown_methods = set(methods) - KNOWN_METHODS
    if unknown_methods:
        raise ValueError(f"Unknown methods: {sorted(unknown_methods)}")

    if not isinstance(classifiers, list) or not classifiers:
        raise ValueError("`classifiers` must be a non-empty list")
    unknown_clf = set(classifiers) - KNOWN_CLASSIFIERS
    if unknown_clf:
        raise ValueError(f"Unknown classifiers: {sorted(unknown_clf)}")

    if not isinstance(seeds, list) or not seeds:
        raise ValueError("`seeds` must be a non-empty list of integers")
    if any((not isinstance(s, int)) or s < 0 for s in seeds):
        raise ValueError("Each seed must be a non-negative integer")

    if not isinstance(d_grids, dict):
        raise ValueError("`d_grids` must be a dict keyed by dataset name")
    for dataset in datasets:
        if dataset not in d_grids:
            raise ValueError(f"Missing d-grid for dataset: {dataset}")
        dims = d_grids[dataset]
        if not isinstance(dims, list) or not dims or any((not isinstance(d, int)) or d <= 0 for d in dims):
            raise ValueError(f"Invalid d-grid for dataset {dataset}: {dims}")

    cv_folds = cfg.get("cv_folds", 3)
    if not isinstance(cv_folds, int) or cv_folds < 1:
        raise ValueError("`cv_folds` must be >= 1")

    # Optional explicit LDA method-specific d override must satisfy d <= C-1.
    method_d_overrides = cfg.get("method_d_overrides", {})
    if not isinstance(method_d_overrides, dict):
        raise ValueError("`method_d_overrides` must be a dict")

    for method_name in ("lda", "pca_lda"):
        if method_name not in method_d_overrides:
            continue
        override = method_d_overrides[method_name]
        if not isinstance(override, dict):
            raise ValueError(f"method_d_overrides['{method_name}'] must be a dict")
        for dataset, dims in override.items():
            if dataset not in DATASET_NUM_CLASSES:
                raise ValueError(f"Unknown dataset in method_d_overrides: {dataset}")
            c_minus_1 = DATASET_NUM_CLASSES[dataset] - 1
            if any(d > c_minus_1 for d in dims):
                raise ValueError(
                    f"{method_name} dims for {dataset} exceed C-1={c_minus_1}: {dims}"
                )

    method_max_dims = cfg.get("method_max_dims", {})
    if not isinstance(method_max_dims, dict):
        raise ValueError("`method_max_dims` must be a dict")
    for method_name, max_dim in method_max_dims.items():
        if method_name not in KNOWN_METHODS:
            raise ValueError(f"Unknown method in method_max_dims: {method_name}")
        if not isinstance(max_dim, int) or max_dim <= 0:
            raise ValueError(f"method_max_dims['{method_name}'] must be a positive integer")


def get_dataset_num_classes(dataset: str) -> int:
    """Return known number of classes for supported datasets."""
    return DATASET_NUM_CLASSES[dataset]
