import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ee6222_dr.pipeline import run_experiments


class _DummySplit:
    def __init__(self, name: str, x_train, y_train, x_test, y_test, image_shape):
        self.name = name
        self.X_train = x_train
        self.y_train = y_train
        self.X_test = x_test
        self.y_test = y_test
        self.image_shape = image_shape


def _make_dummy_dataset(name: str, seed: int):
    rng = np.random.default_rng(seed)
    if name == "fashion_mnist":
        n_classes = 10
        per_class_train = 20
        per_class_test = 8
        feat_dim = 28 * 28
        image_shape = (28, 28)
    else:
        n_classes = 40
        per_class_train = 4
        per_class_test = 2
        feat_dim = 64 * 64
        image_shape = (64, 64)

    y_train = np.repeat(np.arange(n_classes), per_class_train)
    y_test = np.repeat(np.arange(n_classes), per_class_test)

    centers = rng.normal(size=(n_classes, feat_dim)).astype(np.float32) * 0.2
    X_train = centers[y_train] + rng.normal(size=(y_train.size, feat_dim)).astype(np.float32) * 0.1
    X_test = centers[y_test] + rng.normal(size=(y_test.size, feat_dim)).astype(np.float32) * 0.1

    # Keep values in [0,1] for nmf/ae/vae compatibility.
    X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min() + 1e-8)
    X_test = (X_test - X_test.min()) / (X_test.max() - X_test.min() + 1e-8)

    return _DummySplit(name, X_train, y_train, X_test, y_test, image_shape)


def test_quick_pipeline_smoke(monkeypatch, tmp_path: Path) -> None:
    def _fake_loader(dataset: str, seed: int, cfg: dict):
        return _make_dummy_dataset(dataset, seed)

    monkeypatch.setattr("ee6222_dr.pipeline.load_dataset", _fake_loader)

    cfg = {
        "datasets": ["fashion_mnist", "olivetti"],
        "seeds": [0],
        "methods": ["pca", "lda", "ae"],
        "classifiers": ["knn", "mahalanobis", "logistic"],
        "tune_classifier": "logistic",
        "d_grids": {
            "fashion_mnist": [2, 8],
            "olivetti": [2, 10]
        },
        "method_d_overrides": {
            "lda": {
                "fashion_mnist": [1, 5],
                "olivetti": [1, 10]
            }
        },
        "method_grids": {
            "pca": [{"whiten": False}],
            "lda": [{"shrinkage": "none"}],
            "ae": [{"hidden_dim": 32, "epochs": 1}]
        },
        "classifier_params": {
            "knn": {"n_neighbors": 1},
            "mahalanobis": {"reg": 1e-3, "shrinkage": 0.1},
            "logistic": {"C": 1.0, "max_iter": 200}
        },
        "training": {
            "batch_size": 64,
            "ae_epochs": 1,
            "vae_epochs": 1,
            "lr": 1e-3,
            "weight_decay": 1e-5,
            "hidden_dim": 32,
            "deep_max_train_samples": 300
        },
        "cv_folds": 2,
        "n_jobs": -1,
        "interpretability": {
            "ae_d": 8,
            "ae_epochs": 1
        }
    }

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("test_smoke")
    logger.setLevel(logging.INFO)

    summary = run_experiments(cfg=cfg, run_dir=run_dir, device="cpu", logger=logger)

    assert summary["num_records"] > 0
    assert (run_dir / "results_long.csv").exists()
    assert (run_dir / "summary.json").exists()

    fig_dir = run_dir / "figures"
    assert any(fig_dir.glob("accuracy_vs_d_*.png"))

    df = pd.read_csv(run_dir / "results_long.csv")
    assert {"dataset", "method", "classifier", "d", "accuracy", "error_rate"}.issubset(df.columns)
