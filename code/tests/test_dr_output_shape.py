import numpy as np

from ee6222_dr.registry import build_dr_method


def _check_shape_and_finite(Z: np.ndarray, n_samples: int, d: int) -> None:
    assert Z.shape == (n_samples, d)
    assert np.isfinite(Z).all()


def test_dr_methods_output_shape() -> None:
    rng = np.random.default_rng(0)
    n_samples = 60
    in_dim = 20
    d = 3

    X_std = rng.normal(size=(n_samples, in_dim)).astype(np.float32)
    X_pos = rng.random(size=(n_samples, in_dim)).astype(np.float32)
    y = np.repeat(np.arange(6), 10).astype(np.int64)

    training_cfg = {
        "batch_size": 32,
        "ae_epochs": 1,
        "vae_epochs": 1,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "hidden_dim": 32,
        "deep_max_train_samples": 60,
    }

    methods = [
        ("pca", X_std, {"whiten": False}),
        ("lda", X_std, {"shrinkage": "none"}),
        ("pca_lda", X_std, {"pca_components": 10, "shrinkage": "none"}),
        ("kpca", X_std, {"gamma": 1e-3}),
        ("nmf", X_pos, {"init": "nndsvd", "solver": "cd", "max_iter": 100}),
        ("ica", X_std, {"fun": "logcosh", "max_iter": 200}),
        ("ae", X_pos, {"hidden_dim": 32, "epochs": 1}),
        ("vae", X_pos, {"hidden_dim": 32, "epochs": 1, "beta": 1.0}),
    ]

    for method_name, X, params in methods:
        dr = build_dr_method(
            method_name=method_name,
            n_components=d,
            params=params,
            training_cfg=training_cfg,
            device="cpu",
            random_state=0,
        )
        if method_name in {"lda", "pca_lda"}:
            dr.fit(X, y)
        else:
            dr.fit(X)
        Z = dr.transform(X)
        _check_shape_and_finite(Z, n_samples=n_samples, d=d)
