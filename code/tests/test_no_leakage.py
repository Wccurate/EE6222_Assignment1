import numpy as np

from ee6222_dr.preprocess import fit_transform_train_test


def test_scaler_fits_train_only() -> None:
    X_train = np.array([[0.0, 0.0], [2.0, 2.0]], dtype=float)
    X_test = np.array([[100.0, 100.0]], dtype=float)

    scaler, X_train_t, X_test_t = fit_transform_train_test("pca", X_train, X_test)

    # StandardScaler mean should come from train only: (0+2)/2 = 1.
    assert np.allclose(scaler.mean_, np.array([1.0, 1.0]))

    # Train values should be centered around 0.
    assert np.allclose(X_train_t.mean(axis=0), np.array([0.0, 0.0]))

    # Test transform should use same train statistics.
    assert X_test_t[0, 0] > 10.0
