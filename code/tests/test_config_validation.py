from copy import deepcopy

import pytest

from ee6222_dr.config import apply_defaults, validate_config


def test_validate_config_rejects_invalid_lda_dims() -> None:
    cfg = apply_defaults(
        {
            "datasets": ["fashion_mnist"],
            "methods": ["lda"],
            "classifiers": ["knn"],
            "seeds": [0],
            "d_grids": {"fashion_mnist": [2, 4]},
            "method_grids": {"lda": [{"shrinkage": "none"}]},
            "method_d_overrides": {"lda": {"fashion_mnist": [1, 9, 10]}},
        }
    )

    with pytest.raises(ValueError, match="exceed C-1"):
        validate_config(cfg)


def test_validate_config_accepts_valid_setup() -> None:
    cfg = apply_defaults(
        {
            "datasets": ["olivetti"],
            "methods": ["pca", "lda"],
            "classifiers": ["knn", "logistic"],
            "seeds": [0, 1],
            "d_grids": {"olivetti": [2, 10, 39]},
            "method_grids": {
                "pca": [{"whiten": False}],
                "lda": [{"shrinkage": "auto"}],
            },
            "method_d_overrides": {"lda": {"olivetti": [1, 10, 39]}},
        }
    )

    # Should not raise.
    validate_config(deepcopy(cfg))
