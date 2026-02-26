"""Method and classifier factory functions."""

from __future__ import annotations

from typing import Any

from ee6222_dr.clf.knn import KNNClassifier
from ee6222_dr.clf.logistic import LogisticClassifier
from ee6222_dr.clf.mahalanobis import MahalanobisClassifier
from ee6222_dr.dr.ae import AutoEncoderDR
from ee6222_dr.dr.ica import ICADR
from ee6222_dr.dr.kpca import KernelPCADR
from ee6222_dr.dr.lda import LDADR
from ee6222_dr.dr.nmf import NMFDR
from ee6222_dr.dr.pca import PCADR
from ee6222_dr.dr.pca_lda import PCALDADR
from ee6222_dr.dr.vae import VAEDR


def build_dr_method(
    method_name: str,
    n_components: int,
    params: dict[str, Any],
    training_cfg: dict[str, Any],
    device: str,
    random_state: int,
):
    """Instantiate DR method by name."""
    p = dict(params)

    if method_name == "pca":
        return PCADR(n_components=n_components, whiten=bool(p.get("whiten", False)))

    if method_name == "lda":
        return LDADR(n_components=n_components, shrinkage=p.get("shrinkage", None))

    if method_name == "pca_lda":
        default_pca_dim = int(p.get("pca_components", max(2 * n_components, n_components + 1)))
        return PCALDADR(
            n_components=n_components,
            pca_components=default_pca_dim,
            shrinkage=p.get("shrinkage", None),
        )

    if method_name == "kpca":
        return KernelPCADR(n_components=n_components, gamma=float(p.get("gamma", 1e-3)))

    if method_name == "nmf":
        return NMFDR(
            n_components=n_components,
            init=str(p.get("init", "nndsvd")),
            solver=str(p.get("solver", "cd")),
            max_iter=int(p.get("max_iter", 300)),
            random_state=random_state,
        )

    if method_name == "ica":
        return ICADR(
            n_components=n_components,
            fun=str(p.get("fun", "logcosh")),
            max_iter=int(p.get("max_iter", 500)),
            tol=float(p.get("tol", 1e-4)),
            random_state=random_state,
        )

    if method_name == "ae":
        return AutoEncoderDR(
            n_components=n_components,
            hidden_dim=int(p.get("hidden_dim", training_cfg.get("hidden_dim", 256))),
            epochs=int(p.get("epochs", training_cfg.get("ae_epochs", 10))),
            batch_size=int(p.get("batch_size", training_cfg.get("batch_size", 256))),
            lr=float(p.get("lr", training_cfg.get("lr", 1e-3))),
            weight_decay=float(p.get("weight_decay", training_cfg.get("weight_decay", 1e-5))),
            device=device,
            max_train_samples=training_cfg.get("deep_max_train_samples"),
        )

    if method_name == "vae":
        return VAEDR(
            n_components=n_components,
            hidden_dim=int(p.get("hidden_dim", training_cfg.get("hidden_dim", 256))),
            epochs=int(p.get("epochs", training_cfg.get("vae_epochs", 10))),
            batch_size=int(p.get("batch_size", training_cfg.get("batch_size", 256))),
            lr=float(p.get("lr", training_cfg.get("lr", 1e-3))),
            weight_decay=float(p.get("weight_decay", training_cfg.get("weight_decay", 1e-5))),
            beta=float(p.get("beta", training_cfg.get("beta", 1.0))),
            device=device,
            max_train_samples=training_cfg.get("deep_max_train_samples"),
        )

    raise ValueError(f"Unknown method: {method_name}")


def build_classifier(
    classifier_name: str,
    params: dict[str, Any] | None,
    n_jobs: int,
    random_state: int,
):
    """Instantiate classifier by name."""
    p = dict(params or {})

    if classifier_name == "knn":
        return KNNClassifier(n_neighbors=int(p.get("n_neighbors", 1)))

    if classifier_name == "mahalanobis":
        return MahalanobisClassifier(
            reg=float(p.get("reg", 1e-3)),
            shrinkage=float(p.get("shrinkage", 0.1)),
        )

    if classifier_name == "logistic":
        return LogisticClassifier(
            C=float(p.get("C", 1.0)),
            max_iter=int(p.get("max_iter", 1000)),
            n_jobs=n_jobs,
            random_state=random_state,
        )

    raise ValueError(f"Unknown classifier: {classifier_name}")
