"""Main experiment pipeline implementation."""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ee6222_dr.config import get_dataset_num_classes
from ee6222_dr.data import DatasetSplit, load_dataset
from ee6222_dr.metrics import compute_accuracy_and_error
from ee6222_dr.preprocess import fit_transform_train_test
from ee6222_dr.registry import build_classifier, build_dr_method
from ee6222_dr.results_io import save_results_long, write_summary_json
from ee6222_dr.utils.seed import set_global_seed
from ee6222_dr.viz.curves import plot_accuracy_error_curves
from ee6222_dr.viz.interpretability import generate_interpretability_figures
from ee6222_dr.viz.tables import save_best_results_tables


@dataclass
class SelectionResult:
    """Result of parameter selection for one method and d."""

    best_params: dict[str, Any]
    best_score: float


def _is_supervised_method(method_name: str) -> bool:
    return method_name in {"lda", "pca_lda"}


def _expand_param_grid(grid_spec: Any) -> list[dict[str, Any]]:
    """Expand grid spec to a list of dictionaries."""
    if grid_spec is None:
        return [{}]

    if isinstance(grid_spec, list):
        if not grid_spec:
            return [{}]
        if not all(isinstance(item, dict) for item in grid_spec):
            raise ValueError("List-style method grid must contain dictionaries")
        return grid_spec

    if isinstance(grid_spec, dict):
        keys = list(grid_spec.keys())
        values = [v if isinstance(v, list) else [v] for v in grid_spec.values()]
        out = []
        for combo in product(*values):
            out.append(dict(zip(keys, combo)))
        return out or [{}]

    raise ValueError(f"Unsupported grid spec: {type(grid_spec)}")


def _get_method_d_grid(cfg: dict[str, Any], dataset: str, method: str) -> list[int]:
    """Return method-specific d-grid for a dataset."""
    override = cfg.get("method_d_overrides", {}).get(method, {})
    if dataset in override:
        dims = list(override[dataset])
    else:
        dims = list(cfg["d_grids"][dataset])

    dims = sorted({int(d) for d in dims if int(d) > 0})

    if method in {"lda", "pca_lda"}:
        cap = get_dataset_num_classes(dataset) - 1
        dims = [d for d in dims if d <= cap]

    return dims


def _effective_cv_folds(y: np.ndarray, requested_folds: int) -> int:
    """Ensure each class has enough samples for stratified CV."""
    _, counts = np.unique(y, return_counts=True)
    min_count = int(counts.min())
    if min_count < 2:
        raise ValueError("At least two samples per class are required for stratified CV")
    return min(requested_folds, min_count)


def _evaluate_candidate_params(
    method_name: str,
    classifier_name: str,
    params: dict[str, Any],
    d: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int,
    training_cfg: dict[str, Any],
    device: str,
    random_state: int,
    n_jobs: int,
    classifier_params: dict[str, Any] | None,
) -> float:
    """Cross-validated score for one (method, d, params)."""
    folds = _effective_cv_folds(y_train, cv_folds)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)

    scores: list[float] = []
    for train_idx, val_idx in splitter.split(X_train, y_train):
        X_tr = X_train[train_idx]
        y_tr = y_train[train_idx]
        X_val = X_train[val_idx]
        y_val = y_train[val_idx]

        _, X_tr_p, X_val_p = fit_transform_train_test(method_name, X_tr, X_val)

        dr = build_dr_method(
            method_name=method_name,
            n_components=d,
            params=params,
            training_cfg=training_cfg,
            device=device,
            random_state=random_state,
        )

        if _is_supervised_method(method_name):
            dr.fit(X_tr_p, y_tr)
        else:
            dr.fit(X_tr_p)

        Z_tr = dr.transform(X_tr_p)
        Z_val = dr.transform(X_val_p)

        clf = build_classifier(
            classifier_name=classifier_name,
            params=classifier_params,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        clf.fit(Z_tr, y_tr)
        y_pred = clf.predict(Z_val)
        acc, _ = compute_accuracy_and_error(y_val, y_pred)
        scores.append(acc)

    if not scores:
        return float("-inf")
    return float(np.mean(scores))


def _select_best_params(
    method_name: str,
    classifier_name: str,
    d: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_candidates: list[dict[str, Any]],
    cfg: dict[str, Any],
    device: str,
    seed: int,
) -> SelectionResult:
    """Select best params by train-only cross validation."""
    best_score = float("-inf")
    best_params: dict[str, Any] = param_candidates[0] if param_candidates else {}

    classifier_params = cfg.get("classifier_params", {}).get(classifier_name, {})

    for params in param_candidates:
        try:
            score = _evaluate_candidate_params(
                method_name=method_name,
                classifier_name=classifier_name,
                params=params,
                d=d,
                X_train=X_train,
                y_train=y_train,
                cv_folds=int(cfg.get("cv_folds", 3)),
                training_cfg=cfg.get("training", {}),
                device=device,
                random_state=seed,
                n_jobs=int(cfg.get("n_jobs", -1)),
                classifier_params=classifier_params,
            )
        except Exception:
            score = float("-inf")

        if score > best_score:
            best_score = score
            best_params = params

    return SelectionResult(best_params=best_params, best_score=best_score)


def _fit_method_and_project(
    method_name: str,
    d: int,
    params: dict[str, Any],
    split: DatasetSplit,
    cfg: dict[str, Any],
    device: str,
    seed: int,
):
    """Fit train scaler+DR and transform train/test."""
    scaler, X_train_p, X_test_p = fit_transform_train_test(
        method_name=method_name,
        X_train=split.X_train,
        X_test=split.X_test,
    )

    dr = build_dr_method(
        method_name=method_name,
        n_components=d,
        params=params,
        training_cfg=cfg.get("training", {}),
        device=device,
        random_state=seed,
    )

    if _is_supervised_method(method_name):
        dr.fit(X_train_p, split.y_train)
    else:
        dr.fit(X_train_p)

    Z_train = dr.transform(X_train_p)
    Z_test = dr.transform(X_test_p)

    return scaler, dr, Z_train, Z_test


def run_experiments(
    cfg: dict[str, Any],
    run_dir: str | Path,
    device: str,
    logger,
) -> dict[str, Any]:
    """Execute the complete experiment loop and write outputs."""
    run_dir = Path(run_dir)

    rows: list[dict[str, Any]] = []
    dataset_for_interpret: dict[str, DatasetSplit] = {}

    tune_classifier = cfg.get("tune_classifier", "logistic")
    if tune_classifier not in cfg["classifiers"]:
        tune_classifier = cfg["classifiers"][0]

    for dataset in cfg["datasets"]:
        logger.info("Starting dataset: %s", dataset)

        for seed in cfg["seeds"]:
            set_global_seed(seed)
            split = load_dataset(dataset=dataset, seed=seed, cfg=cfg)
            if dataset not in dataset_for_interpret:
                dataset_for_interpret[dataset] = split

            logger.info(
                "Loaded %s | seed=%d | train=%d | test=%d | D=%d",
                dataset,
                seed,
                split.X_train.shape[0],
                split.X_test.shape[0],
                split.X_train.shape[1],
            )

            for method in cfg["methods"]:
                d_grid = _get_method_d_grid(cfg, dataset, method)
                if not d_grid:
                    logger.warning("Skip %s on %s: empty valid d-grid", method, dataset)
                    continue

                param_candidates = _expand_param_grid(cfg.get("method_grids", {}).get(method, [{}]))

                for d in d_grid:
                    logger.info("Tuning %s | dataset=%s | seed=%d | d=%d", method, dataset, seed, d)

                    selection = _select_best_params(
                        method_name=method,
                        classifier_name=tune_classifier,
                        d=d,
                        X_train=split.X_train,
                        y_train=split.y_train,
                        param_candidates=param_candidates,
                        cfg=cfg,
                        device=device,
                        seed=seed,
                    )

                    if selection.best_score == float("-inf"):
                        logger.warning(
                            "No valid params for %s on %s seed=%d d=%d",
                            method,
                            dataset,
                            seed,
                            d,
                        )
                        continue

                    try:
                        _, _, Z_train, Z_test = _fit_method_and_project(
                            method_name=method,
                            d=d,
                            params=selection.best_params,
                            split=split,
                            cfg=cfg,
                            device=device,
                            seed=seed,
                        )
                    except Exception as exc:
                        logger.warning(
                            "Failed fitting %s on %s seed=%d d=%d: %s",
                            method,
                            dataset,
                            seed,
                            d,
                            str(exc),
                        )
                        logger.debug(traceback.format_exc())
                        continue

                    for clf_name in cfg["classifiers"]:
                        try:
                            clf_params = cfg.get("classifier_params", {}).get(clf_name, {})
                            clf = build_classifier(
                                classifier_name=clf_name,
                                params=clf_params,
                                n_jobs=int(cfg.get("n_jobs", -1)),
                                random_state=seed,
                            )
                            clf.fit(Z_train, split.y_train)
                            y_pred = clf.predict(Z_test)
                            acc, err = compute_accuracy_and_error(split.y_test, y_pred)

                            rows.append(
                                {
                                    "dataset": dataset,
                                    "seed": seed,
                                    "method": method,
                                    "classifier": clf_name,
                                    "d": int(d),
                                    "accuracy": acc,
                                    "error_rate": err,
                                    "best_params": json.dumps(selection.best_params, sort_keys=True),
                                    "tune_classifier": tune_classifier,
                                    "cv_score": selection.best_score,
                                }
                            )
                        except Exception as exc:
                            logger.warning(
                                "Classifier failed | dataset=%s seed=%d method=%s d=%d clf=%s err=%s",
                                dataset,
                                seed,
                                method,
                                d,
                                clf_name,
                                str(exc),
                            )
                            logger.debug(traceback.format_exc())

    results_path = save_results_long(run_dir, rows)
    df = pd.read_csv(results_path)

    curve_paths = plot_accuracy_error_curves(df, run_dir / "figures") if not df.empty else []
    table_paths = save_best_results_tables(df, run_dir / "tables") if not df.empty else []

    interpret_paths: list[Path] = []
    for dataset, split in dataset_for_interpret.items():
        try:
            interpret_paths.extend(
                generate_interpretability_figures(
                    dataset_name=dataset,
                    X_train=split.X_train,
                    image_shape=split.image_shape,
                    figures_dir=run_dir / "figures",
                    device=device,
                    ae_d=int(cfg.get("interpretability", {}).get("ae_d", 16)),
                    ae_epochs=int(cfg.get("interpretability", {}).get("ae_epochs", 8)),
                )
            )
        except Exception as exc:
            logger.warning("Interpretability figure generation failed for %s: %s", dataset, str(exc))
            logger.debug(traceback.format_exc())

    summary = {
        "num_records": int(df.shape[0]),
        "datasets": cfg["datasets"],
        "methods": cfg["methods"],
        "classifiers": cfg["classifiers"],
        "seeds": cfg["seeds"],
        "device": device,
        "results_csv": str(results_path),
        "figures": [str(p) for p in curve_paths + interpret_paths],
        "tables": [str(p) for p in table_paths],
    }

    if not df.empty:
        best = (
            df.sort_values("accuracy", ascending=False)
            .groupby(["dataset", "method", "classifier"], as_index=False)
            .first()
        )
        summary["best_results"] = best.to_dict(orient="records")

    write_summary_json(run_dir, summary)
    return summary


def summarize_from_run_dir(run_dir: str | Path) -> dict[str, Any]:
    """Recompute summary/tables from existing results file."""
    run_dir = Path(run_dir)
    results_path = run_dir / "results_long.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results file: {results_path}")

    df = pd.read_csv(results_path)
    table_paths = save_best_results_tables(df, run_dir / "tables") if not df.empty else []

    summary = {
        "num_records": int(df.shape[0]),
        "results_csv": str(results_path),
        "tables": [str(p) for p in table_paths],
    }

    if not df.empty:
        best = (
            df.sort_values("accuracy", ascending=False)
            .groupby(["dataset", "method", "classifier"], as_index=False)
            .first()
        )
        summary["best_results"] = best.to_dict(orient="records")

    write_summary_json(run_dir, summary)
    return summary
