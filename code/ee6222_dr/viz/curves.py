"""Curve plotting utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ee6222_dr.config import DATASET_NUM_CLASSES
from ee6222_dr.metrics import aggregate_curves


def _plot_one_metric(
    agg_df: pd.DataFrame,
    dataset: str,
    value_col: str,
    std_col: str,
    ylabel: str,
    out_path: Path,
) -> None:
    plt.figure(figsize=(10, 6))

    ds_df = agg_df[agg_df["dataset"] == dataset].copy()
    for (method, classifier), group in ds_df.groupby(["method", "classifier"]):
        group = group.sort_values("d")
        label = f"{method}+{classifier}"
        x = group["d"].to_numpy()
        y = group[value_col].to_numpy()
        s = group[std_col].to_numpy()

        plt.plot(x, y, marker="o", linewidth=1.8, label=label)
        plt.fill_between(x, y - s, y + s, alpha=0.15)

    if dataset in DATASET_NUM_CLASSES:
        c_minus_1 = DATASET_NUM_CLASSES[dataset] - 1
        plt.axvline(c_minus_1, color="gray", linestyle="--", linewidth=1.0, label=f"LDA C-1={c_minus_1}")

    plt.title(f"{dataset}: {ylabel} vs d")
    plt.xlabel("Embedding dimension d")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_accuracy_error_curves(df: pd.DataFrame, figures_dir: str | Path) -> list[Path]:
    """Generate accuracy/error curves with std bands."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    agg = aggregate_curves(df)
    if agg.empty:
        return []
    out_paths: list[Path] = []

    for dataset in sorted(agg["dataset"].unique()):
        acc_path = figures_dir / f"accuracy_vs_d_{dataset}.png"
        err_path = figures_dir / f"error_vs_d_{dataset}.png"

        _plot_one_metric(
            agg_df=agg,
            dataset=dataset,
            value_col="mean_accuracy",
            std_col="std_accuracy",
            ylabel="Accuracy",
            out_path=acc_path,
        )
        _plot_one_metric(
            agg_df=agg,
            dataset=dataset,
            value_col="mean_error",
            std_col="std_error",
            ylabel="Error rate",
            out_path=err_path,
        )

        out_paths.extend([acc_path, err_path])

    return out_paths


def plot_from_run_dir(run_dir: str | Path) -> list[Path]:
    """Load results_long.csv from run dir and plot core curves."""
    run_dir = Path(run_dir)
    results_path = run_dir / "results_long.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results file: {results_path}")
    df = pd.read_csv(results_path)
    return plot_accuracy_error_curves(df, run_dir / "figures")
