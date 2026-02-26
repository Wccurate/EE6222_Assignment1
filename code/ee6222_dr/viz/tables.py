"""Table export utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ee6222_dr.metrics import summarize_best_results


def save_best_results_tables(df: pd.DataFrame, table_dir: str | Path) -> list[Path]:
    """Save best result table per dataset."""
    table_dir = Path(table_dir)
    table_dir.mkdir(parents=True, exist_ok=True)

    best = summarize_best_results(df)
    out_paths: list[Path] = []

    for dataset, group in best.groupby("dataset"):
        out_path = table_dir / f"best_results_{dataset}.csv"
        group.to_csv(out_path, index=False)
        out_paths.append(out_path)

    return out_paths
