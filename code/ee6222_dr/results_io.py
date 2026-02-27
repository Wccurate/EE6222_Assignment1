"""Run artifact writing utilities."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def create_run_dir(base_output: str | Path, experiment_name: str) -> Path:
    """Create timestamped run directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_output) / f"{experiment_name}_{timestamp}"
    (run_dir / "figures").mkdir(parents=True, exist_ok=False)
    (run_dir / "tables").mkdir(parents=True, exist_ok=False)
    return run_dir


def save_config_snapshot(run_dir: Path, cfg: dict[str, Any]) -> Path:
    """Persist config used for a run."""
    path = run_dir / "config_snapshot.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    return path


def save_results_long(run_dir: Path, rows: list[dict[str, Any]]) -> Path:
    """Write long-format experiment records."""
    columns = [
        "dataset",
        "seed",
        "method",
        "classifier",
        "d",
        "accuracy",
        "error_rate",
        "best_params",
        "tune_classifier",
        "cv_score",
        "status",
    ]
    df = pd.DataFrame(rows).reindex(columns=columns)
    path = run_dir / "results_long.csv"
    df.to_csv(path, index=False)
    return path


def write_summary_json(run_dir: Path, summary: dict[str, Any]) -> Path:
    """Persist summary JSON."""
    path = run_dir / "summary.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return path
