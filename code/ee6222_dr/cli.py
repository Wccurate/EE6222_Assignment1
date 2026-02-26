"""Command-line interface for EE6222 DR experiments."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from ee6222_dr.config import apply_mode_overrides, load_config, validate_config
from ee6222_dr.pipeline import run_experiments, summarize_from_run_dir
from ee6222_dr.results_io import create_run_dir, save_config_snapshot, write_summary_json
from ee6222_dr.utils.device import resolve_device
from ee6222_dr.utils.logger import build_logger
from ee6222_dr.viz.curves import plot_from_run_dir


def _try_git_hash(cwd: Path) -> str | None:
    """Return git commit hash if available."""
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return output.strip()
    except Exception:
        return None


def cmd_run(args: argparse.Namespace) -> int:
    """Run full experiment pipeline and write artifacts."""
    cfg = load_config(args.config)
    cfg = apply_mode_overrides(cfg, args.mode)
    validate_config(cfg)

    device = resolve_device(args.device)

    run_dir = create_run_dir(args.output, cfg.get("experiment_name", "ee6222_dr"))
    save_config_snapshot(run_dir, cfg)

    logger = build_logger(run_dir / "logs.txt")
    logger.info("Run directory: %s", run_dir)
    logger.info("Mode=%s Device=%s", args.mode, device)

    summary = run_experiments(cfg=cfg, run_dir=run_dir, device=device, logger=logger)
    summary["run_dir"] = str(run_dir)
    summary["mode"] = args.mode

    git_hash = _try_git_hash(Path.cwd())
    if git_hash is not None:
        summary["git_hash"] = git_hash

    write_summary_json(run_dir, summary)

    print(json.dumps({"status": "ok", "run_dir": str(run_dir)}, ensure_ascii=False))
    return 0


def cmd_plot(args: argparse.Namespace) -> int:
    """Generate core curves from existing results_long.csv."""
    out_paths = plot_from_run_dir(args.run_dir)
    print(json.dumps({"status": "ok", "figures": [str(p) for p in out_paths]}, ensure_ascii=False))
    return 0


def cmd_summarize(args: argparse.Namespace) -> int:
    """Regenerate summary/tables from existing results file."""
    summary = summarize_from_run_dir(args.run_dir)
    print(json.dumps({"status": "ok", "summary": summary}, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build top-level argparse parser."""
    parser = argparse.ArgumentParser(description="EE6222 dimensionality reduction experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_run = subparsers.add_parser("run", help="run experiments")
    p_run.add_argument("--config", required=True, help="Path to JSON config")
    p_run.add_argument("--mode", choices=["quick", "full"], default="quick")
    p_run.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p_run.add_argument("--output", default="outputs/runs", help="Output base directory")
    p_run.set_defaults(func=cmd_run)

    p_plot = subparsers.add_parser("plot", help="plot curves from existing run")
    p_plot.add_argument("--run_dir", required=True, help="Run directory containing results_long.csv")
    p_plot.set_defaults(func=cmd_plot)

    p_sum = subparsers.add_parser("summarize", help="regenerate summary/tables")
    p_sum.add_argument("--run_dir", required=True, help="Run directory containing results_long.csv")
    p_sum.set_defaults(func=cmd_summarize)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
