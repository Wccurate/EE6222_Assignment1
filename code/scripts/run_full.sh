#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python -m ee6222_dr.cli run \
  --config configs/full.json \
  --mode full \
  --device auto \
  --output outputs/runs

RUN_DIR="$(ls -td outputs/runs/ee6222_dr_full_* | head -n1)"

python -m ee6222_dr.cli plot --run_dir "$RUN_DIR"
python -m ee6222_dr.cli summarize --run_dir "$RUN_DIR"

echo "Full run completed: $RUN_DIR"
