# EE6222 Assignment 1

This repository contains both parts of the EE6222 Assignment 1 submission:
the experiment code and the LaTeX report.

## Repository Layout

- `code/`: dimensionality-reduction experiment pipeline, configs, scripts, tests, and outputs
- `report/`: LaTeX source and generated PDF for the final report

Detailed usage for the experiment project is in [code/README.md](code/README.md).

## Quick Start (Code)

```bash
cd code
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash scripts/run_quick.sh
```

## Notes

The root `.gitignore` already excludes:
- local editor settings
- macOS system files
- Python cache and virtual environments
- TeX intermediate build artifacts
