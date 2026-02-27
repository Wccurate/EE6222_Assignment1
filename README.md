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
conda create -n ee6222-dr python=3.10 -y
conda activate ee6222-dr
pip install -r requirements.txt
bash scripts/run_quick.sh
```

## Notes

The root `.gitignore` already excludes:
- local editor settings
- macOS system files
- Python cache and Conda environments
- TeX intermediate build artifacts
