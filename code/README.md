# EE6222 Assignment 1: DR Experiment Project

This project implements a reproducible pipeline for dimensionality reduction
and classification experiments used in EE6222 Assignment 1.

## 1. Scope

- Datasets: `Fashion-MNIST`, `Olivetti Faces`
- DR methods: `PCA`, `LDA`, `PCA->LDA`, `Kernel PCA`, `NMF`, `ICA`, `AE`, `VAE`
- Classifiers: `1-NN`, `Mahalanobis`, `LogisticRegression`
- Main outputs:
  - `results_long.csv`
  - `summary.json`
  - accuracy/error curves vs dimension
  - interpretability figures (PCA eigenimages, NMF components, AE reconstructions)
  - best-result tables (CSV)

## 2. Environment Setup

Use Conda (Miniconda or Anaconda) with Python 3.10.

```bash
conda create -n ee6222-dr python=3.10 -y
conda activate ee6222-dr
pip install -r requirements.txt
```

## 3. Project Structure

```text
ee6222_dr/          # main package
configs/            # quick/full JSON configs
scripts/            # one-click run scripts
tests/              # unit + smoke tests
notebooks/          # visualization notebook
outputs/runs/       # experiment artifacts
```

## 4. Running Experiments

Activate the environment before running any command:

```bash
conda activate ee6222-dr
```

### 4.1 Quick run

```bash
bash scripts/run_quick.sh
```

### 4.2 Full run

```bash
bash scripts/run_full.sh
```

### 4.3 Current accelerated full settings

To keep runtime manageable, the current `configs/full.json` uses:

- `seeds: [0]`
- `cv_folds: 1` (single stratified holdout in parameter selection)
- `method_max_dims.kpca: 16`
- `method_max_dims.nmf: 128`

### 4.4 CLI usage

```bash
# Run experiments
python -m ee6222_dr.cli run --config configs/quick.json --mode quick --device auto --output outputs/runs

# Plot from an existing run directory
python -m ee6222_dr.cli plot --run_dir outputs/runs/<your_run_dir>

# Rebuild summary/tables from an existing run directory
python -m ee6222_dr.cli summarize --run_dir outputs/runs/<your_run_dir>
```

## 5. How N/A Results Are Recorded

When a combination is skipped or fails (for example due to dimension caps,
invalid parameter candidates, fit failures, or classifier failures), one
placeholder row is still written to `results_long.csv` with:

- `accuracy = N/A`
- `error_rate = N/A`
- `cv_score = N/A`
- `status` indicating the reason (for example `N/A: skipped_by_method_max_dim`)

Plotting and summary generation automatically ignore N/A rows.
`summary.json` also includes:

- `num_valid_records`
- `num_na_records`

## 6. Output Layout

Each run creates:

```text
outputs/runs/<experiment_name>_<timestamp>/
  config_snapshot.json
  logs.txt
  results_long.csv
  summary.json
  figures/
    accuracy_vs_d_<dataset>.png
    error_vs_d_<dataset>.png
    pca_eigenimages_<dataset>.png
    nmf_components_<dataset>.png
    ae_reconstruction_<dataset>_d<d>.png
  tables/
    best_results_<dataset>.csv
```

## 7. Leakage Control Policy

- Scalers are fitted on training data only.
- DR models are fitted on training data only, then applied to train/test separately.
- Hyperparameters are selected on train-only splits.
- Test data is used only for final evaluation.

## 8. Tests

```bash
pytest -q
```

Current tests cover:

- config validation (including LDA dimension limits)
- no-leakage preprocessing behavior
- DR output shape/value sanity checks
- end-to-end smoke test on synthetic data

## 9. Notebook

`notebooks/demo_visualization.ipynb` reads a run directory and shows the main
curves quickly.

## 10. Notes

- Datasets are downloaded automatically on first run.
- `full` mode can be expensive; verify environment with `quick` first.
- `--device auto` uses CUDA when available and falls back to CPU.
- If `conda activate` is not available in your shell, run `conda init` once and reopen the terminal.
