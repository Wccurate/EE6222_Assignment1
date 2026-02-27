# EE6222 Assignment 1: Structured DR Experiment Project

本项目基于 `6222ass1.md` 中定义的实验 pipeline，提供一个可复现、可一键运行、可视化完整的降维分类实验框架。

## 1. 目标与范围

- 数据集：`Fashion-MNIST`、`Olivetti Faces`
- DR 方法：`PCA / LDA / PCA->LDA / Kernel PCA / NMF / ICA / AE / VAE`
- 分类器：`1-NN / Mahalanobis / LogisticRegression`
- 输出：
  - `results_long.csv`（长表）
  - `summary.json`
  - 核心曲线图（accuracy/error vs d）
  - 解释性图（PCA eigenimages、NMF components、AE reconstruction）
  - 最佳结果表（CSV）

## 2. 环境安装

建议 Python 3.10+。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. 目录说明

```text
ee6222_dr/          # 主代码包
configs/            # quick/full JSON 配置
scripts/            # 一键运行脚本
tests/              # 单测 + smoke 测试
notebooks/          # 可视化演示 notebook
outputs/runs/       # 实验输出目录
```

## 4. 运行方式

### 4.1 一键 quick（推荐先跑）

```bash
bash scripts/run_quick.sh
```

### 4.2 一键 full

```bash
bash scripts/run_full.sh
```

### 4.2.1 当前 full 加速配置（已落地）

为缩短 full 运行时间，当前仓库中的 `configs/full.json` 已启用以下设置：

- `seeds: [0]`（只跑一个随机种子）
- `cv_folds: 1`（快速模式：单次分层 holdout，而非多折 CV）
- `method_max_dims.kpca: 16`（KPCA 最高只跑到 16 维）
- `method_max_dims.nmf: 128`（NMF 最高只跑到 128 维）

### 4.2.2 缺失结果的记录方式

- 对于被维度上限裁掉、参数无效、拟合失败、分类器失败等情况，`results_long.csv` 会补一行占位记录：
  - `accuracy = N/A`
  - `error_rate = N/A`
  - `cv_score = N/A`
  - `status` 字段给出原因（如 `N/A: skipped_by_method_max_dim`）
- 画图与 summary/table 会自动跳过 `N/A` 行，保证流程可继续执行。
- `summary.json` 额外提供：
  - `num_valid_records`
  - `num_na_records`

### 4.3 CLI 分步运行

```bash
# 运行实验
python -m ee6222_dr.cli run --config configs/quick.json --mode quick --device auto --output outputs/runs

# 基于已有 run 目录画图
python -m ee6222_dr.cli plot --run_dir outputs/runs/<your_run_dir>

# 基于已有 run 目录重建 summary/table
python -m ee6222_dr.cli summarize --run_dir outputs/runs/<your_run_dir>
```

## 5. 输出结构

每次运行会创建一个新目录：

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

## 6. 无泄露策略

- 所有 scaler 仅在训练集拟合。
- DR 仅在训练集拟合，再分别 transform train/test。
- 超参数由训练集内 CV 选择，不使用测试集调参。
- 测试集仅用于最终一次评估。

## 7. 测试

```bash
pytest -q
```

测试覆盖：
- 配置校验（含 LDA 维度上限）
- 预处理无泄露
- DR 输出形状与数值有效性
- 端到端 smoke（合成数据）

## 8. Notebook 演示

`notebooks/demo_visualization.ipynb` 用于读取某次 run 目录并快速展示核心曲线。

## 9. 常见问题

- 首次运行会下载数据集（网络正常时自动进行）。
- `full` 配置计算量较大，建议先使用 `quick` 验证环境。
- `--device auto` 会优先使用 CUDA，不可用时自动回退到 CPU。
