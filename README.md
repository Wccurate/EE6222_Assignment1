# EE6222 Assignment 1

本仓库是 EE6222 Assignment 1 的根目录，包含代码实验项目与报告项目两部分。

## 目录结构

- `code/`
  - 结构化实验代码项目（Dimensionality Reduction pipeline）
  - 包含实验配置、运行脚本、可视化与测试
  - 详细说明见 [`code/README.md`](code/README.md)
- `report/`
  - 报告（LaTeX）项目目录
  - 用于撰写并导出最终提交报告

## 快速开始（代码部分）

```bash
cd code
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash scripts/run_quick.sh
```

## Git 说明

- 根目录 `.gitignore` 已覆盖：
  - VS Code 本地配置
  - macOS 系统文件
  - Python 缓存与虚拟环境
  - TeX 编译中间产物

