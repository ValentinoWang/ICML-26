# Sibylla Experiment Notes

本目录用于放置 **Sibylla / 先知模块** 的独立实验脚本。

## 实验文件索引

### 1. `run_projected_logsig_experiment.py`

- 作用：
  当前主实验。实现了高维潜轨迹生成、低秩投影、历史窗口加一步前瞻期望、弧弦比 `R` 诊断、动态阶数选择 `m in {1,2,3}`、截断 `log-signature` 与零填充固定宽度输入。
- 典型用途：
  验证 Sibylla 的“自适应几何诊断”思路是否优于固定阶数基线。
- 运行方式：

```bash
conda run -n base python "/Users/gordongauerk/Projects/Neutron's_machine_learning/Generalized-Recursive-Stability/sibylla_experiment/run_projected_logsig_experiment.py" --cpu
```

- 常用可选参数：

```bash
conda run -n base python "/Users/gordongauerk/Projects/Neutron's_machine_learning/Generalized-Recursive-Stability/sibylla_experiment/run_projected_logsig_experiment.py" \
  --cpu \
  --seed 67 \
  --train-per-family 600 \
  --test-per-family 200 \
  --epochs 150 \
  --out-dir "/Users/gordongauerk/Projects/Neutron's_machine_learning/Generalized-Recursive-Stability/sibylla_experiment/results_projected_logsig"
```

- 输出文件：
  - `results_projected_logsig/metrics.csv`
  - `results_projected_logsig/adaptive_orders.csv`
  - `results_projected_logsig/projected_logsig_summary.png`

### 2. `run_experiment.py`

- 作用：
  兼容入口。它直接调用 `run_projected_logsig_experiment.py` 的 `main()`，用于保留旧命令路径。
- 何时使用：
  仅当已有文档或旧命令还在引用 `run_experiment.py` 时使用。
- 运行方式：

```bash
conda run -n base python "/Users/gordongauerk/Projects/Neutron's_machine_learning/Generalized-Recursive-Stability/sibylla_experiment/run_experiment.py" --cpu
```

### 3. `run_legacy_signature_experiment.py`

- 作用：
  旧版原型实验。直接在二维路径上做普通截断 `signature`，不包含高维潜轨迹投影，也不包含一步前瞻窗口和 `log-signature`。
- 典型用途：
  与当前主实验做历史对照，回答“老实验图是什么意思”“新实验比老实验变化了什么”。
- 运行方式：

```bash
conda run -n base python "/Users/gordongauerk/Projects/Neutron's_machine_learning/Generalized-Recursive-Stability/sibylla_experiment/run_legacy_signature_experiment.py" --cpu
```

- 输出文件：
  - `results_legacy/metrics.csv`
  - `results_legacy/adaptive_orders.csv`
  - `results_legacy/legacy_signature_summary.png`

## 新实验的约定

- 每增加一个新实验，都新增一个独立脚本，命名建议为：
  - `run_<idea>_experiment.py`
- 不要直接覆盖已有实验脚本，除非用户明确要求。
- 每次新增实验后，都更新本文件，补充：
  - 实验脚本文件名
  - 实验目标
  - 运行命令
  - 输出文件位置

## 环境约定

- Python 运行统一优先使用：

```bash
conda run -n base python ...
```

- 默认结果目录是：

```text
/Users/gordongauerk/Projects/Neutron's_machine_learning/Generalized-Recursive-Stability/sibylla_experiment/results_projected_logsig
```
