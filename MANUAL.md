# ICML 复现/运行总手册（Release 版）

本手册面向两类读者：
- 想快速验证“核心方法确实能跑、确实有改进趋势”的 reviewer（CPU 分钟级）。
- 想按需复跑论文中不同实验（vision / LLM / ablation）的读者（可能需要额外依赖、数据下载、GPU）。

注意：本仓库是一个“轻量可读 + 可跑”的代码快照，**不包含**大体量数据、模型 checkpoint、以及 `Total_results/` 等实验输出（这些输出可再生，且体积巨大）。

---

## 0. 目录与入口（你应该从哪里开始）

在仓库根目录下：
- 核心方法：`filter/`、`bias/`
- 最小 demo（推荐 reviewer 先跑这个）：`bash run.sh demo`
- 全实验脚本快照：`Experiments/`
- 每个实验的独立依赖：`Experiments/exp*/requirements.txt`
- 全量可选依赖（一个文件装齐大多数实验）：`requirements_full.txt`
- 复现矩阵（范围/资源/数据说明）：`REPRODUCIBILITY.md`

---

## 1. 最小复现（CPU，分钟级）

目标：不下载任何数据/模型，也能看到 set-aware + Δϕ 的效果曲线。

```bash
pip install -r requirements.txt
bash run.sh demo
```

输出在：
- `results_demo/set_filter_metrics.csv`
- `results_demo/set_filter_plot.png`

---

## 2. 运行实验的通用规则（exp1-exp13 都适用）

### 2.1 安装依赖（两种方式）

方式 A：只跑某一个实验（更轻量）

```bash
pip install -r Experiments/exp1_bias_sources/requirements.txt
```

方式 B：想跑多组实验（更省事）

```bash
pip install -r requirements_full.txt
```

### 2.2 运行方式（两种方式）

方式 A：用统一 runner（推荐；自动设置 `PYTHONPATH`）

```bash
python Experiments/run_all_experiments.py --dry-run
python Experiments/run_all_experiments.py --only exp1_bias_sources exp2_bias_sensitivity
```

方式 B：直接跑单个脚本（需要设置 `PYTHONPATH`）

```bash
export PYTHONPATH="$(pwd):$(pwd)/Experiments:${PYTHONPATH:-}"
python Experiments/exp8_mnist_recursive/run_exp8_mnist_recursive.py --cpu --help
```

### 2.3 输出路径约定（重要）

大多数实验脚本默认把 CSV/PNG/JSON 写到：
- `Experiments/Total_results/Tables/<exp_name>/...`
- `Experiments/Total_results/Figures/<exp_name>/...`

这个目录会在首次运行时自动创建；它不属于“代码上传包”的内容（可再生且容易很大）。

---

## 3. exp1-exp13 实验索引（按类别）

下面每个实验目录都包含一个 `requirements.txt`，可直接 `pip install -r`。

### 3.1 Toy / Regression / Ablation（CPU/GPU 都可）

- exp1：`Experiments/exp1_bias_sources/`
  - 入口：`run_exp1_bias_sources.py`
  - 依赖：`Experiments/exp1_bias_sources/requirements.txt`
  - 数据：合成
  - 备注：`plot_exp1.py` 默认 `text.usetex=True`，若系统没装 LaTeX 会报错；可改为 False 或安装 TeXLive。

- exp2：`Experiments/exp2_bias_sensitivity/`
  - 入口：`run_exp2_bias_sensitivity.py`
  - 数据：合成

- exp3：`Experiments/exp3_data_efficiency/`
  - 入口：`run_exp3_data_efficiency.py`
  - 数据：合成

- exp4：`Experiments/exp4_bias_correction_visualization/`
  - 入口：`run_exp4_bias_correction_visualization.py`
  - 数据：合成
  - 备注：该实验的绘图脚本会自动检测 LaTeX（有则用，无则回退）。

- exp5：`Experiments/exp5_high_dim_scalability/`
  - 入口：`run_exp5_high_dim_scalability.py`
  - 数据：合成

- exp6：`Experiments/exp6_arch_ablation/` + `Experiments/exp6_arch_ablation_gated/`
  - 入口：各自的 `run_exp6_arch_ablation.py`
  - 数据：合成

- exp7（回归数据集递归）：`Experiments/exp7_recursive_regression/`
  - 入口：`run_exp7_recursive_regression.py`
  - 数据：`sklearn.datasets.fetch_california_housing`（首次运行会下载；离线需提前缓存）

- exp7（方差/attention 诊断）：`Experiments/exp7_variance_attention/`
  - 入口：`run_exp7_variance_attention.py`
  - 数据：合成
  - 备注：绘图/tsne 依赖 `pandas`、`scikit-learn`

### 3.2 Vision（MNIST / CIFAR-10）

- exp8（MNIST recursive）：`Experiments/exp8_mnist_recursive/`
  - 入口：`run_exp8_mnist_recursive.py`
  - 数据：`torchvision.datasets.MNIST(download=True)`（首次运行会下载）
  - CPU 可跑但慢；建议小 generations 先 smoke test

- exp9（CIFAR-10 set-aware）：`Experiments/exp9_cifar10_setaware/`
  - 入口：`run_exp9_cifar10_setaware.py`
  - 数据：`torchvision.datasets.CIFAR10(download=True)`（首次运行会下载）
  - 说明：CPU-only 机器请显式加 `--device cpu`（会很慢）；GPU 更现实

### 3.3 LLM（GPT-2 / Embedding topology / Qwen-Llama one-shot）

- exp11（GPT-2 recursion）：`Experiments/exp11_gpt2_model/`
  - 入口：`run_exp11_gpt2_model.py`
  - 数据/模型：Hugging Face `datasets` + `transformers`（首次运行会下载；离线需要提前缓存）
  - MAUVE：如需 `mauve_eval.py`，需要额外安装 `mauve-text`

- exp12（Embedding topology）：`Experiments/exp12_Embedding_model/`
  - 入口：`run_exp12_embedding_topology.py`
  - 依赖：`sentence-transformers` + `scikit-learn`
  - 数据：需要 exp11 生成的文本文件（本 release 不内置）

- exp13（Qwen2 one-shot G0->G1）：`Experiments/exp13_Qwen_model/`
  - 入口：`run_exp12_oneshot.py`
  - 数据：用户自备 JSONL（每行含 `text` 字段）
  - 备注：该实验依赖 Unsloth/LoRA/4-bit 训练栈；见 `Experiments/exp13_Qwen_model/README.md`

### 3.4 计算开销（benchmark）

- exp10（time/memory cost）：`Experiments/exp10_time_cost/`
  - 入口：`run_exp10_time_cost.py`
  - 备注：GPU 上用 `torch.cuda.Event` 更准；CPU 上会 fallback 到 `perf_counter`

---

## 4. 推荐的“审稿人友好”复现策略（不追求跑全套）

如果你希望在有限时间内最大化复现说服力，建议按顺序：
1) 先跑最小 demo：`bash run.sh demo`
2) 选 1-2 个合成实验（exp1/exp5/exp6）验证 ablation/规模趋势
3) 选 1 个 vision（exp8 或 exp9）验证“非合成”的可行性
4) LLM（exp11/exp13）作为“有资源再跑”的补充（文档提供可复跑入口与依赖）
