# Reproducibility Notes (ICML)

This repository snapshot is intended to make the **core method** easy to run and inspect.
The full paper includes additional large-scale experiments (e.g., CIFAR-10 and LMs) that
require extra dependencies, datasets, and cached model checkpoints.

## What This Release Covers

- Core SAGE-Filter modules:
  - set-aware model (`filter/set_aware/`)
  - pointwise baseline (`filter/standard/`)
  - training losses and a small regression demo runner
- A code-only snapshot of the full experiment entrypoints under `Experiments/`
  (scripts only; no `Total_results/`, no datasets, no checkpoints).

## Quickstart

```bash
pip install -r requirements.txt
bash run.sh demo
```

The demo generates:
- `results_demo/set_filter_metrics.csv`
- `results_demo/set_filter_plot.png`

## Optional: Full-Experiment Dependencies

Some experiment scripts (especially LM / CIFAR-10) require extra packages:

```bash
pip install -r requirements_full.txt
```

## Reproduction Matrix (What You Can Re-Run)

This compact release is organized around a **minimal runnable demo** plus **code-only experiment scripts**.
Some scripts require external datasets / pretrained models (not bundled here).

Legend:
- `requirements.txt`: minimal demo deps (CPU-friendly)
- `requirements_full.txt`: optional deps for vision/LLM experiments
- "Data": whether the experiment is synthetic, auto-download via `torchvision` / `datasets`, or user-provided

| Category | Experiment | Entry point (from `Release/`) | Data | Deps | Typical compute | Notes / outputs |
|---|---|---|---|---|---|---|
| Core (toy) | Regression demo (minutes) | `python filter/run_filter_experiment.py --cpu --out-dir results_demo` | Synthetic | minimal | CPU | Writes `results_demo/*.csv,*.png` |
| Toy / regression | Bias sources sweep | `python Experiments/exp1_bias_sources/run_exp1_bias_sources.py` | Synthetic | full | CPU/GPU | Writes under `Experiments/Total_results/.../exp1_bias_sources/` |
| Toy / regression | Bias sensitivity | `python Experiments/exp2_bias_sensitivity/run_exp2_bias_sensitivity.py` | Synthetic | full | CPU/GPU | Writes under `Experiments/Total_results/.../exp2_bias_sensitivity/` |
| Toy / regression | Data efficiency | `python Experiments/exp3_data_efficiency/run_exp3_data_efficiency.py` | Synthetic | full | CPU/GPU | Writes under `Experiments/Total_results/.../exp3_data_efficiency/` |
| Toy / regression | Bias correction viz | `python Experiments/exp4_bias_correction_visualization/run_exp4_bias_correction_visualization.py` | Synthetic | full | CPU/GPU | Writes figures/tables under `Experiments/Total_results/.../exp4_bias_correction_visualization/` |
| Toy / regression | High-dim scalability | `python Experiments/exp5_high_dim_scalability/run_exp5_high_dim_scalability.py` | Synthetic | full | CPU/GPU | Writes under `Experiments/Total_results/.../exp5_high_dim_scalability/` |
| Toy / regression | Architecture ablation | `python Experiments/exp6_arch_ablation/run_exp6_arch_ablation.py` | Synthetic | full | CPU/GPU | Writes under `Experiments/Total_results/.../exp6_arch_ablation/` |
| Toy / regression | Gated ablation | `python Experiments/exp6_arch_ablation_gated/run_exp6_arch_ablation.py` | Synthetic | full | CPU/GPU | Writes under `Experiments/Total_results/.../exp6_arch_ablation_gated/` |
| Regression (real data) | Recursive regression | `python Experiments/exp7_recursive_regression/run_exp7_recursive_regression.py` | California Housing (via `sklearn`, first run downloads) | full | CPU/GPU | Writes under `Experiments/Total_results/.../exp7_recursive_regression/` |
| Toy / regression | Variance/attention study | `python Experiments/exp7_variance_attention/run_exp7_variance_attention.py` | Synthetic | full | CPU/GPU | Figure script uses `sklearn/pandas/seaborn` |
| Vision | MNIST recursive | `python Experiments/exp8_mnist_recursive/run_exp8_mnist_recursive.py --cpu` | MNIST (`torchvision`) | full | CPU (slow) / GPU | By default writes under `Experiments/Total_results/.../exp8_mnist_recursive/` |
| Vision | CIFAR-10 set-aware | `python Experiments/exp9_cifar10_setaware/run_exp9_cifar10_setaware.py --device cpu` | CIFAR-10 (`torchvision`) | full | GPU recommended | On CPU-only machines, set `--device cpu` explicitly |
| LLM | GPT-2 recursion (multi-gen) | `python Experiments/exp11_gpt2_model/run_exp11_gpt2_model.py --device cuda` | Wikitext (`datasets`) + GPT-2 (`transformers`) | full | GPU recommended | Uses HF caches; can run offline if models/data are pre-cached |
| LLM | Embedding topology analysis | `python Experiments/exp12_Embedding_model/run_exp12_embedding_topology.py --help` | Text files from exp11 | full | CPU/GPU | Analysis depends on saved generations (not bundled) |
| LLM (one-shot) | Qwen2 G0→G1 | `python Experiments/exp13_Qwen_model/run_exp12_oneshot.py --help` | user JSONL (`text` field) | full + extra | 24GB GPU class | Requires Unsloth/LoRA stack (not in minimal deps) |

## How To Run Experiment Scripts From This Release

The `Experiments/` folder here is a **code-only snapshot**. To make imports work, run from `Release/`
and ensure both `Release/` and `Release/Experiments/` are on `PYTHONPATH`.

Option A (recommended): use the experiment runner (it sets `PYTHONPATH` for subprocesses):

```bash
python Experiments/run_all_experiments.py --dry-run
python Experiments/run_all_experiments.py --only exp1_bias_sources exp2_bias_sensitivity --dry-run
```

Option B: run a single experiment directly:

```bash
export PYTHONPATH="$(pwd):$(pwd)/Experiments:${PYTHONPATH:-}"
python Experiments/exp8_mnist_recursive/run_exp8_mnist_recursive.py --cpu --generations 3 --seeds 1088
```

Per-experiment dependencies are also listed under each experiment folder, e.g.:

```bash
pip install -r Experiments/exp8_mnist_recursive/requirements.txt
```

## Determinism / Seeds

The demo uses a fixed RNG seed (see `filter/data.py:set_seed`) and can be overridden via
the CLI argument `--seed` in `filter/run_filter_experiment.py`.

## Hardware

- The regression demo runs on CPU.
- GPU is optional (not required for the minimal demo).

## Notes On Full Experiments

The full experimental pipelines used to generate the paper figures live in the main
workspace (outside this compact release) and may require:
- `transformers`, `datasets`, MAUVE evaluation, and cached LMs
- CIFAR-10 data and training code
- substantial GPU memory/time for multi-generation LM recursion

If you need the full reproduction package, we recommend releasing a separate artifact
with:
- pinned dependencies (conda env / docker)
- scripts that download datasets/models
- the exact commands used to produce each figure/table
