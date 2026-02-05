# SAGE-Filter (Release Package)

This folder contains a lightweight, self-contained code snapshot for the paper:
`Generalized Recursive Stability: Mitigating Model Collapse in Biased Estimation via Set-Aware Geometric Filtering`.

The goal is to provide:
- The core SAGE-Filter implementation (set-aware reweighting + explicit drift correction head)
- A small, fast regression demo that runs on CPU in minutes

Notes:
- This release package intentionally excludes large datasets, model checkpoints, and experiment logs.
- Do **NOT** commit real API keys. Use `.env.example` as a template if needed.
- For a concise “what is included/excluded” checklist, see `PLAN.md`.

## Contents

- `bias/`: simple biased estimators (ridge, Bayes MAP)
- `filter/`: SAGE-Filter + pointwise baselines + toy demo runner
- `Common_Utils/`: deterministic seeding helpers (optional)
- `Experiments/` (optional): code-only snapshots of the full experiment scripts (no datasets / no logs / no checkpoints)
- `requirements_full.txt` (optional): extra dependencies for running some full experiments

## Installation

Create a clean Python environment (Python 3.10+ recommended) and install minimal deps:

```bash
pip install -r requirements.txt
```

This demo requires `torch`, `numpy`, `matplotlib`. GPU is optional.

## Quickstart (Regression Demo)

Run a small biased-recursion regression simulation:

```bash
python filter/run_filter_experiment.py --cpu --out-dir results_demo
```

Or equivalently:

```bash
bash run.sh demo
```

Outputs:
- `results_demo/set_filter_metrics.csv`
- `results_demo/set_filter_plot.png`

## Reproducing Full Paper Experiments

The full experimental pipelines (CIFAR-10, GPT-2/Qwen2-7B recursion, MAUVE, etc.) require additional dependencies
and data/model caches. For transparency, this release includes a **code-only** snapshot under `Experiments/`.

To install optional dependencies:

```bash
pip install -r requirements_full.txt
```

Important: the heavy experiments may still require external datasets/models and substantial compute.
See `REPRODUCIBILITY.md` (reproduction matrix) and `Experiments/README.md` (how to run the snapshots).
For a Chinese end-to-end run guide, see `MANUAL.md`.
