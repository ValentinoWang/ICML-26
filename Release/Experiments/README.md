# Experiments (Code-Only Snapshot)

This folder contains a **code-only** snapshot of the experiment entrypoints used in the full project.
It is included for transparency and for readers who want to re-run selected experiments.

Not included (by design):
- `Total_results/` (large logs/figures/tables)
- datasets / caches (e.g., MNIST/CIFAR downloads, Hugging Face `datasets` cache)
- pretrained model weights / checkpoints
- large generated artifacts

## Install Optional Dependencies

From the repository root:

```bash
pip install -r requirements_full.txt
```

Each experiment folder also ships a small `requirements.txt` so you can install only what you need:

```bash
pip install -r Experiments/exp1_bias_sources/requirements.txt
```

## Recommended: Dry-Run First

The runner prints the exact commands without executing them:

```bash
python Experiments/run_all_experiments.py --dry-run
python Experiments/run_all_experiments.py --only exp1_bias_sources exp2_bias_sensitivity --dry-run
```

## Running A Subset

```bash
python Experiments/run_all_experiments.py --only exp1_bias_sources exp2_bias_sensitivity exp7_recursive_regression
```

## Output Locations

Most scripts write under:
- `Experiments/Total_results/Tables/<exp_name>/...`
- `Experiments/Total_results/Figures/<exp_name>/...`

relative to the repository root.

## Notes For Vision / LLM Experiments

- Vision (MNIST/CIFAR): scripts use `torchvision`. If you are offline, pre-download datasets or point to a local cache
  via the experiment's `--data-root` (if provided).
- LLM (GPT-2 / Qwen / Llama): scripts use Hugging Face `transformers`/`datasets` and will download models/datasets
  unless they are already cached. To run offline, pre-cache artifacts and set:
  `HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`.
