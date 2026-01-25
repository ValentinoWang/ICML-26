# ICML Reviewer-Focused Ablation Plan

This plan targets the minimal, reviewer-facing evidence for Meta/Proxy-mode robustness and PPL leash reference stability. It is written to be runnable but does not execute anything by itself.

## Goal
Provide defensible, minimal-cost empirical evidence for:
1) clean-val size/quality robustness in Meta/Proxy-mode (CIFAR-10), and
2) sliding vs fixed PPL leash reference (GPT-2).

## Scope
- Default: single seed (1088), full 5 generations.
- Expand to 3 seeds only if reviewers require stronger statistical robustness.
- Outputs are organized for direct inclusion in a short main-text table + appendix detail.

---

## A. CIFAR-10 Clean-Val Size Ablation (No Noise)

**Purpose**: Show trend and failure point as clean validation size scales.

**Settings**:
- `--meta-clean-val`
- `--clean-val-source train_holdout` (avoid test leakage)
- `--clean-val-strategy stratified`
- `--clean-val-size 100 / 1000 / 10000`
- `seed=1088`, `mode=set_aware`, tuned hyperparameters

**Command (scripted)**:
```
SEEDS=1088 CLEAN_VAL_SIZES="100 1000 10000" CLEAN_VAL_NOISES="0.0" \
  bash exp9_cifar10_setaware/waste/scripts/run_cleanval_size_noise_ablation.sh
```

**Expected cost**: ~1–2 hours per run; total ~3–6 hours.

**Outputs**:
- `Total_results/Tables/exp9_cifar10_setaware/cleanval_ablation/size{SIZE}_noise0.0/exp9_seed1088_merged.csv`

---

## B. CIFAR-10 Clean-Val Label Noise Ablation (Fixed size=1k)

**Purpose**: Simulate imperfect human supervision; check robustness under label noise.

**Settings**:
- `--clean-val-size 1000`
- `--clean-val-noise-rate 0.0 / 0.1 / 0.2`
- other settings same as A

**Command (scripted)**:
```
SEEDS=1088 CLEAN_VAL_SIZES="1000" CLEAN_VAL_NOISES="0.0 0.1 0.2" \
  bash exp9_cifar10_setaware/waste/scripts/run_cleanval_size_noise_ablation.sh
```

**Expected cost**: ~3–6 hours total.

**Outputs**:
- `Total_results/Tables/exp9_cifar10_setaware/cleanval_ablation/size1000_noise{NOISE}/exp9_seed1088_merged.csv`

---

## C. GPT-2 PPL Leash Reference: Sliding vs Fixed

**Purpose**: Check for slow drift vs fixed anchor stability.

**Settings**:
- `--ppl-leash-strength 1.0 --ppl-leash-tau 0.7 --ppl-leash-mode upper`
- `--methods set_aware`
- `seed=1088`
- `--ppl-leash-ref-mode sliding` vs `fixed`

**Command template** (scripted at `exp11_gpt2_model/run_leash_ref_ablation.sh`):
```
python exp11_gpt2_model/run_exp11_gpt2_model.py \
  --device cuda \
  --seeds 1088 \
  --methods set_aware \
  ... (same as run_reproducible.sh) ... \
  --ppl-leash-ref-mode sliding \
  --results-path <sliding_path>

python exp11_gpt2_model/run_exp11_gpt2_model.py \
  --device cuda \
  --seeds 1088 \
  --methods set_aware \
  ... (same as run_reproducible.sh) ... \
  --ppl-leash-ref-mode fixed \
  --results-path <fixed_path>
```

**Expected cost**: ~6–10 hours per run; total ~12–20 hours.

**Outputs**:
- `Total_results/Tables/exp11_gpt2_model/Results/leash_ref_sliding/1088/metrics_diversity_ppl.csv`
- `Total_results/Tables/exp11_gpt2_model/Results/leash_ref_fixed/1088/metrics_diversity_ppl.csv`

---

## D. Reporting Plan (Main Text + Appendix)

**Core metrics**:
- CIFAR-10: `acc`, `worst_class_acc`, `ess_score`, `pseudo_label_hist` from merged CSV.
- GPT-2: `val_ppl`, `distinct4` from metrics CSV.

**Failure-point criterion**:
- clean-val too small or too noisy if
  - worst-class accuracy no longer improves vs baseline,
  - or ESS/class histogram shows collapse.

**Write-up**:
- Main text: small table with (size, noise) vs worst-class acc, plus sliding vs fixed PPL leash.
- Appendix: full CSV tables and exact commands.

---

## Optional Downscale (If Time-Constrained)
- Reduce to 3 generations, and lower epochs/steps:
  - `--gen0-epochs 10`, `--finetune-epochs 15`, `--filter-steps 100`
- Expected to cut runtime to ~1/2–1/3, with slightly noisier evidence.
