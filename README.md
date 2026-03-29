# Rebuttal Experiment Bundle (Anonymous Export)

This directory is the sanitized anonymous export of the rebuttal artifact bundle for reviewers `5p9M`, `kzcn`, `GqZq`, and `Bg6X`.

## Structure

- `shared/rebuttal_results/`
  - New rebuttal-only experiments from `Experiments/Rebuttal/`
  - Includes seed-level metrics, summary tables, clean-val ablations, CIFAR synthetic-fraction sweep, and the unsupervised GPT-2 surrogate results
- `shared/original_scripts/`
  - Original experiment scripts from `Experiments/exp5`, `exp6`, `exp8`, `exp9`, `exp10`, `exp11`, and `exp13`
  - Local absolute paths and machine-specific defaults were replaced with portable placeholders
- `shared/frozen_outputs/`
  - Frozen CSV/figure exports copied from `Paper/LaTEX/icml2026/Tables/` and `Paper/LaTEX/icml2026/Figures/`
  - These are included because several original experiment results are already materialized there as the paper-ready outputs
- `reviewers/`
  - Reviewer-specific lightweight folders pointing to the most relevant rebuttal artifacts
- `tex/rebuttal_exp_bundle.tex`
  - LaTeX overview of what is included and how it maps to each reviewer concern
- `PUSH_INSTRUCTIONS.md`
  - Safe procedure for creating a brand-new anonymous git repository without exposing prior history

## Reviewer Index

- `reviewers/5p9M/`
  - Clean-val size ablation
  - CIFAR synthetic-fraction sweep
  - GPT-2 unsupervised surrogate
  - Qwen MAUVE clarification note
- `reviewers/kzcn/`
  - `epsilon` bound note
  - GPT-2 clean-val size ablation
  - GPT-2 unsupervised surrogate
  - Compute table
- `reviewers/GqZq/`
  - Direct-clean-data rebuttal tables
  - CIFAR synthetic-fraction sweep
  - Original CIFAR seed tables
  - Qwen PPL trajectory figure
- `reviewers/Bg6X/`
  - `epsilon` bound note
  - GPT-2 unsupervised surrogate
  - Qwen MAUVE clarification
  - High-dimensional and ablation tables
  - Compute table

## Notes

- This bundle is intentionally selective rather than a verbatim copy of the entire `Experiments/` tree.
- The goal is to make the anonymous repository easy to inspect by reviewers while preserving the scripts and result files that are actually cited in rebuttal.
- For the original experiments where the local `Experiments/` tree mainly retains scripts and the frozen outputs were already exported to the paper tables/figures, both are included here.
- Absolute paths, local cache paths, and machine-specific launch paths were rewritten to relative or placeholder paths for anonymous release.
- This directory should be copied into a fresh folder and pushed from a brand-new git repository; do not attach the existing project history to a public anonymous remote.
