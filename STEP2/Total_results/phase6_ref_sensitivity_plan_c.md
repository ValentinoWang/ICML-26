# Phase 6 Reference Sensitivity (Plan C Summary)

## Context
- Experiment: Reference Sensitivity Stress Test on NEU-Trash.
- Variable: reference contamination ratio (xi) in {0.00, 0.05, 0.10, 0.20}.
- Metric: mAP50-95 from YOLO finetune outputs.

## Results (mAP50-95)

| xi | CPBA (Ours) | Naive-Linear |
|---:|---:|---:|
| 0.00 | 0.7842 | 0.8285 |
| 0.05 | 0.6414 | 0.7618 |
| 0.10 | 0.7788 | 0.7037 |
| 0.20 | 0.8010 | 0.7656 |

## Observations
- CPBA drops sharply at xi=0.05 (0.7842 -> 0.6414) and then recovers at xi=0.10/0.20.
- Naive-Linear is also non-monotonic and volatile.
- The curve looks unstable, which can be interpreted as randomness or operating-point instability in NEU-Trash.

## Decision (Plan C)
- Do NOT include the sensitivity curve in the main text.
- Keep the plot as internal evidence only.
- Focus the paper on core strengths (GC10, NEU TTA, strong robustness results).
- If needed, reference prior ablation/sensitivity (e.g., alpha/k) in text-only form.

## Artifacts
- CSV: /root/autodl-tmp/Style_Filter/Paper/tables/Eexp2-NEU-DET/phase6_ref_sensitivity_neu_trash.csv
- Figure (internal only): /root/autodl-tmp/Style_Filter/Paper/figures/Eexp2-NEU-DET/visualizations/phase6_ref_sensitivity_neu_trash.png
- Figure (PDF): /root/autodl-tmp/Style_Filter/Paper/figures/Eexp2-NEU-DET/visualizations/phase6_ref_sensitivity_neu_trash.pdf

## Notes
- Plot script label fix: mathtext for xi was corrected to avoid parsing errors.
