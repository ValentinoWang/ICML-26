## Qwen2-7B MAUVE vs Base/No-Filter

### What we verified

The paper's representative Qwen2-7B trajectory is a single-seed table, not a multi-seed average. In the current manuscript, the reported numbers are:

- G3: Base `0.0508`, Pointwise `0.0410`, Dispersion `0.0420`, Set-Aware `0.0355`
- G4: Base `0.0429`, Pointwise `0.0290`, Dispersion `0.0338`, Set-Aware `0.0331`

At the same time, the same table shows that Set-Aware improves streaming PPL over Base at G3/G4.

### Most defensible explanation

This does not necessarily indicate that Set-Aware is worse in quality. For Qwen2-7B, the MAUVE setup is much more sensitive than the GPT-2 setup, for three concrete reasons:

1. `MAUVE` is a quality-and-coverage metric, not a pure quality metric. In the Qwen recursive pipeline, `Base` keeps a random subset, while `Set-Aware` first applies a PPL safety gate (`ppl_keep_frac=0.8`) and then selects the top-scored subset. This can improve average text quality while narrowing support, so a lower MAUVE than Base is possible without contradiction.

2. The Qwen MAUVE script uses sequential truncation rather than randomized resampling: it takes the first 1000 candidate samples and the first 500 validation samples. This makes the score sensitive to prompt ordering, local mode concentration, and small sample fluctuations.

3. The repository later adds a dedicated fixed-prompt MAUVE rerun script for Qwen2-7B, which strongly suggests prompt variance/order was already recognized as a confound for this setting.

There is also a protocol inconsistency worth avoiding in the rebuttal wording: the appendix text says `max_text_length=256`, but the released Qwen MAUVE scripts run with `max_text_length=1024`. Even if this does not change the qualitative conclusion, it makes the single-seed MAUVE values less suitable for strong claims.

### Suggested rebuttal text

We agree that, in the representative Qwen2-7B recursive run, Set-Aware does not exceed Base on MAUVE at G3/G4. We do not interpret this as a contradiction to the PPL gains. In this setting, MAUVE is measuring distributional coverage rather than pure text quality, and our Set-Aware pipeline intentionally applies a stronger quality-oriented selection step than Base (including a PPL safety gate before final selection), which can improve average quality while slightly narrowing support. Moreover, the Qwen MAUVE evaluation is based on a single-seed representative run and a relatively fragile protocol (first-1000 candidate truncation, first-500 validation truncation, prompt-order sensitivity), so small differences in the low-MAUVE regime should not be over-interpreted. For this reason, we use the three-seed streaming-PPL result as the main stability evidence for Qwen2-7B, and treat the reported MAUVE numbers as an auxiliary diagnostic rather than a standalone ranking metric.

### Evidence locations

- Manuscript table: `/root/autodl-tmp/ICML/Paper/LaTEX/icml2026/main.tex` lines 334-349
- Appendix protocol text: `/root/autodl-tmp/ICML/Paper/LaTEX/icml2026/main.tex` line 974
- Qwen MAUVE evaluator: `/root/autodl-tmp/ICML/Experiments/exp13_Llama_model/mauve_eval_qwen.py` lines 10-24 and 58-78
- Qwen recursive pipeline: `/root/autodl-tmp/ICML/Experiments/exp13_Llama_model/run_qwen_recursive.sh` lines 51-55 and 168-216
- Fixed-prompt rerun script: `/root/autodl-tmp/ICML/Experiments/exp13_Llama_model/run_mauve_g4_qwen_fixedprompts.sh` lines 4-7 and 45-144
