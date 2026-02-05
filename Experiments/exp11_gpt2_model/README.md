# exp11_gpt2_model

Recursive GPT-2 collapse experiment with set-aware mitigation. The script can run multiple pipelines in parallel across generations: `no_filter`, `pointwise` (perplexity-based), `ppl_safety` (drop extreme-PPL candidates only), `rep_filter` (hard n-gram repetition filter), `dispersion` (non-learned geometric weights from embedding dispersion), and `set_aware` (MiniLM embeddings + transformer set filter). Defaults follow the 2k-subset protocol (10k candidates → 2k training samples, 3 epochs per gen, 128-token sequences).

## Files (modular, high cohesion / low coupling)
- `run_exp11_gpt2_model.py` — orchestrates the experiment loop only.
- `data.py` — seed, tokenizer/model prep, Wikitext subset loader, tokenization helpers.
- `training.py` — GPT-2 fine-tuning, per-sample PPL, validation PPL.
- `generation.py` — GPT-2 sampling and MiniLM embedding.
- `filter_module.py` — set-aware filter training/inference and selection logic.
- `metrics.py` — Distinct-n computation.

## Quick start
```bash
pip install torch transformers datasets
python Experiments/exp11_gpt2_model/run_exp11_gpt2_model.py --generations 5 --candidate-pool 10000 --train-samples 2000 --save-checkpoints
```
The script will download GPT-2 small and Wikitext-103 (via `datasets`). It also pulls `sentence-transformers/all-MiniLM-L6-v2` for embeddings (handled inside `transformers.AutoModel`/`AutoTokenizer`).

## Key arguments
- Data/loop: `--wikitext-train-size` (50k–100k), `--candidate-pool` (10k), `--train-samples` (2k), `--generations` (5–10), `--max-new-tokens` (128).
- Filtering: `--filter-set-size` (1024), `--filter-steps` (200), `--filter-knn` (32), `--filter-temperature`, `--filter-ess-tau`, `--filter-ess-weight`.
- Filtering (Δϕ): `--delta-phi-scale` to enable/scale Δϕ logit correction inside the set-aware filter (set to `0.0` to disable).
- Semantic leash (PPL): `--ppl-leash-mode`/`--ppl-leash-tau`/`--ppl-leash-strength` penalize selecting high-PPL candidates relative to clean validation `val_ppl`.
- Training: `--epochs-per-gen` (3), `--lr` (5e-5), `--train-batch-size` (8), `--initial-epochs` (1) for a light Wikitext warm-up.
- Evaluation: `--eval-sample-size` (Distinct-n), `--val-eval-size` (validation PPL size).
- Logging: `--results-path` (JSON log), `--save-checkpoints` to dump per-generation model weights.
- Stabilization: `--mix-original-ratio` to mix a fraction of original Wikitext samples into each generation's train set (e.g., 0.1 for 10% anchors).
- Runtime: `--methods set_aware` to run only the modified method and skip baselines.
- Repro: `run_dispersion.sh` runs the `dispersion` baseline under the same protocol.

## Outputs
- 默认基路径 `Experiments/Total_results/Tables/exp11_gpt2_model/Results/metrics_diversity_ppl.json`（仓库根相对路径），并为每个 seed 写入 `<base_dir>/<seed>/<base_name>.{json,csv}`。如传入 `--results-path <path/to/your.json>`，会在 `<path/to>/<seed>/<your.json|.csv>` 下生成。Checkpoints 在对应 seed 目录内（开启 `--save-checkpoints` 时）。
- 新版 CSV 会在 `val_ppl / distinct2-4` 之外，额外记录训练集质量侧统计：`train_unique_line_ratio`、`train_rep4_intra`、`train_gzip_ratio`、`train_avg_words`（对齐当前代被选中的训练文本；用于检测复读/塌缩与“胡言乱语”风险）。

## Repro scripts
- `run_reproducible.sh`: dphi1 + dphi1_leash (main results)
- `run_dispersion.sh`: non-learned geometric baseline
- `run_ppl_safety.sh`: pointwise safety baseline
- `run_rep_filter.sh <thr>`: repetition filter baseline (e.g., `0.6`)
- `run_pointwise_mix_original.sh <ratio>`: mix-original baseline (e.g., `0.5`)

## Practical tips (RTX 4090)
- Keep `--generation-batch` at 64–128 so the filter sees large sets; `--embed-batch-size` at 64 keeps MiniLM fast.
- For a ~2.5–3 h full 10-gen run: candidate_pool=10k, train_samples=2k, epochs_per_gen=3, filter_steps≈200.
