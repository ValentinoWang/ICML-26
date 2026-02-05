# Exp13: One-Shot LLM Validation (Qwen2)

Goal: a single G0 -> G1 jump showing Set-Aware filtering improves quality at low overhead on a modern LLM (Qwen2).
Target GPU: 4090D (24GB). Use Unsloth + LoRA for 4-bit training.

## Environment
```bash
# Create the env (name used in the paper/repro)
conda create -n kiwitest python=3.10
conda activate kiwitest
```

## Data
Provide JSONL with a `text` field.
- `seed.jsonl`: 2k high-quality texts (G0 SFT)
- `val.jsonl`: clean validation texts (PPL)

## Protocol (G0 -> G1)
1. G0 SFT (baseline adapter).
2. Generate 2k-5k candidates from G0.
3. Filter candidates:
   - Baseline: random 1k
   - Set-Aware: attention-based filter selects 1k
4. G1 SFT using each filtered set.
5. Evaluate PPL on clean validation.

## Scripts
All stages are in `run_exp12_oneshot.py`.

Default model in the examples below:
`unsloth/qwen2-7b-bnb-4bit`

### 1) Train G0 adapter
```bash
python exp13_Qwen_model/run_exp12_oneshot.py \
  g0_sft \
  --model-name unsloth/qwen2-7b-bnb-4bit \
  --train-jsonl data/seed.jsonl \
  --output-dir exp13_Qwen_model/outputs/g0_adapter
```

### 2) Generate candidates (G0 -> candidates)
```bash
python exp13_Qwen_model/run_exp12_oneshot.py \
  generate \
  --model-name unsloth/qwen2-7b-bnb-4bit \
  --adapter-dir exp13_Qwen_model/outputs/g0_adapter \
  --seed-jsonl data/seed.jsonl \
  --out-jsonl exp13_Qwen_model/outputs/candidates.jsonl \
  --num-candidates 3000
```

### 3) Train Set-Aware filter (cheap proxy)
```bash
python exp13_Qwen_model/run_exp12_oneshot.py \
  train_filter \
  --model-name unsloth/qwen2-7b-bnb-4bit \
  --seed-jsonl data/seed.jsonl \
  --candidates-jsonl exp13_Qwen_model/outputs/candidates.jsonl \
  --out-ckpt exp13_Qwen_model/outputs/filter_ckpt.pt
```

### 4) Select candidates
Baseline (random):
```bash
python exp13_Qwen_model/run_exp12_oneshot.py \
  select \
  --mode random \
  --candidates-jsonl exp13_Qwen_model/outputs/candidates.jsonl \
  --out-jsonl exp13_Qwen_model/outputs/baseline_selected.jsonl \
  --select-k 1000
```

Set-Aware:
```bash
python exp13_Qwen_model/run_exp12_oneshot.py \
  select \
  --mode set_aware \
  --model-name unsloth/qwen2-7b-bnb-4bit \
  --filter-ckpt exp13_Qwen_model/outputs/filter_ckpt.pt \
  --candidates-jsonl exp13_Qwen_model/outputs/candidates.jsonl \
  --out-jsonl exp13_Qwen_model/outputs/setaware_selected.jsonl \
  --select-k 1000
```

### 5) G1 SFT (baseline vs set-aware)
```bash
python exp13_Qwen_model/run_exp12_oneshot.py \
  g1_sft \
  --model-name unsloth/qwen2-7b-bnb-4bit \
  --adapter-dir exp13_Qwen_model/outputs/g0_adapter \
  --train-jsonl exp13_Qwen_model/outputs/baseline_selected.jsonl \
  --output-dir exp13_Qwen_model/outputs/g1_base

python exp13_Qwen_model/run_exp12_oneshot.py \
  g1_sft \
  --model-name unsloth/qwen2-7b-bnb-4bit \
  --adapter-dir exp13_Qwen_model/outputs/g0_adapter \
  --train-jsonl exp13_Qwen_model/outputs/setaware_selected.jsonl \
  --output-dir exp13_Qwen_model/outputs/g1_ours
```

### 6) Evaluate PPL
```bash
python exp13_Qwen_model/run_exp12_oneshot.py \
  eval \
  --model-name unsloth/qwen2-7b-bnb-4bit \
  --adapter-dir exp13_Qwen_model/outputs/g1_base \
  --val-jsonl data/val.jsonl

python exp13_Qwen_model/run_exp12_oneshot.py \
  eval \
  --model-name unsloth/qwen2-7b-bnb-4bit \
  --adapter-dir exp13_Qwen_model/outputs/g1_ours \
  --val-jsonl data/val.jsonl
```

## Notes
- Filter training uses a proxy objective that aligns the weighted candidate embedding with the clean-seed mean.
- This is intentionally minimal: one generation (G0 -> G1) only.
