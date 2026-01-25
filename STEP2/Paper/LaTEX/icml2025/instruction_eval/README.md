# Minimal instruction-following eval (A/B win-rate)

This folder contains a tiny instruction set plus scripts to generate answers and compare Pointwise vs Set-Aware with an LLM judge.

## 1) Prompts
- Default: `instruction_eval/prompts_50.jsonl` (50 ASCII prompts).
- You can replace or extend it with your own JSONL format: `{"id": 1, "prompt": "..."}`.
 - To use a different file, set `PROMPTS_PATH`, e.g. `instruction_eval/prompts_100.jsonl`.

## 2) Generate responses
Example (Qwen2-7B, G4 Pointwise vs G4 Set-Aware):

```bash
python instruction_eval/run_generate.py \
  --model-name unsloth/qwen2-7b-bnb-4bit \
  --adapter-dir /root/autodl-tmp/ICML/STEP2/exp13_Llama_model/outputs/rec_seed1088_qwen_g0_g4_b1p5/adapter_g4_pointwise \
  --prompts-jsonl instruction_eval/prompts_50.jsonl \
  --out-jsonl instruction_eval/pointwise_g4.jsonl \
  --load-in-4bit \
  --template instruct

python instruction_eval/run_generate.py \
  --model-name unsloth/qwen2-7b-bnb-4bit \
  --adapter-dir /root/autodl-tmp/ICML/STEP2/exp13_Llama_model/outputs/rec_seed1088_qwen_g0_g4_b1p5/adapter_g4_set_aware \
  --prompts-jsonl instruction_eval/prompts_50.jsonl \
  --out-jsonl instruction_eval/set_aware_g4.jsonl \
  --load-in-4bit \
  --template instruct
```

## 3) Judge with Gemini
Set API key and run pairwise judgment (A = ours, B = baseline).

```bash
export GEMINI_API_KEY=YOUR_KEY
python instruction_eval/run_judge_gemini.py \
  --responses-a instruction_eval/set_aware_g4.jsonl \
  --responses-b instruction_eval/pointwise_g4.jsonl \
  --out-jsonl instruction_eval/gemini_judgments.jsonl \
  --shuffle
```

## 4) Summarize
```bash
python instruction_eval/summarize_results.py \
  --results-jsonl instruction_eval/gemini_judgments.jsonl \
  --label-a set_aware \
  --label-b pointwise \
  --latex-out instruction_eval/if_table_row.tex
```

## One-shot script
If you prefer a single script instead of CLI args, run:
```bash
AI_API_KEY=YOUR_KEY python instruction_eval/run_eval_gemini3.py
```
This uses the cached Qwen2-7B snapshot and Gemini via the proxy endpoint.
You can swap judges, for example:
```bash
AI_API_KEY=YOUR_KEY GEMINI_MODEL=gemini-3-flash-preview JUDGE_MAX_TOKENS=128 \
  python instruction_eval/run_eval_gemini3.py
```
For `gemini-3-pro-preview`, set `JUDGE_MAX_TOKENS>=1024` to avoid empty responses (we used 8192).
If you want to keep multiple result files, set `OUTPUT_TAG`, e.g. `OUTPUT_TAG=flash_100`.
