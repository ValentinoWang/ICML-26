import json
import os
import pathlib
import random
import sys
import time
from typing import Dict, List, Tuple

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import run_generate
import run_judge_gemini


# ======== CONFIG ========
PROMPTS_PATH = pathlib.Path(
    os.environ.get("PROMPTS_PATH", str(SCRIPT_DIR / "prompts_50.jsonl"))
)
OUTPUT_DIR = SCRIPT_DIR

# Local cached model snapshot (offline).
MODEL_SNAPSHOT = pathlib.Path(
    "/root/.cache/huggingface/hub/models--unsloth--qwen2-7b-bnb-4bit/snapshots/1239d1b09cba92f486e612d73b18c59ea6f8af3a"
)

ADAPTER_POINTWISE = pathlib.Path(
    "/root/autodl-tmp/ICML/STEP2/exp13_Llama_model/outputs/rec_seed1088_qwen_g0_g4_b1p5/adapter_g4_pointwise"
)
ADAPTER_SET_AWARE = pathlib.Path(
    "/root/autodl-tmp/ICML/STEP2/exp13_Llama_model/outputs/rec_seed1088_qwen_g0_g4_b1p5/adapter_g4_set_aware"
)

OUT_POINTWISE = OUTPUT_DIR / "pointwise_g4.jsonl"
OUT_SET_AWARE = OUTPUT_DIR / "set_aware_g4.jsonl"

GEMINI_ENDPOINT = os.environ.get("GEMINI_ENDPOINT", "https://api.zhizengzeng.com/google")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
API_KEY = os.environ.get("AI_API_KEY") or os.environ.get("GEMINI_API_KEY")

MODEL_TAG = GEMINI_MODEL.replace("/", "_").replace(".", "_")
OUTPUT_TAG = os.environ.get("OUTPUT_TAG")
if OUTPUT_TAG:
    OUT_JUDGMENTS = OUTPUT_DIR / f"judgments_{OUTPUT_TAG}.jsonl"
else:
    OUT_JUDGMENTS = OUTPUT_DIR / f"judgments_{MODEL_TAG}.jsonl"

SEED = 1088
BATCH_SIZE = 2
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.2
TOP_P = 0.9
REPETITION_PENALTY = 1.05
MAX_SEQ_LEN = 1024
SHUFFLE = True

MAX_RETRIES = 6
SLEEP_SEC = 3.0
JUDGE_MAX_TOKENS = int(os.environ.get("JUDGE_MAX_TOKENS", "128"))


def _set_offline_env() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("UNSLOTH_DISABLE_COMPILE", "1")
    os.environ.setdefault("UNSLOTH_DISABLE_PATCHING", "1")


def _find_model_snapshot(path: pathlib.Path) -> pathlib.Path:
    if path.exists():
        return path
    snapshots = sorted(path.parent.glob("snapshots/*"))
    if not snapshots:
        raise FileNotFoundError(f"Missing model snapshot under {path.parent}")
    return snapshots[-1]


def _generate_responses(adapter_dir: pathlib.Path, out_path: pathlib.Path) -> None:
    run_generate.set_seed(SEED)
    items = run_generate.load_prompts(PROMPTS_PATH, limit=None)
    prompts = [run_generate.format_prompt(item["prompt"], "instruct") for item in items]

    model_dir = _find_model_snapshot(MODEL_SNAPSHOT)
    model, tokenizer, _ = run_generate.load_model_and_tokenizer(
        str(model_dir),
        adapter_dir,
        max_seq_len=MAX_SEQ_LEN,
        load_in_4bit=True,
    )
    model.eval()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        total = len(prompts)
        for start in range(0, total, BATCH_SIZE):
            batch_prompts = prompts[start : start + BATCH_SIZE]
            outputs = run_generate.generate_batch(
                model,
                tokenizer,
                batch_prompts,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                repetition_penalty=REPETITION_PENALTY,
            )
            for i, full_text in enumerate(outputs):
                item = items[start + i]
                prompt_text = batch_prompts[i]
                response = full_text
                if full_text.startswith(prompt_text):
                    response = full_text[len(prompt_text) :].strip()
                response = run_generate.trim_response(response)
                record = {
                    "id": item["id"],
                    "prompt": item["prompt"],
                    "response": response,
                }
                f.write(json.dumps(record, ensure_ascii=True) + "\n")
            done = min(start + len(batch_prompts), total)
            print(f"Generated {done}/{total} prompts")


def _load_existing(path: pathlib.Path) -> Dict[int, dict]:
    if not path.exists():
        return {}
    data = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            data[int(obj["id"])] = obj
    return data


def _needs_regen(path: pathlib.Path, expected_total: int) -> bool:
    if not path.exists():
        return True
    lines = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                lines.append(line)
    if len(lines) != expected_total:
        return True
    for line in lines:
        try:
            obj = json.loads(line)
        except Exception:
            return True
        resp = obj.get("response", "")
        if "\nInstruction:" in resp or "\n\nInstruction:" in resp:
            return True
        if "\n### Instruction" in resp:
            return True
    return False


def _judge_pairwise() -> None:
    if not API_KEY:
        raise SystemExit("Missing API key in AI_API_KEY or GEMINI_API_KEY.")

    data_a = run_judge_gemini.load_responses(str(OUT_SET_AWARE))
    data_b = run_judge_gemini.load_responses(str(OUT_POINTWISE))
    ids = sorted(set(data_a.keys()) & set(data_b.keys()))
    if not ids:
        raise SystemExit("No overlapping ids between response files.")

    url = run_judge_gemini.build_url(GEMINI_ENDPOINT, GEMINI_MODEL, API_KEY)

    existing = _load_existing(OUT_JUDGMENTS)
    OUT_JUDGMENTS.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if OUT_JUDGMENTS.exists() and existing else "w"
    with OUT_JUDGMENTS.open(mode, encoding="utf-8") as f:
        total = len(ids)
        for i, idx in enumerate(ids, start=1):
            if idx in existing:
                continue
            prompt = data_a[idx]["prompt"]
            a_text = data_a[idx]["response"]
            b_text = data_b[idx]["response"]
            order = "A/B"
            if SHUFFLE and random.random() < 0.5:
                a_text, b_text = b_text, a_text
                order = "B/A"

            judge_prompt = run_judge_gemini.build_judge_prompt(prompt, a_text, b_text)
            result_text = ""
            for attempt in range(MAX_RETRIES):
                try:
                    result_text = run_judge_gemini.call_gemini(
                        judge_prompt, url, temperature=0.0, max_tokens=JUDGE_MAX_TOKENS
                    )
                    if not result_text.strip():
                        raise RuntimeError("Empty judge response")
                    break
                except Exception:
                    if attempt == MAX_RETRIES - 1:
                        raise
                    time.sleep(SLEEP_SEC)

            choice = run_judge_gemini.normalize_choice(result_text)
            if SHUFFLE and order == "B/A":
                if choice == "A":
                    choice = "B"
                elif choice == "B":
                    choice = "A"

            record = {
                "id": idx,
                "prompt": prompt,
                "winner": choice,
                "order": order,
                "response_a": data_a[idx]["response"],
                "response_b": data_b[idx]["response"],
                "judge_raw": result_text,
            }
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

            if i % 5 == 0 or i == total:
                print(f"Judged {i}/{total}")

    counts = {"A": 0, "B": 0, "tie": 0}
    with OUT_JUDGMENTS.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            winner = obj.get("winner", "tie")
            if winner not in counts:
                winner = "tie"
            counts[winner] += 1
    total_done = sum(counts.values())
    win_rate = counts["A"] / total_done if total_done else 0.0
    print(
        f"A wins: {counts['A']} | B wins: {counts['B']} | ties: {counts['tie']} | "
        f"A win-rate: {win_rate:.3f}"
    )


def main() -> None:
    _set_offline_env()
    random.seed(SEED)

    expected_total = len(run_generate.load_prompts(PROMPTS_PATH, limit=None))

    def count_lines(path: pathlib.Path) -> int:
        if not path.exists():
            return 0
        with path.open("r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())

    if _needs_regen(OUT_POINTWISE, expected_total):
        print("Generating Pointwise responses...")
        _generate_responses(ADAPTER_POINTWISE, OUT_POINTWISE)
    else:
        print(f"Using cached {OUT_POINTWISE}")

    if _needs_regen(OUT_SET_AWARE, expected_total):
        print("Generating Set-Aware responses...")
        _generate_responses(ADAPTER_SET_AWARE, OUT_SET_AWARE)
    else:
        print(f"Using cached {OUT_SET_AWARE}")

    print("Judging with Gemini...")
    _judge_pairwise()


if __name__ == "__main__":
    main()
