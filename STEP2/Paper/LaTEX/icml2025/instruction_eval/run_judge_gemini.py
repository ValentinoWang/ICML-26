import argparse
import json
import os
import random
import time
from typing import Dict

import requests


def load_responses(path: str) -> Dict[int, dict]:
    items: Dict[int, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            items[int(obj["id"])] = obj
    return items


def build_gemini_url(endpoint: str, model: str) -> str:
    base = endpoint.rstrip("/")
    if "{model}" in base:
        url = base.format(model=model)
    else:
        if "/v1beta/models" not in base:
            base = base + "/v1beta/models"
        url = f"{base}/{model}:generateContent"
    return url


def build_openai_url(endpoint: str) -> str:
    base = endpoint.rstrip("/")
    if base.endswith("/responses"):
        return base
    if base.endswith("/v1"):
        return f"{base}/responses"
    return f"{base}/v1/responses"


def call_gemini(
    prompt: str,
    url: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    key_in_query: bool,
) -> str:
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "responseMimeType": "text/plain",
            "thinkingConfig": {"includeAnswer": True},
        },
    }
    headers = {"Content-Type": "application/json"}
    final_url = url
    if key_in_query:
        if "key=" not in final_url:
            sep = "&" if "?" in final_url else "?"
            final_url = f"{final_url}{sep}key={api_key}"
    else:
        headers["x-goog-api-key"] = api_key
    resp = requests.post(final_url, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        pass
    try:
        choice = data["choices"][0]
        if "message" in choice and "content" in choice["message"]:
            return choice["message"]["content"].strip()
        if "text" in choice:
            return choice["text"].strip()
    except Exception:
        pass
    if "output" in data and isinstance(data["output"], str):
        return data["output"].strip()
    return ""


def extract_openai_text(data: dict) -> str:
    if isinstance(data.get("output_text"), str) and data["output_text"].strip():
        return data["output_text"].strip()
    output = data.get("output")
    if isinstance(output, list):
        for item in output:
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if part.get("type") == "output_text" and part.get("text"):
                    return str(part["text"]).strip()
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        message = choice.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("text"):
                        return str(part["text"]).strip()
        if isinstance(choice.get("text"), str) and choice["text"].strip():
            return choice["text"].strip()
    nested = data.get("data")
    if isinstance(nested, dict):
        return extract_openai_text(nested)
    return ""


def call_openai(
    prompt: str,
    url: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "input": prompt,
        "temperature": temperature,
        "max_output_tokens": max_tokens,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(str(data["error"]))
    text = extract_openai_text(data)
    if not text:
        raise RuntimeError("Empty OpenAI response")
    return text


def normalize_choice(text: str) -> str:
    if not text:
        return "tie"
    import re

    t = text.strip().lower()
    match = re.search(r"\b(a|b|tie)\b", t)
    if not match:
        return "tie"
    token = match.group(1)
    if token == "a":
        return "A"
    if token == "b":
        return "B"
    return "tie"


def build_judge_prompt(prompt: str, a_text: str, b_text: str) -> str:
    return (
        "You are a strict evaluator. Compare Assistant A and Assistant B for the user instruction. "
        "Choose the better response based on correctness, helpfulness, and instruction-following. "
        "If they are equally good, answer tie. Respond with only one token: A, B, or tie.\n\n"
        f"User instruction:\n{prompt}\n\n"
        f"Assistant A:\n{a_text}\n\n"
        f"Assistant B:\n{b_text}\n"
    )


def _load_existing(path: str) -> tuple[Dict[int, str], list[dict], bool]:
    if not os.path.exists(path):
        return {}, [], False
    data: Dict[int, str] = {}
    records: Dict[int, dict] = {}
    had_invalid = False
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            try:
                idx = int(obj["id"])
            except Exception:
                continue
            raw = str(obj.get("judge_raw", "")).strip()
            if not raw:
                had_invalid = True
                continue
            winner = obj.get("winner", "tie")
            if winner not in ("A", "B", "tie"):
                winner = "tie"
            data[idx] = winner
            records[idx] = obj
    return data, list(records.values()), had_invalid


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge A/B responses with Gemini or OpenAI API.")
    parser.add_argument("--responses-a", required=True)
    parser.add_argument("--responses-b", required=True)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--provider", default=os.environ.get("JUDGE_PROVIDER", "gemini"))
    parser.add_argument("--model")
    parser.add_argument("--endpoint")
    parser.add_argument("--api-key")
    parser.add_argument(
        "--key-in-query",
        action="store_true",
        default=os.environ.get("GEMINI_KEY_IN_QUERY", "0") == "1",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=1088)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--sleep", type=float, default=2.0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    provider = args.provider.lower()
    if provider not in ("gemini", "openai"):
        raise SystemExit("Unsupported provider. Use gemini or openai.")

    model = args.model
    if not model:
        if provider == "openai":
            model = os.environ.get("OPENAI_MODEL", "gpt-5")
        else:
            model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

    endpoint = args.endpoint
    if not endpoint:
        if provider == "openai":
            endpoint = os.environ.get("OPENAI_BASE_URL") or os.environ.get(
                "OPENAI_API_BASE", "https://api.openai.com"
            )
        else:
            endpoint = os.environ.get(
                "GEMINI_ENDPOINT", "https://generativelanguage.googleapis.com/v1beta/models"
            )

    api_key = args.api_key
    if not api_key:
        if provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("AI_API_KEY")
        else:
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("AI_API_KEY")
    if not api_key:
        if provider == "openai":
            raise SystemExit("Missing API key. Set OPENAI_API_KEY or pass --api-key.")
        raise SystemExit("Missing API key. Set GEMINI_API_KEY or pass --api-key.")

    if provider == "openai" and args.max_tokens < 16:
        args.max_tokens = 16

    random.seed(args.seed)
    data_a = load_responses(args.responses_a)
    data_b = load_responses(args.responses_b)

    ids = sorted(set(data_a.keys()) & set(data_b.keys()))
    if not ids:
        raise SystemExit("No overlapping ids between response files.")
    if args.limit:
        ids = ids[: args.limit]

    if provider == "openai":
        url = build_openai_url(endpoint)
    else:
        url = build_gemini_url(endpoint, model)

    existing, existing_records, had_invalid = _load_existing(args.out_jsonl)
    if had_invalid:
        with open(args.out_jsonl, "w", encoding="utf-8") as f:
            for record in sorted(existing_records, key=lambda r: r.get("id", 0)):
                f.write(json.dumps(record, ensure_ascii=True) + "\n")
    counts = {"A": 0, "B": 0, "tie": 0}
    for winner in existing.values():
        counts[winner] += 1

    mode = "a" if existing else "w"
    with open(args.out_jsonl, mode, encoding="utf-8") as f:
        total = len(ids)
        for i, idx in enumerate(ids, start=1):
            if idx in existing:
                continue
            prompt = data_a[idx]["prompt"]
            a_text = data_a[idx]["response"]
            b_text = data_b[idx]["response"]
            order = "A/B"
            if args.shuffle and random.random() < 0.5:
                a_text, b_text = b_text, a_text
                order = "B/A"

            judge_prompt = build_judge_prompt(prompt, a_text, b_text)

            result_text = ""
            for attempt in range(args.max_retries):
                try:
                    if provider == "openai":
                        result_text = call_openai(
                            judge_prompt, url, api_key, model, args.temperature, args.max_tokens
                        )
                    else:
                        result_text = call_gemini(
                            judge_prompt,
                            url,
                            api_key,
                            args.temperature,
                            args.max_tokens,
                            args.key_in_query,
                        )
                    break
                except Exception:
                    if attempt == args.max_retries - 1:
                        raise
                    time.sleep(args.sleep)

            choice = normalize_choice(result_text)
            if args.shuffle and order == "B/A":
                if choice == "A":
                    choice = "B"
                elif choice == "B":
                    choice = "A"

            counts[choice] += 1
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

    total = sum(counts.values())
    win_rate = counts["A"] / total if total else 0.0
    print(f"A wins: {counts['A']} | B wins: {counts['B']} | ties: {counts['tie']} | A win-rate: {win_rate:.3f}")


if __name__ == "__main__":
    main()
