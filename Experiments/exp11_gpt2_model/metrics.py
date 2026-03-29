from typing import Sequence


def distinct_n(texts: Sequence[str], n: int) -> float:
    total = 0
    unique = 0
    for t in texts:
        tokens = t.strip().split()
        if len(tokens) < n:
            continue
        total += len(tokens) - n + 1
        grams = set()
        for i in range(len(tokens) - n + 1):
            grams.add(tuple(tokens[i : i + n]))
        unique += len(grams)
    if total == 0:
        return 0.0
    return unique / total
