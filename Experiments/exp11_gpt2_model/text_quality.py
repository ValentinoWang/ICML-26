from __future__ import annotations

import gzip
from dataclasses import dataclass
from typing import Iterable, Sequence


def _words(text: str) -> list[str]:
    return [t for t in text.strip().split() if t]


def unique_line_ratio(texts: Sequence[str]) -> float:
    if not texts:
        return 0.0
    norm = [t.strip() for t in texts if t.strip()]
    if not norm:
        return 0.0
    return len(set(norm)) / float(len(norm))


def avg_words(texts: Sequence[str]) -> float:
    if not texts:
        return 0.0
    lens = [len(_words(t)) for t in texts if t.strip()]
    if not lens:
        return 0.0
    return sum(lens) / float(len(lens))


def mean_intra_repetition(texts: Sequence[str], n: int = 4) -> float:
    """
    Average per-sample n-gram repetition ratio:
        rep = 1 - (#unique ngrams / #total ngrams)  in each sample, then averaged.
    """
    reps: list[float] = []
    for t in texts:
        toks = _words(t)
        if len(toks) < n:
            continue
        total = len(toks) - n + 1
        grams = set(tuple(toks[i : i + n]) for i in range(total))
        unique = len(grams)
        reps.append(1.0 - (unique / float(total)) if total > 0 else 0.0)
    if not reps:
        return 0.0
    return sum(reps) / float(len(reps))


def ascii_ratio(texts: Sequence[str]) -> float:
    """
    Fraction of characters that are ASCII (a crude 'gibberish' detector).
    """
    total = 0
    ascii_cnt = 0
    for t in texts:
        for ch in t:
            total += 1
            if ord(ch) < 128:
                ascii_cnt += 1
    if total == 0:
        return 0.0
    return ascii_cnt / float(total)


def gzip_compression_ratio(texts: Sequence[str]) -> float:
    """
    Compression ratio on the concatenated text. Lower => more redundant/repetitive.
    """
    joined = "\n".join(t.strip() for t in texts if t.strip())
    if not joined:
        return 0.0
    raw = joined.encode("utf-8", errors="ignore")
    comp = gzip.compress(raw)
    return len(comp) / float(len(raw))


@dataclass(frozen=True)
class TextQualityMetrics:
    n_lines: int
    unique_line_ratio: float
    avg_words: float
    rep4_intra: float
    ascii_ratio: float
    gzip_ratio: float


def compute_text_quality(texts: Sequence[str]) -> TextQualityMetrics:
    return TextQualityMetrics(
        n_lines=len([t for t in texts if t.strip()]),
        unique_line_ratio=unique_line_ratio(texts),
        avg_words=avg_words(texts),
        rep4_intra=mean_intra_repetition(texts, n=4),
        ascii_ratio=ascii_ratio(texts),
        gzip_ratio=gzip_compression_ratio(texts),
    )


def read_lines(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [line.rstrip("\n") for line in f]


def iter_nonempty(lines: Iterable[str]) -> list[str]:
    return [l for l in (ln.strip() for ln in lines) if l]

