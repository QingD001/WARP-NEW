"""Open-domain QA answer normalization, Exact Match, and token F1."""

from __future__ import annotations

import re
import string
from collections import Counter


def _normalize(text: str) -> str:
    text = text.lower()
    text = "".join(char for char in text if char not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def answer_em(prediction: str, gold: str) -> float:
    """Exact match after lowercasing and stripping punctuation/articles."""
    return float(_normalize(prediction) == _normalize(gold))


def answer_f1(prediction: str, gold: str) -> float:
    """F1 from normalized token-multiset overlap."""
    predicted = _normalize(prediction).split()
    expected = _normalize(gold).split()
    common = sum((Counter(predicted) & Counter(expected)).values())
    if not predicted or not expected:
        return float(predicted == expected)
    if common == 0:
        return 0.0
    precision, recall = common / len(predicted), common / len(expected)
    return 2 * precision * recall / (precision + recall)
