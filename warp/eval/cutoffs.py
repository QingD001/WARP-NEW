"""Shared retrieval cutoffs for the main table and IRCoT eval."""

from __future__ import annotations

RETRIEVAL_KS: tuple[int, ...] = (2, 3, 5, 10)
READER_TOP_K: int = 5


def metric_names(ks: tuple[int, ...] = RETRIEVAL_KS) -> tuple[str, ...]:
    names: list[str] = []
    for k in ks:
        names.append(f"evidence_recall@{k}")
        names.append(f"complete_evidence@{k}")
    return tuple(names)
