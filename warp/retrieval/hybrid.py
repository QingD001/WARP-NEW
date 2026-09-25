"""Reciprocal Rank Fusion shared by BM25, dense, and graph lists."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any
from warp.audit import emit

from warp.models import Document, SearchResult
from .bm25 import BM25Retriever
from .dense import DenseRetriever
from .reranker import Reranker


def reciprocal_rank_fusion(
    result_lists: Sequence[Sequence[SearchResult]],
    k: int = 10,
    rrf_constant: int = 60,
    weights: Sequence[float] | None = None,
    source: str = "rrf",
) -> list[SearchResult]:
    """Fuse by rank, not raw scores; break ties by doc_id."""
    scores: dict[str, float] = defaultdict(float)
    regions: dict[str, str | None] = {}
    if weights is None:
        weights = [1.0] * len(result_lists)
    if len(weights) != len(result_lists):
        raise ValueError("RRF requires exactly one weight per result list")
    for weight, results in zip(weights, result_lists):
        for rank, result in enumerate(results, 1):
            scores[result.doc_id] += weight / (rrf_constant + rank)
            regions.setdefault(result.doc_id, result.region_id)
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]
    return [SearchResult(doc_id, score, source, rank + 1, regions[doc_id])
            for rank, (doc_id, score) in enumerate(ranked)]


class HybridRetriever:
    """Shared BM25 + dense base index used by every method."""
    def __init__(self, bm25: BM25Retriever, dense: DenseRetriever, rrf_constant: int = 60) -> None:
        self.bm25 = bm25
        self.dense = dense
        self.rrf_constant = rrf_constant

    def fit(self, documents: list[Document]) -> "HybridRetriever":
        """Fit lexical and dense indexes on the same corpus."""
        self.bm25.fit(documents)
        self.dense.fit(documents)
        return self

    def search(self, query: str, k: int = 10, doc_ids: set[str] | None = None) -> list[SearchResult]:
        """RRF over deeper per-channel pools to reduce single-list truncation."""
        depth = max(k * 2, k)
        return reciprocal_rank_fusion(
            [self.bm25.search(query, depth, doc_ids), self.dense.search(query, depth, doc_ids)],
            k=k, rrf_constant=self.rrf_constant, source="hybrid",
        )


def fuse_and_rerank(
    query: str, result_lists: Sequence[Sequence[SearchResult]], reranker: Reranker, *,
    k: int, candidate_k: int, source: str, weights: Sequence[float] | None = None,
    trace: dict[str, Any] | None = None,
) -> list[SearchResult]:
    """Single final ranking path for Base, probe, selective, and Full Graph."""
    if candidate_k < k:
        raise ValueError("candidate_k must be at least k")
    fused = reciprocal_rank_fusion(result_lists, candidate_k, weights=weights, source=source)
    if trace is not None:
        trace["fused_candidate_doc_ids"] = [row.doc_id for row in fused]
    ranked = reranker.rerank(query, fused, k)
    emit("fusion_rerank", {"query": query, "source": source, "inputs": result_lists,
                           "weights": weights, "fused": fused, "ranked": ranked,
                           "candidate_k": candidate_k, "k": k})
    return ranked
