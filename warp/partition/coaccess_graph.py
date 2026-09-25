"""Cheap document co-access graph from the train/design workload.

Used only for corpus partition, not as a knowledge graph. Weights mix query
top-k co-occurrence with a small semantic kNN term. Dev/test queries must not
be passed to this builder.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations

from warp.models import Document, Query
from warp.retrieval.dense import DenseRetriever
from warp.retrieval.hybrid import HybridRetriever
from warp.retrieval.ann import semantic_knn


@dataclass
class CoaccessGraph:
    """Total, query, and semantic edges plus per-query retrieval traces."""
    nodes: list[str]
    edges: dict[tuple[str, str], float]
    query_edges: dict[tuple[str, str], float]
    semantic_edges: dict[tuple[str, str], float]
    query_results: dict[str, list[str]]

    def neighbors(self) -> dict[str, dict[str, float]]:
        """Expand an undirected edge map into an adjacency list."""
        output: dict[str, dict[str, float]] = {node: {} for node in self.nodes}
        for (left, right), weight in self.edges.items():
            output[left][right] = weight
            output[right][left] = weight
        return output


class CoaccessGraphBuilder:
    """w_ij = w_query + lambda * w_semantic."""
    def __init__(self, top_k: int = 20, semantic_k: int = 3, semantic_lambda: float = 0.05,
                 min_semantic_similarity: float = 0.0) -> None:
        self.top_k = top_k
        self.semantic_k = semantic_k
        self.semantic_lambda = semantic_lambda
        self.min_semantic_similarity = min_semantic_similarity

    def build(self, documents: list[Document], queries: list[Query], retriever: HybridRetriever) -> CoaccessGraph:
        """Add pairwise edges from each design query top-k, then weak semantic edges."""
        # Query edges are the main community-detection signal.
        query_edges: dict[tuple[str, str], float] = defaultdict(float)
        query_results: dict[str, list[str]] = {}
        for query in queries:
            ids = [result.doc_id for result in retriever.search(query.text, self.top_k)]
            query_results[query.id] = ids
            for left, right in combinations(sorted(set(ids)), 2):
                query_edges[(left, right)] += 1.0

        # Semantic edges keep documents unseen by train queries from being isolated.
        semantic_edges: dict[tuple[str, str], float] = defaultdict(float)
        if self.semantic_k > 0 and self.semantic_lambda > 0:
            dense: DenseRetriever = retriever.dense
            ids = [doc.id for doc in documents]
            neighbors = semantic_knn(ids, [dense.vector(doc_id) for doc_id in ids], self.semantic_k)
            for doc_id, values in neighbors.items():
                for other_id, similarity in values:
                    if similarity <= self.min_semantic_similarity:
                        continue
                    edge = tuple(sorted((doc_id, other_id)))
                    semantic_edges[edge] = max(semantic_edges[edge], similarity)

        edges: dict[tuple[str, str], float] = dict(query_edges)
        for edge, similarity in semantic_edges.items():
            edges[edge] = edges.get(edge, 0.0) + self.semantic_lambda * similarity
        return CoaccessGraph([doc.id for doc in documents], edges, dict(query_edges),
                             dict(semantic_edges), query_results)
