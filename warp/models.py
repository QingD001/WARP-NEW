"""Shared WARP-G dataclasses.

Loaders emit Document/Query, retrievers exchange SearchResult, and physical
design talks via Region / RegionFeatures / ConstructionCost. Keep these
objects simple so the graph backend can change without touching the advisor
or evaluator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Document:
    """A retrievable passage; `id` is the stable eval and backend key."""
    id: str
    text: str
    title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def content(self) -> str:
        """Title plus text, the payload sent to lexical, dense, and graph backends."""
        return f"{self.title}\n{self.text}".strip()


@dataclass(slots=True)
class Query:
    """A design or eval question; gold_doc_ids lists full evidence IDs."""
    id: str
    text: str
    gold_doc_ids: list[str] = field(default_factory=list)
    answer: str | list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SearchResult:
    """One ranked hit; source/region_id record which physical structure produced it."""
    doc_id: str
    score: float
    source: str
    rank: int = 0
    region_id: str | None = None


@dataclass(slots=True)
class Region:
    """A passage-ID set from co-access community detection."""
    id: str
    doc_ids: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DatasetBundle:
    """Shared corpus plus strictly separated train/design, dev, and test queries."""
    documents: list[Document]
    train: list[Query]
    dev: list[Query] = field(default_factory=list)
    test: list[Query] = field(default_factory=list)


@dataclass(slots=True)
class ConstructionCost:
    """Measured multi-dimensional build cost; wall time is not converted to money."""
    input_tokens: int = 0
    output_tokens: int = 0
    embedding_tokens: int = 0
    wall_seconds: float = 0.0
    nodes: int = 0
    edges: int = 0
    storage_bytes: int = 0
    estimated_usd: float = 0.0

    @property
    def selection_cost(self) -> float:
        """Stable pre-construction cost proxy used by the budget selector."""
        return float(self.input_tokens + self.output_tokens + self.embedding_tokens)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable dict."""
        return asdict(self)


@dataclass(slots=True)
class RegionFeatures:
    """Region-local evidence features plus query-level context."""
    region_id: str
    num_docs: float
    num_tokens: float
    query_freq: float
    base_recall: float
    failure_rate: float
    avg_retrieval_entropy: float
    multi_doc_rate: float
    embedding_dispersion: float
    coaccess_density: float
    global_base_recall: float = 0.0
    global_failure_rate: float = 0.0
    global_multi_doc_rate: float = 0.0
    gold_query_rate: float = 0.0
    cross_region_gold_rate: float = 0.0

    @classmethod
    def names(cls) -> list[str]:
        """Fixed feature-column order."""
        return [
            "num_docs", "num_tokens", "query_freq", "base_recall",
            "failure_rate", "avg_retrieval_entropy", "multi_doc_rate",
            "embedding_dispersion", "coaccess_density",
            "global_base_recall", "global_failure_rate", "global_multi_doc_rate",
            "gold_query_rate", "cross_region_gold_rate",
        ]

    def vector(self) -> list[float]:
        """Feature vector in `names()` order."""
        return [float(getattr(self, name)) for name in self.names()]

    def to_dict(self) -> dict[str, Any]:
        """Auditable feature dict including region_id."""
        return asdict(self)
