"""Regional-graph handle and graph-build protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from warp.models import ConstructionCost, Document, Region


@dataclass
class RegionalGraph:
    """Regional graph handle; `backend` holds the official HippoRAG2 instance."""

    region_id: str
    doc_ids: list[str]
    cost: ConstructionCost
    metadata: dict[str, object] = field(default_factory=dict)
    backend: Any = field(default=None, repr=False)


class GraphBuilder(Protocol):
    """Graph-build protocol used by the advisor."""

    def estimate_cost(self, region: Region, documents: list[Document]) -> float:
        """Cost estimate used by budget selection before a real build."""
        ...

    def build(self, region: Region, documents: list[Document]) -> RegionalGraph:
        """Build one regional graph and record actual cost."""
        ...

    def build_full_graph(self, region: Region, documents: list[Document]) -> RegionalGraph:
        """Build the isolated corpus-wide Full Graph baseline."""
        ...
