"""Regional-graph retrieval protocol."""

from typing import Protocol

from warp.models import SearchResult
from .builder import RegionalGraph


class GraphRetriever(Protocol):
    """Graph-retrieval protocol used by the advisor and pipeline."""

    def search(self, query: str, graph: RegionalGraph, k: int = 10) -> list[SearchResult]:
        """Retrieve passages from one materialized regional graph."""
        ...

    def stats(self) -> dict[str, float | int]:
        ...

    def delta(self, before: dict[str, float | int]) -> dict[str, float | int]:
        ...
