"""Minimal protocol shared by all base retrievers."""

from __future__ import annotations

from typing import Protocol

from warp.models import Document, SearchResult


class Retriever(Protocol):
    """Full-corpus search or a `doc_ids` subset."""
    def fit(self, documents: list[Document]) -> "Retriever":
        """Build an index on the shared corpus."""
        ...

    def search(self, query: str, k: int = 10, doc_ids: set[str] | None = None) -> list[SearchResult]:
        """Search the full corpus or a doc_ids subset."""
        ...
