"""Exact dense retrieval with an explicit external encoder."""

from __future__ import annotations

from typing import Any

from warp.models import Document, SearchResult
from warp.utils import cosine


class DenseRetriever:
    """Exact vector retriever; documents and queries share one encoder."""

    def __init__(self, encoder: Any, model_name: str) -> None:
        if encoder is None:
            raise TypeError("DenseRetriever requires an explicit production encoder")
        self.model_name = model_name
        self.encoder = encoder
        self.documents: list[Document] = []
        self.vectors: list[list[float]] = []
        self._index: dict[str, int] = {}

    def fit(self, documents: list[Document]) -> "DenseRetriever":
        """Encode the corpus once and check one vector per document."""
        self.documents = list(documents)
        self.vectors = self.encoder.encode_documents([doc.content for doc in documents])
        if len(self.vectors) != len(self.documents):
            raise RuntimeError(
                f"Dense encoder returned {len(self.vectors)} vectors for {len(self.documents)} documents"
            )
        self._index = {doc.id: index for index, doc in enumerate(documents)}
        return self

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode queries with the official query instruction."""
        values = self.encoder.encode_queries(texts)
        if len(values) != len(texts):
            raise RuntimeError(f"Dense encoder returned {len(values)} vectors for {len(texts)} queries")
        return values

    def vector(self, doc_id: str) -> list[float]:
        """Cached document vector for semantic edges and dispersion."""
        return self.vectors[self._index[doc_id]]

    def search(self, query: str, k: int = 10, doc_ids: set[str] | None = None) -> list[SearchResult]:
        """Exact cosine retrieval."""
        if k <= 0:
            return []
        query_vector = self.encode([query])[0]
        allowed = None if doc_ids is None else set(doc_ids)
        scores = [
            (index, cosine(query_vector, vector))
            for index, vector in enumerate(self.vectors)
            if allowed is None or self.documents[index].id in allowed
        ]
        scores.sort(key=lambda item: (-item[1], self.documents[item[0]].id))
        return [
            SearchResult(self.documents[index].id, score, "dense", rank + 1)
            for rank, (index, score) in enumerate(scores[:k])
        ]
