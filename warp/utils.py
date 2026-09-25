"""Dependency-free helpers for text, vectors, JSON, and batching."""

from __future__ import annotations

import hashlib
import json
import math
import re
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Deterministic tokenizer shared by BM25 and pre-build cost estimates."""
    return TOKEN_RE.findall(text.lower())


def stable_hash(value: str, modulo: int) -> int:
    """Stable hash; avoids Python process-level hash randomization."""
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % modulo


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity; zero vectors return 0."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def minmax(values: list[float]) -> list[float]:
    """Scale values to [0, 1]; a constant-zero column stays zero."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi == lo:
        return [1.0 if hi else 0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def read_json_records(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL, a JSON list, or a data/documents/queries wrapper."""
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in handle if line.strip()]
        value = json.load(handle)
    if isinstance(value, list):
        return value
    for key in ("data", "documents", "queries", "items"):
        if isinstance(value.get(key), list):
            return value[key]
    raise ValueError(f"Cannot find a record list in {path}")


def write_json(path: str | Path, value: Any) -> None:
    """Create parents and write UTF-8 indented JSON atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def batches(values: list[Any], size: int) -> Iterable[list[Any]]:
    """Yield contiguous batches; the last batch may be shorter."""
    for start in range(0, len(values), size):
        yield values[start:start + size]
