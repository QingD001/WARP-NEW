"""IRCoT 风格多步检索：每步检索 → 推理扩展 → 再检索，并落盘全量 step log。

这与 `warp.retrieval.multistep.retrieve_steps` 的 passage-feedback 不是同一条协议。
检索决策不读取 gold；gold 只用于事后逐步指标。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from warp.audit import emit
from warp.eval.cutoffs import RETRIEVAL_KS
from warp.eval.retrieval import query_metrics, serialize_ranked
from warp.models import Document, Query, SearchResult
from warp.utils import write_json


IRCOT_PROMPT = """You are performing multi-hop retrieval.
Question: {question}
Documents retrieved so far:
{documents}

If the documents are sufficient to answer the question, reply with exactly END.
Otherwise reply with one short search query that would retrieve the missing evidence.
Do not answer the question. Reply with END or a search query only.
"""


def merge_ranked(existing: list[SearchResult], incoming: list[SearchResult]) -> list[SearchResult]:
    """按 doc_id 去重，保留更高分，重写 rank。"""
    best: dict[str, SearchResult] = {}
    for result in existing + incoming:
        previous = best.get(result.doc_id)
        if previous is None or result.score > previous.score:
            best[result.doc_id] = result
    ordered = sorted(best.values(), key=lambda item: (-item.score, item.doc_id))
    return [
        SearchResult(item.doc_id, item.score, item.source, rank + 1, item.region_id)
        for rank, item in enumerate(ordered)
    ]


def _format_documents(
    results: list[SearchResult],
    documents: Mapping[str, Document] | None,
    *,
    snippet_chars: int,
) -> str:
    lines: list[str] = []
    for item in results:
        snippet = ""
        if documents is not None and item.doc_id in documents:
            snippet = documents[item.doc_id].content[:snippet_chars].replace("\n", " ").strip()
        if snippet:
            lines.append(f"- {item.doc_id} ({item.source}): {snippet}")
        else:
            lines.append(f"- {item.doc_id} ({item.source})")
    return "\n".join(lines) if lines else "(none)"


def _generation_usage(generate: Callable[[str], str]) -> dict[str, Any]:
    usage = getattr(generate, "last_usage", None)
    return dict(usage) if isinstance(usage, dict) else {}


def run_multistep_retrieval(
    queries: list[Query],
    search: Callable[[str, int], tuple[list[SearchResult], dict[str, Any]]],
    generate: Callable[[str], str],
    *,
    max_steps: int,
    retrieval_k: int,
    ks: tuple[int, ...] = RETRIEVAL_KS,
    log_path: Path | None = None,
    method: str = "unknown",
    documents: Mapping[str, Document] | None = None,
    snippet_chars: int = 400,
) -> dict[str, Any]:
    """对每条 query 跑固定步数上限的 IRCoT 检索，并可选写入 JSONL 轨迹。"""
    if max_steps < 1:
        raise ValueError("multistep max_steps must be >= 1")
    if retrieval_k <= 0 or snippet_chars <= 0:
        raise ValueError("retrieval_k and snippet_chars must be positive")
    eligible = [query for query in queries if query.gold_doc_ids]
    if not eligible:
        raise ValueError("Multistep retrieval requires gold evidence")
    handle = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("w", encoding="utf-8")
    per_query: dict[str, dict[str, float]] = {}
    ranked: dict[str, list[dict[str, Any]]] = {}
    cutoff = max(ks)
    try:
        for query in eligible:
            accumulated: list[SearchResult] = []
            current_query = query.text
            step_rows: list[dict[str, Any]] = []
            for step in range(1, max_steps + 1):
                results, trace = search(current_query, retrieval_k)
                accumulated = merge_ranked(accumulated, results)
                gold = set(query.gold_doc_ids)
                step_found = {item.doc_id for item in results[:cutoff]}
                accumulated_found = {item.doc_id for item in accumulated[:cutoff]}
                generation = ""
                next_query = current_query
                stop_reason = "continue"
                if step == max_steps:
                    stop_reason = "max_steps"
                else:
                    generation = generate(IRCOT_PROMPT.format(
                        question=query.text,
                        documents=_format_documents(
                            results[:retrieval_k], documents, snippet_chars=snippet_chars,
                        ),
                    )).strip()
                    if not generation:
                        stop_reason = "empty_generation"
                    elif generation.upper().startswith("END"):
                        stop_reason = "model_end"
                    else:
                        next_query = generation.splitlines()[0].strip() or current_query
                        if not next_query:
                            stop_reason = "empty_generation"
                row = {
                    "query_id": query.id,
                    "method": method,
                    "event": "step",
                    "step": step,
                    "stop_reason": stop_reason,
                    "original_query": query.text,
                    "step_query": current_query,
                    "next_query": next_query,
                    "generation": generation,
                    "base_hit_regions": trace.get("base_hit_regions", []),
                    "router_regions": trace.get("router_regions", []),
                    "bridge_regions": trace.get("bridge_regions", []),
                    "routed_regions": trace.get("routed_regions", []),
                    "base_candidate_doc_ids": trace.get("base_candidate_doc_ids", []),
                    "graph_candidate_doc_ids": trace.get("graph_candidate_doc_ids", {}),
                    "step_results": serialize_ranked(results),
                    "accumulated_results": serialize_ranked(accumulated[:cutoff]),
                    "step_gold_hit": sorted(gold & step_found),
                    "gold_hit": sorted(gold & accumulated_found),
                    "step_metrics": query_metrics(results, query, ks),
                    "metrics": query_metrics(accumulated, query, ks),
                    "tokens": trace.get("tokens", {}),
                    "generation_usage": _generation_usage(generate) if generation else {},
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                step_rows.append(row)
                emit("ircot_step", row)
                if handle is not None:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                if stop_reason != "continue":
                    break
                current_query = next_query
            final_metrics = query_metrics(accumulated, query, ks)
            per_query[query.id] = final_metrics
            ranked[query.id] = serialize_ranked(accumulated[:cutoff])
            complete_row = {
                "query_id": query.id,
                "method": method,
                "event": "query_complete",
                "steps": len(step_rows),
                "stop_reason": step_rows[-1]["stop_reason"],
                "metrics": final_metrics,
                "ranked_results": ranked[query.id],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            emit("ircot_query_complete", complete_row)
            if handle is not None:
                handle.write(json.dumps(complete_row, ensure_ascii=False) + "\n")
    finally:
        if handle is not None:
            handle.close()
    summary: dict[str, Any] = {
        "num_queries": len(eligible),
        "max_steps": max_steps,
        "retrieval_ks": list(ks),
        "log_path": str(log_path) if log_path is not None else None,
        "per_query": per_query,
        "ranked_results": ranked,
        "protocol": "ircot",
    }
    for metric in next(iter(per_query.values())):
        values = [per_query[query.id][metric] for query in eligible]
        summary[metric] = sum(values) / len(values)
    if log_path is not None:
        write_json(log_path.with_suffix(".summary.json"), {
            "method": method,
            "num_queries": len(eligible),
            "max_steps": max_steps,
            "protocol": "ircot",
            **{key: summary[key] for key in summary if key not in {"per_query", "ranked_results"}},
        })
    return summary
