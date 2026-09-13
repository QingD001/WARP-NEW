"""在任意检索结果上运行固定的官方 HippoRAG2 QA reader。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from warp.models import Document, Query, SearchResult
from .qa import answer_em, answer_f1
from warp.audit import emit, audit_context


def evaluate_hipporag2_reader(
    queries: list[Query],
    search: Callable[[str, int], list[SearchResult]],
    documents: list[Document],
    hipporag_graph: Any,
    top_k: int = 5,
    *, retrieved_doc_ids: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Use HippoRAG 2's frozen QA prompt and QA LLM on arbitrary retrieval output."""
    if top_k <= 0:
        raise ValueError("Reader top_k must be positive")
    try:
        from hipporag.utils.misc_utils import QuerySolution
    except ImportError as exc:
        raise ImportError("Install the project with `pip install -e .` to run HippoRAG reader evaluation") from exc
    rag = hipporag_graph.backend
    if rag is None:
        raise TypeError("Reader evaluation requires an official HippoRAG2 full graph")
    # Reader 始终使用同一个 full-graph HippoRAG 实例中的 prompt manager/QA LLM，
    # 但 docs 由待比较的检索方法提供，因此只改变 evidence，不改变生成器。
    doc_map = {doc.id: doc for doc in documents}
    eligible = [query for query in queries if query.answer is not None]
    if not eligible:
        raise ValueError("Reader evaluation requires gold answers")
    if len({query.id for query in eligible}) != len(eligible):
        raise ValueError("Reader query IDs must be unique")
    if any(isinstance(query.answer, list) and not query.answer for query in eligible):
        raise ValueError("Reader gold answer lists must not be empty")
    solutions = []
    for query in eligible:
        results = (search(query.text, top_k) if retrieved_doc_ids is None else
                   [SearchResult(doc_id, 0.0, "saved_retrieval") for doc_id in retrieved_doc_ids[query.id][:top_k]])
        solutions.append(QuerySolution(
            question=query.text,
            docs=[doc_map[result.doc_id].content for result in results[:top_k]],
            doc_scores=None,
            doc_metadata=[{"warp_doc_id": result.doc_id} for result in results[:top_k]],
        ))
    tracker = rag._warp_usage_tracker
    before = tracker.get("reader")
    tracker.phase = "reader"
    tracker.audit_context = audit_context()
    previous_top_k = rag.global_config.qa_top_k
    rag.global_config.qa_top_k = top_k
    try:
        answered, raw_responses, raw_metadata = rag.qa(solutions)
    finally:
        tracker.phase = "idle"
        rag.global_config.qa_top_k = previous_top_k
    if len(answered) != len(eligible):
        raise RuntimeError("Reader returned a different number of answers than queries")
    emit("reader_raw", {"queries": eligible, "input_documents": [solution.docs for solution in solutions],
                        "responses": raw_responses, "metadata": raw_metadata})
    em_total = f1_total = 0.0
    predictions: list[dict[str, Any]] = []
    # 多答案问题取所有规范答案中的最佳 EM/F1，这是 QA benchmark 的常规口径。
    for query, solution, original in zip(eligible, answered, solutions):
        if solution.question != query.text:
            raise RuntimeError("Reader returned answers in a different query order")
        golds = query.answer if isinstance(query.answer, list) else [query.answer]
        em = max(answer_em(solution.answer, gold) for gold in golds)
        f1 = max(answer_f1(solution.answer, gold) for gold in golds)
        em_total += em
        f1_total += f1
        retrieved = [item.get("warp_doc_id") for item in (original.doc_metadata or []) if item.get("warp_doc_id")]
        predictions.append({
            "query_id": query.id,
            "query": query.text,
            "retrieved_doc_ids": retrieved,
            "prediction": solution.answer,
            "gold_answers": golds,
            "answer_em": em, "answer_f1": f1,
        })
    after = tracker.get("reader")
    usage = {key: after.get(key, 0) - before.get(key, 0) for key in set(after) | set(before)}
    return {
        "answer_em": em_total / len(eligible), "answer_f1": f1_total / len(eligible),
        "num_queries": len(eligible), "reader_usage": usage, "predictions": predictions,
    }
