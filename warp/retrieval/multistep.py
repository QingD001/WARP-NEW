"""Bounded passage-feedback retrieval, without gold access or extra LLM calls.

This is evidence expansion, not an implementation of IRCoT reasoning.
"""
from warp.audit import emit
from warp.retrieval.hybrid import fuse_and_rerank


def retrieve_steps(query, k, search_once, documents, reranker, *, candidate_k,
                   steps=1, feedback_docs=2, feedback_chars=400, trace=None):
    if steps <= 0 or feedback_docs <= 0 or feedback_chars <= 0 or candidate_k < k:
        raise ValueError("Invalid multistep limits")
    doc_map = documents if isinstance(documents, dict) else {doc.id: doc for doc in documents}
    runs, history, used = [], [], set()
    current_query = query
    stop = "step_limit"
    final = []
    for step in range(steps):
        detail = {}
        rows = search_once(current_query, candidate_k, detail)
        runs.append(rows)
        # Always judge final relevance against the original question.
        if step == 0:
            # search_once already reranked against the original query. Preserve
            # exact single-step behavior and avoid a duplicate CrossEncoder pass.
            final = rows[:k]
            detail.setdefault("fused_candidate_doc_ids", [r.doc_id for r in rows])
        else:
            final = fuse_and_rerank(query, runs, reranker, k=k, candidate_k=candidate_k,
                                    source="multistep", trace=detail)
        history.append({"step": step + 1, "search_query": current_query,
                        "retrieved_doc_ids": [r.doc_id for r in rows],
                        "final_doc_ids": [r.doc_id for r in final], **detail})
        if step + 1 == steps:
            break
        feedback = [r.doc_id for r in final if r.doc_id not in used and r.doc_id in doc_map][:feedback_docs]
        if not feedback:
            stop = "no_new_feedback"
            break
        used.update(feedback)
        history[-1]["feedback_doc_ids"] = feedback
        current_query = query + "\nEvidence:\n" + "\n".join(doc_map[key].content[:feedback_chars] for key in feedback)
    regional_candidates = {}
    for step in history:
        for region, ids in step.get("graph_candidate_doc_ids", {}).items():
            regional_candidates.setdefault(region, set()).update(ids)
    payload = {"base_candidate_doc_ids": sorted({key for step in history for key in step.get("base_candidate_doc_ids", [])}),
               "graph_candidate_doc_ids": {r: sorted(ids) for r, ids in regional_candidates.items()},
               "routed_regions": sorted({r for step in history for r in step.get("routed_regions", [])}),
               "graph_call_count": sum(len(step.get("routed_regions", [])) for step in history),
               "fused_candidate_doc_ids": history[-1].get("fused_candidate_doc_ids", []),
               "candidate_summary_scope": "union_across_steps; fused_candidates_from_final_step",
               "query": query, "steps": history, "stop_reason": stop,
               "steps_executed": len(history), "expansion": "passage_feedback"}
    if trace is not None:
        trace.update(payload)
    emit("multistep_retrieval", payload)
    return final
