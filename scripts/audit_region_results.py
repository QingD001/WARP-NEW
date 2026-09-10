#!/usr/bin/env python3
"""Compare regional methods using saved artifacts; never rerun retrieval/models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _stage_summary(row: dict[str, Any], k: int) -> dict[str, Any] | None:
    traces = row.get("retrieval_traces")
    if traces is None:
        return None
    counts = {key: 0 for key in (
        "queries_with_graph_calls", "graph_calls", "queries_with_new_graph_documents",
        "queries_with_new_graph_gold", "queries_with_new_graph_gold_after_fusion",
        "queries_with_new_graph_gold_in_topk", "queries_with_complete_gold_in_base_candidates",
        "queries_with_complete_gold_in_union", "queries_with_complete_gold_after_fusion",
    )}
    final = row.get("retrieved_doc_ids", {})
    for qid, trace in traces.items():
        base = set(trace.get("base_candidate_doc_ids", []))
        graph = {doc_id for ids in trace.get("graph_candidate_doc_ids", {}).values() for doc_id in ids}
        gold = set(trace["gold_doc_ids"])
        fused = set(trace["fused_candidate_doc_ids"])
        new_gold = (graph - base) & gold
        counts["queries_with_graph_calls"] += bool(trace.get("routed_regions", []))
        counts["graph_calls"] += trace.get("graph_call_count", len(trace.get("routed_regions", [])))
        counts["queries_with_new_graph_documents"] += bool(graph - base)
        counts["queries_with_new_graph_gold"] += bool(new_gold)
        counts["queries_with_new_graph_gold_after_fusion"] += bool(new_gold & fused)
        counts["queries_with_new_graph_gold_in_topk"] += bool(new_gold & set(final.get(qid, [])[:k]))
        counts["queries_with_complete_gold_in_base_candidates"] += bool(gold) and gold <= base
        counts["queries_with_complete_gold_in_union"] += bool(gold) and gold <= (base | graph)
        counts["queries_with_complete_gold_after_fusion"] += bool(gold) and gold <= fused
    return {"num_queries": len(traces), "k": k, **counts}


def compare_rows(candidate: dict[str, Any], reference: dict[str, Any], k: int = 10) -> dict[str, Any]:
    left, right = candidate["per_query"], reference["per_query"]
    if not left or set(left) != set(right):
        raise ValueError("Comparison requires non-empty, identical query ID coverage")
    selected_left, selected_right = set(candidate["selected_regions"]), set(reference["selected_regions"])
    union = selected_left | selected_right
    metrics = {}
    for name in (f"evidence_recall@{k}", f"complete_evidence@{k}"):
        differences = {qid: left[qid][name] - right[qid][name] for qid in left}
        metrics[name] = {
            "mean_difference": sum(differences.values()) / len(left),
            "improved_queries": sum(value > 0 for value in differences.values()),
            "degraded_queries": sum(value < 0 for value in differences.values()),
            "equal_queries": sum(value == 0 for value in differences.values()),
            "changed_query_ids": sorted(qid for qid, value in differences.items() if value != 0),
        }
    output = {
        "candidate": candidate["method"], "reference": reference["method"],
        "num_queries": len(left),
        "candidate_selected_regions": sorted(selected_left),
        "reference_selected_regions": sorted(selected_right),
        "same_region_set": selected_left == selected_right,
        "region_jaccard": len(selected_left & selected_right) / len(union) if union else 1.0,
        "metrics": metrics,
        "candidate_stages": _stage_summary(candidate, k),
        "reference_stages": _stage_summary(reference, k),
    }
    result_left, result_right = candidate.get("retrieved_doc_ids"), reference.get("retrieved_doc_ids")
    output["result_comparison"] = None
    if result_left is not None and result_right is not None:
        if set(result_left) != set(left) or set(result_right) != set(right):
            raise ValueError("Saved result IDs do not match metric query coverage")
        output["result_comparison"] = {
            "same_ranked_topk_queries": sum(result_left[q][:k] == result_right[q][:k] for q in left),
            "same_topk_document_set_queries": sum(set(result_left[q][:k]) == set(result_right[q][:k]) for q in left),
        }
    trace_left, trace_right = candidate.get("retrieval_traces"), reference.get("retrieval_traces")
    output["trace_comparison"] = None
    if trace_left is not None and trace_right is not None:
        if set(trace_left) != set(left) or set(trace_right) != set(right):
            raise ValueError("Saved traces do not match metric query coverage")
        output["trace_comparison"] = {
            "same_routed_region_set_queries": sum(
                set(trace_left[q]["routed_regions"]) == set(trace_right[q]["routed_regions"]) for q in left),
            "same_fused_candidate_set_queries": sum(
                set(trace_left[q]["fused_candidate_doc_ids"]) == set(trace_right[q]["fused_candidate_doc_ids"])
                for q in left),
        }
    return output


def audit_artifact(artifact: dict[str, Any], candidate: str, reference: str, k: int) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for row in artifact["quality_cost_curve"]:
        if row["method"] not in {candidate, reference}:
            continue
        key = (row.get("design_seed"), row.get("fold"), row["budget_fraction"])
        group = groups.setdefault(key, {})
        if row["method"] in group:
            raise ValueError(f"Duplicate method trial: {key}, {row['method']}")
        group[row["method"]] = row
    output = []
    for (seed, fold, budget), group in groups.items():
        if candidate not in group or reference not in group:
            raise ValueError(f"Missing comparison method at seed={seed}, fold={fold}, budget={budget}")
        output.append({"design_seed": seed, "fold": fold, "budget_fraction": budget,
                       **compare_rows(group[candidate], group[reference], k)})
    if not output:
        raise ValueError("No matching regional method trials in quality_cost_curve")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--candidate", default="warp")
    parser.add_argument("--reference", default="random_region")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.k <= 0:
        parser.error("--k must be positive")
    artifact = json.loads(args.input.read_text(encoding="utf-8"))
    output = {"input": str(args.input), "comparisons": audit_artifact(
        artifact, args.candidate, args.reference, args.k,
    ), "note": "Null comparisons mean the old artifact did not save document IDs or traces; equal metrics do not imply equal retrieval."}
    rendered = json.dumps(output, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
