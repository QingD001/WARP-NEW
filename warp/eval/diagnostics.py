"""Post-retrieval evidence diagnostics; never supplied to retrieval decisions."""


def annotate_steps(trace, gold_doc_ids, final_doc_ids, ks):
    gold = set(gold_doc_ids)
    steps = trace.get("steps") or [{**trace, "final_doc_ids": final_doc_ids}]
    previous = {k: set() for k in ks}
    output = []
    for step in steps:
        candidates = set(step.get("fused_candidate_doc_ids", step.get("retrieved_doc_ids", [])))
        values = {}
        for k in ks:
            found = set(step.get("final_doc_ids", [])[:k]) & gold
            values[str(k)] = {
                "evidence_recall": len(found) / len(gold),
                "complete_evidence": float(gold <= found),
                "new_gold": sorted(found - previous[k]), "lost_gold": sorted(previous[k] - found),
                "candidate_gold_dropped": sorted((candidates & gold) - found),
            }
            previous[k] = found
        output.append({"step": step.get("step", 1), "metrics_by_k": values,
                       "candidate_gold": sorted(candidates & gold)})
    trace["evidence_diagnostics"] = output
    return trace
