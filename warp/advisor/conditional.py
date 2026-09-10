"""Greedy conditional measurement with bounded pair lookahead; no deployment-token cutoff."""
from itertools import combinations
import random
from warp.advisor.objective import utility
from warp.audit import emit


def select_conditional(model, budget=None):
    cfg = model.config
    # A common sample makes candidate gains comparable and includes collateral losses.
    queries = sorted(model.bundle.train, key=lambda q: q.id)
    if len(queries) > cfg.conditional_max_queries:
        queries = sorted(random.Random(cfg.seed).sample(queries, cfg.conditional_max_queries), key=lambda q: q.id)
    ranked = sorted(model.probes, key=lambda key: (
        -model.features[key].query_freq * max(model.estimated_gains[key], 0) / model.costs[key], key))
    if budget is not None:
        ranked = [region_id for region_id in ranked if model.costs[region_id] <= budget + 1e-9]
    candidates = ranked[:cfg.conditional_candidates]
    cache, records, selected = {}, [], set()

    def measure(regions):
        key = tuple(sorted(regions))
        if key not in cache:
            rows = []
            for q in queries:
                results = model.search(q.text, cfg.retrieval_k, set(regions))
                rows.append({"query_id": q.id, "doc_ids": [r.doc_id for r in results],
                             "utility": utility(results, q.gold_doc_ids, cfg.benefit_objective, cfg.complete_weight),
                             "evidence_recall": utility(results, q.gold_doc_ids, "evidence_recall"),
                             "complete_evidence": utility(results, q.gold_doc_ids, "complete_evidence")})
            cache[key] = rows
            emit("conditional_measurement", {"selected_regions": key, "per_query": rows})
        return cache[key]

    if not candidates:
        return [], {"rounds": [], "evaluated_sets": 0, "sample_query_ids": []}
    for _ in range(cfg.conditional_rounds):
        base = measure(selected)
        spent = sum(model.costs[r] for r in selected)
        remaining = [r for r in candidates if r not in selected]
        proposals = [(r,) for r in remaining]
        # Pairs can escape the zero-singleton-gain trap; no new graphs are built.
        proposals += list(combinations(remaining, 2))[:cfg.conditional_pairs]
        scored = []
        for addition in proposals:
            cost = sum(model.costs[r] for r in addition)
            if budget is not None and spent + cost > budget + 1e-9:
                continue
            proposed = selected | set(addition)
            if tuple(sorted(proposed)) not in cache and len(cache) >= cfg.conditional_max_evaluations:
                continue
            rows = measure(proposed)
            deltas = [row["utility"] - old["utility"] for row, old in zip(rows, base)]
            gain = sum(deltas) / (len(deltas) + cfg.gain_prior_queries)
            scored.append({"addition": list(addition), "gain": gain, "score": gain / cost,
                           "per_query_delta": dict(zip([q.id for q in queries], deltas))})
        best = max(scored, key=lambda row: row["score"], default=None)
        record = {"selected_before": sorted(selected), "candidates": scored,
                  "chosen": best["addition"] if best and best["gain"] > 0 else []}
        records.append(record)
        emit("conditional_selection", record)
        if not record["chosen"]:
            break
        selected.update(record["chosen"])
    return sorted(selected), {"rounds": records, "evaluated_sets": len(cache),
                              "sample_query_ids": [q.id for q in queries],
                              "candidate_regions": candidates,
                              "evaluation_limit_reached": len(cache) >= cfg.conditional_max_evaluations}
