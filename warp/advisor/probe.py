"""分层选择 Region 构图，并用最终部署排序路径生成监督标签。"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from warp.audit import emit

from warp.graph.builder import GraphBuilder, RegionalGraph
from warp.graph.retriever import GraphRetriever
from warp.models import Document, Query, Region, RegionFeatures, SearchResult
from warp.retrieval.hybrid import fuse_and_rerank
from warp.retrieval.reranker import Reranker


def evidence_recall(results: list[SearchResult], gold_doc_ids: list[str]) -> float:
    gold = set(gold_doc_ids)
    if not gold:
        raise ValueError("Probe queries require gold evidence")
    return len(gold & {result.doc_id for result in results}) / len(gold)


def complete_evidence(results: list[SearchResult], gold_doc_ids: list[str]) -> float:
    gold = set(gold_doc_ids)
    if not gold:
        raise ValueError("Probe queries require gold evidence")
    return float(gold.issubset({result.doc_id for result in results}))


@dataclass
class ProbeOutcome:
    region_id: str
    gain: float
    recall_gain: float
    complete_gain: float
    base_utility: float
    graph_utility: float
    query_count: int
    graph: RegionalGraph
    per_query: list[dict] = field(default_factory=list)
    eligible_query_count: int = 0
    retrieval_usage: dict = field(default_factory=dict)


class RegionProber:
    """使用与上线检索完全一致的 RRF + CrossEncoder 计算 counterfactual gain。"""

    def __init__(
        self, graph_builder: GraphBuilder, graph_retriever: GraphRetriever, reranker: Reranker,
        probe_fraction: float, retrieval_k: int, candidate_k: int, objective: str, seed: int,
        max_queries: int | None = None, complete_weight: float = 0.5, search_fn=None,
    ) -> None:
        if not 0.0 < probe_fraction <= 1.0:
            raise ValueError("probe_fraction must be in (0, 1]")
        if objective not in {"evidence_recall", "complete_evidence", "mixed"}:
            raise ValueError("Probe objective must be evidence_recall, complete_evidence or mixed")
        self.graph_builder = graph_builder
        self.graph_retriever = graph_retriever
        self.reranker = reranker
        self.probe_fraction = probe_fraction
        self.retrieval_k = retrieval_k
        self.candidate_k = candidate_k
        self.objective = objective
        self.seed = seed
        if max_queries is not None and max_queries <= 0:
            raise ValueError("probe max_queries must be positive")
        self.max_queries = max_queries
        self.complete_weight = complete_weight
        self.search_fn = search_fn
        if not 0 <= complete_weight <= 1:
            raise ValueError("complete_weight must be in [0, 1]")

    def select_probe_regions(self, regions: list[Region], features: dict[str, RegionFeatures], *,
                             costs: dict[str, float] | None = None, budget: float | None = None) -> list[Region]:
        eligible = [region for region in regions if features[region.id].query_freq > 0]
        if not eligible:
            return []
        costs = costs or {r.id: float(features[r.id].num_tokens) for r in eligible}
        if any(costs.get(r.id, 0) <= 0 for r in eligible):
            raise ValueError("Probe selection requires positive costs")
        if budget is not None and budget < 0:
            raise ValueError("Probe budget must be nonnegative")
        count = min(len(eligible), max(1, round(len(eligible) * self.probe_fraction)))
        # Local missing evidence is an acquisition heuristic, not a gain label.
        def priority(region):
            f = features[region.id]
            opportunity = f.query_freq * f.gold_query_rate * (1 - f.base_recall)
            return (-opportunity / costs[region.id], costs[region.id], region.id)
        ranked = sorted(eligible, key=priority)
        exploratory = sorted(eligible, key=lambda r: r.id)
        random.Random(self.seed).shuffle(exploratory)
        selected, spent = [], 0.0
        remaining = {r.id for r in eligible}
        while remaining and len(selected) < count:
            # Every fifth choice explores, including the second choice for small designs.
            pool = exploratory if len(selected) % 5 == 1 else ranked
            candidate = next((r for r in pool if r.id in remaining and
                              (budget is None or spent + costs[r.id] <= budget)), None)
            if candidate is None:
                break
            selected.append(candidate)
            remaining.remove(candidate.id)
            spent += costs[candidate.id]
        return sorted(selected, key=lambda region: region.id)

    def run(
        self, probe_regions: list[Region], documents: list[Document], queries: list[Query],
        region_queries: dict[str, list[str]], base_results: dict[str, list[SearchResult]],
    ) -> dict[str, ProbeOutcome]:
        query_map = {query.id: query for query in queries}
        outcomes: dict[str, ProbeOutcome] = {}
        base_cache = {}
        for region in probe_regions:
            graph = self.graph_builder.build(region, documents)
            before = self.graph_retriever.stats()
            base_recall_values: list[float] = []
            graph_recall_values: list[float] = []
            base_complete_values: list[float] = []
            graph_complete_values: list[float] = []
            query_ids = sorted(region_queries.get(region.id, []))
            eligible_count = len(query_ids)
            if self.max_queries is not None and len(query_ids) > self.max_queries:
                query_ids = sorted(random.Random(f"{self.seed}:{region.id}").sample(query_ids, self.max_queries))
            records = []
            for query_id in query_ids:
                query = query_map[query_id]
                if self.search_fn is not None:
                    if query_id not in base_cache:
                        base_cache[query_id] = self.search_fn(query.text, self.retrieval_k, None)
                    base_ranked = base_cache[query_id]
                    graph_ranked = self.search_fn(query.text, self.retrieval_k, graph)
                else:
                    if query_id not in base_cache:
                        base_cache[query_id] = fuse_and_rerank(
                            query.text, [base_results[query_id]], self.reranker,
                            k=self.retrieval_k, candidate_k=self.candidate_k, source="probe_base")
                    base_ranked = base_cache[query_id]
                    graph_results = self.graph_retriever.search(query.text, graph, self.candidate_k)
                    graph_ranked = fuse_and_rerank(
                        query.text, [base_results[query_id], graph_results], self.reranker,
                        k=self.retrieval_k, candidate_k=self.candidate_k, source="probe_graph",
                    )
                base_recall_values.append(evidence_recall(base_ranked, query.gold_doc_ids))
                graph_recall_values.append(evidence_recall(graph_ranked, query.gold_doc_ids))
                base_complete_values.append(complete_evidence(base_ranked, query.gold_doc_ids))
                graph_complete_values.append(complete_evidence(graph_ranked, query.gold_doc_ids))
                local_gold = sorted(set(query.gold_doc_ids) & set(region.doc_ids))
                record = {"query_id": query_id, "region_id": region.id, "gold_doc_ids": query.gold_doc_ids,
                          "local_gold_doc_ids": local_gold,
                          "base_results": [row.doc_id for row in base_ranked],
                          "graph_results": [row.doc_id for row in graph_ranked],
                          "new_gold": sorted((set(row.doc_id for row in graph_ranked) - set(row.doc_id for row in base_ranked)) & set(query.gold_doc_ids)),
                          "lost_gold": sorted((set(row.doc_id for row in base_ranked) - set(row.doc_id for row in graph_ranked)) & set(query.gold_doc_ids)),
                          "recall_gain": graph_recall_values[-1] - base_recall_values[-1],
                          "complete_gain": graph_complete_values[-1] - base_complete_values[-1],
                          "local_recall_gain": (evidence_recall(graph_ranked, local_gold) - evidence_recall(base_ranked, local_gold)) if local_gold else None}
                record["utility_gain"] = ((1 - self.complete_weight) * record["recall_gain"] + self.complete_weight * record["complete_gain"]
                    if self.objective == "mixed" else record["recall_gain"] if self.objective == "evidence_recall" else record["complete_gain"])
                records.append(record)
                emit("probe_query", record)
            if not base_recall_values:
                raise ValueError(f"Probe region {region.id} has no routed design queries")
            base_recall = sum(base_recall_values) / len(base_recall_values)
            graph_recall = sum(graph_recall_values) / len(graph_recall_values)
            base_complete = sum(base_complete_values) / len(base_complete_values)
            graph_complete = sum(graph_complete_values) / len(graph_complete_values)
            recall_gain = graph_recall - base_recall
            complete_gain = graph_complete - base_complete
            if self.objective == "evidence_recall":
                gain, base_utility, graph_utility = recall_gain, base_recall, graph_recall
            elif self.objective == "complete_evidence":
                gain, base_utility, graph_utility = complete_gain, base_complete, graph_complete
            else:
                w = self.complete_weight
                base_utility = (1 - w) * base_recall + w * base_complete
                graph_utility = (1 - w) * graph_recall + w * graph_complete
                gain = graph_utility - base_utility
            outcomes[region.id] = ProbeOutcome(
                region.id, gain, recall_gain, complete_gain, base_utility, graph_utility,
                len(base_recall_values), graph, records, eligible_count, self.graph_retriever.delta(before),
            )
        return outcomes
