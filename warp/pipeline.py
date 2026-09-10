"""WARP-G 的训练/设计阶段与测试阶段编排器。

`fit` 是唯一允许读取 train/design query 的设计入口；评测函数只接收显式传入的
dev/test query。图 builder、图 retriever、base retriever 和 reranker 均可注入，
因此 advisor 算法不绑定具体 GraphRAG 实现。
"""

from __future__ import annotations

import math

from dataclasses import asdict, dataclass
import time
import random
import statistics
from typing import Any
from warp.audit import audit_scope, emit, audit_context

from warp.advisor.conditional import select_conditional
from warp.advisor.objective import utility
from warp.retrieval.multistep import retrieve_steps
from warp.advisor.features import RegionFeatureExtractor
from warp.advisor.estimator import estimate_benefits
from warp.advisor.probe import ProbeOutcome, RegionProber, complete_evidence, evidence_recall
from warp.advisor.selector import RegionSelector
from warp.eval.diagnostics import annotate_steps
from warp.eval.construction_cost import aggregate_costs
from warp.eval.cutoffs import RETRIEVAL_KS
from warp.eval.retrieval import evaluate_retrieval
from warp.graph.builder import GraphBuilder, RegionalGraph
from warp.graph.retriever import GraphRetriever
from warp.models import DatasetBundle, Region, RegionFeatures, SearchResult
from warp.partition.coaccess_graph import CoaccessGraph, CoaccessGraphBuilder
from warp.partition.leiden import RegionPartitioner
from warp.retrieval.hybrid import HybridRetriever, fuse_and_rerank
from warp.retrieval.reranker import Reranker


@dataclass
class WARPConfig:
    """与具体 HippoRAG2 参数解耦的 WARP 物理设计超参数。"""
    retrieval_k: int = 10
    candidate_k: int = 50
    routing_k: int = 20
    coaccess_k: int = 20
    semantic_k: int = 3
    semantic_lambda: float = 0.05
    probe_fraction: float = 0.2
    probe_max_queries: int | None = 64
    probe_budget_fraction: float | None = 0.1
    gain_prior_queries: float = 16.0
    benefit_objective: str = "mixed"
    complete_weight: float = 0.5
    selection_mode: str = "conditional"
    conditional_max_queries: int = 32
    conditional_candidates: int = 6
    conditional_pairs: int = 6
    conditional_rounds: int = 3
    conditional_max_evaluations: int = 16
    retrieval_steps: int = 1
    feedback_docs: int = 2
    feedback_chars: int = 400
    dispersion_pairs: int = 4096
    interaction_pairs: int = 0
    partition_resolution: float = 1.0
    min_region_size: int = 1
    partition_mode: str = "combined"
    seed: int = 42
    retrieval_ks: tuple[int, ...] = RETRIEVAL_KS
    multistep_max_steps: int = 0

    def __post_init__(self) -> None:
        if self.benefit_objective not in {"mixed", "evidence_recall", "complete_evidence"} or not 0 <= self.complete_weight <= 1:
            raise ValueError("Invalid benefit objective or complete_weight")
        if self.selection_mode not in {"conditional", "independent"}:
            raise ValueError("selection_mode must be conditional or independent")
        if any(v <= 0 for v in (self.conditional_max_queries, self.conditional_candidates,
                                self.conditional_rounds, self.conditional_max_evaluations, self.retrieval_steps, self.feedback_docs, self.feedback_chars)) or self.conditional_pairs < 0:
            raise ValueError("Design/retrieval limits must be positive; pair count may be zero")
        if not math.isfinite(self.gain_prior_queries) or self.gain_prior_queries < 0:
            raise ValueError("gain_prior_queries must be finite and nonnegative")
        if self.retrieval_k <= 0 or self.routing_k <= 0 or self.candidate_k < max(self.retrieval_k, self.routing_k):
            raise ValueError("candidate_k must cover positive retrieval_k and routing_k")
        if self.dispersion_pairs <= 0 or self.min_region_size <= 0 or self.interaction_pairs < 0:
            raise ValueError("dispersion_pairs, interaction_pairs and min_region_size must be positive")
        if self.partition_mode not in {"combined", "query", "semantic", "random"}:
            raise ValueError("partition_mode must be combined, query, semantic, or random")
        if self.probe_budget_fraction is not None and not 0 < self.probe_budget_fraction <= 1:
            raise ValueError("probe_budget_fraction must be in (0, 1]")
        self.retrieval_ks = tuple(int(k) for k in self.retrieval_ks)
        if not self.retrieval_ks or any(k <= 0 for k in self.retrieval_ks):
            raise ValueError("retrieval_ks must be a non-empty sequence of positive cutoffs")
        if int(self.multistep_max_steps) < 0:
            raise ValueError("multistep_max_steps must be nonnegative")
        self.multistep_max_steps = int(self.multistep_max_steps)


class WARPG:
    """End-to-end WARP-G physical-design and retrieval pipeline."""

    def __init__(self, config: WARPConfig, graph_builder: GraphBuilder,
                 graph_retriever: GraphRetriever, reranker: Reranker,
                 base_retriever: HybridRetriever) -> None:
        self.config = config
        self.base = base_retriever
        self.graph_builder = graph_builder
        self.graph_retriever = graph_retriever
        self.reranker = reranker
        self.selector = RegionSelector(config.seed)
        self.bundle: DatasetBundle | None = None
        self.coaccess: CoaccessGraph | None = None
        self.regions: list[Region] = []
        self.region_map: dict[str, Region] = {}
        self.doc_region: dict[str, str] = {}
        self.features: dict[str, RegionFeatures] = {}
        self.region_queries: dict[str, list[str]] = {}
        self.probes: dict[str, ProbeOutcome] = {}
        self.estimated_gains: dict[str, float] = {}
        self.graphs: dict[str, RegionalGraph] = {}
        self.full_graph: RegionalGraph | None = None
        self.costs: dict[str, float] = {}
        self.design_timings: dict[str, float] = {}
        self.interaction_analysis: dict[str, Any] = {}
        self.design_retrieval_usage: dict[str, float | int] = {}
        self.selection_reports = {}
        self.selection_cache = {}
        self._feedback_documents = None

    def fit(self, bundle: DatasetBundle) -> "WARPG":
        """完成基础索引、分区、特征、probe 和 收益估计。"""
        if not bundle.documents or not bundle.train:
            raise ValueError("WARP-G requires a non-empty corpus and training/design queries")
        doc_ids = [document.id for document in bundle.documents]
        if len(doc_ids) != len(set(doc_ids)):
            raise ValueError("Corpus document IDs must be unique")
        if any(not document.content.strip() for document in bundle.documents):
            raise ValueError("Corpus documents must contain text")
        contents = [document.content for document in bundle.documents]
        if len(contents) != len(set(contents)):
            raise ValueError("HippoRAG document contents must be unique")
        known = set(doc_ids)
        for split_name, queries in (("train", bundle.train), ("dev", bundle.dev), ("test", bundle.test)):
            query_ids = [query.id for query in queries]
            if len(query_ids) != len(set(query_ids)):
                raise ValueError(f"{split_name} query IDs must be unique")
            missing = sorted({doc_id for query in queries for doc_id in query.gold_doc_ids if doc_id not in known})
            if missing:
                raise ValueError(f"{split_name} gold evidence is absent from corpus: {missing[:10]}")
        self.bundle = bundle
        # Region IDs are reused across designs; in-memory graphs from an earlier
        # fit must never survive into a new corpus/partition (disk caching is
        # independently protected by the builder's content fingerprint).
        self.graphs = {}
        self.full_graph = None
        self.probes = {}
        self.estimated_gains = {}
        self.coaccess = None
        self.regions = []
        self.region_map = {}
        self.doc_region = {}
        self.features = {}
        self.region_queries = {}
        self.costs = {}
        self.interaction_analysis = {}
        self.design_retrieval_usage = {}
        self.selection_reports = {}
        self.selection_cache = {}
        self._feedback_documents = None
        self.design_timings = {}
        # Base 索引覆盖 100% corpus，是所有方法共享且不计为选择性 Graph 成本的底座。
        started = time.perf_counter()
        self.base.fit(bundle.documents)
        self.design_timings["base_index_seconds"] = time.perf_counter() - started
        # 下面所有 workload 信号只来自 bundle.train，避免 test leakage。
        started = time.perf_counter()
        self.coaccess = CoaccessGraphBuilder(
            self.config.coaccess_k, self.config.semantic_k, self.config.semantic_lambda,
        ).build(bundle.documents, bundle.train, self.base)
        partitioner = RegionPartitioner(
            resolution=self.config.partition_resolution, seed=self.config.seed,
            min_region_size=self.config.min_region_size,
        )
        if self.config.partition_mode == "combined":
            self.regions = partitioner.partition(self.coaccess)
        elif self.config.partition_mode == "query":
            query_graph = CoaccessGraph(
                self.coaccess.nodes, dict(self.coaccess.query_edges), dict(self.coaccess.query_edges), {},
                self.coaccess.query_results,
            )
            self.regions = partitioner.partition(query_graph)
        elif self.config.partition_mode == "semantic":
            semantic_edges = {key: self.config.semantic_lambda * value
                              for key, value in self.coaccess.semantic_edges.items()}
            semantic_graph = CoaccessGraph(
                self.coaccess.nodes, semantic_edges, {}, dict(self.coaccess.semantic_edges),
                self.coaccess.query_results,
            )
            self.regions = partitioner.partition(semantic_graph)
        else:
            reference = partitioner.partition(self.coaccess)
            shuffled = list(self.coaccess.nodes)
            random.Random(self.config.seed).shuffle(shuffled)
            sizes = [len(region.doc_ids) for region in reference]
            offset = 0
            self.regions = []
            for index, size in enumerate(sizes):
                self.regions.append(Region(f"r{index:04d}", sorted(shuffled[offset:offset + size])))
                offset += size
        self.design_timings["partition_seconds"] = time.perf_counter() - started
        self.region_map = {region.id: region for region in self.regions}
        self.doc_region = {doc_id: region.id for region in self.regions for doc_id in region.doc_ids}
        started = time.perf_counter()
        extractor = RegionFeatureExtractor(
            self.config.routing_k, self.config.candidate_k,
            self.config.dispersion_pairs, self.config.seed,
        )
        self.features, self.region_queries, base_results = extractor.extract(
            self.regions, bundle.documents, bundle.train, self.base, self.coaccess,
        )
        self.design_timings["feature_seconds"] = time.perf_counter() - started
        # 选择前只能使用 token proxy；actual cost 必须等真实 build 后才可观测。
        self.costs = {region.id: self.graph_builder.estimate_cost(region, bundle.documents)
                      for region in self.regions}
        prober = RegionProber(
            self.graph_builder, self.graph_retriever, self.reranker,
            self.config.probe_fraction, self.config.retrieval_k, self.config.candidate_k,
            self.config.benefit_objective, self.config.seed,
            max_queries=self.config.probe_max_queries,
            complete_weight=self.config.complete_weight,
            search_fn=self._probe_search,
        )
        probe_regions = prober.select_probe_regions(
            self.regions, self.features, costs=self.costs,
            budget=(self.config.probe_budget_fraction * self.full_graph_cost
                    if self.config.probe_budget_fraction is not None else None))
        before_design_retrieval = self.graph_retriever.stats()
        self.probes = prober.run(probe_regions, bundle.documents, bundle.train,
                                 self.region_queries, base_results)
        self.graphs.update({region_id: outcome.graph for region_id, outcome in self.probes.items()})
        started = time.perf_counter()
        self.estimated_gains = estimate_benefits(
            self.features, self.probes, self.config.gain_prior_queries)
        self.design_timings["benefit_estimation_seconds"] = time.perf_counter() - started
        self.design_retrieval_usage = self.graph_retriever.delta(before_design_retrieval)
        with audit_scope(stage="interaction_analysis"):
            self.interaction_analysis = self._probe_interactions()
        return self

    def _probe_search(self, query, k, graph):
        if graph is not None:
            self.graphs[graph.region_id] = graph
        return self.search(query, k, {graph.region_id} if graph is not None else set())

    def routing_diagnostics(self, queries: list[Any]) -> dict[str, float]:
        """测量 Base router 能否命中 gold evidence 所在 Region。"""
        eligible = [query for query in queries if query.gold_doc_ids]
        if not eligible:
            raise ValueError("Routing diagnostics require gold evidence")
        complete = any_hit = 0.0
        for query in eligible:
            gold_regions = {self.doc_region[doc_id] for doc_id in query.gold_doc_ids}
            routed = {self.doc_region[result.doc_id]
                      for result in self.base.search(query.text, self.config.candidate_k)[:self.config.routing_k]}
            complete += float(gold_regions.issubset(routed))
            any_hit += float(bool(gold_regions & routed))
        return {
            "queries": float(len(eligible)),
            "any_gold_region_recall": any_hit / len(eligible),
            "complete_gold_region_recall": complete / len(eligible),
        }

    def _probe_interactions(self) -> dict[str, Any]:
        """直接测量区域图二阶交互，检验独立 gain 假设。"""
        if self.bundle is None:
            raise RuntimeError("fit must be called first")
        region_ids = sorted(self.probes)
        pairs = [(left, right) for index, left in enumerate(region_ids) for right in region_ids[index + 1:]]
        if len(pairs) > self.config.interaction_pairs:
            pairs = random.Random(self.config.seed).sample(pairs, self.config.interaction_pairs)
        query_map = {query.id: query for query in self.bundle.train}
        before = self.graph_retriever.stats()
        rows: list[dict[str, float | str | int]] = []
        for left, right in pairs:
            query_ids = sorted(set(self.region_queries[left]) | set(self.region_queries[right]))
            base_values, left_values, right_values, joint_values = [], [], [], []
            for query_id in query_ids:
                query = query_map[query_id]
                metric = lambda results, gold: utility(results, gold, self.config.benefit_objective, self.config.complete_weight)
                base_values.append(metric(self.search(query.text, self.config.retrieval_k, set()), query.gold_doc_ids))
                left_values.append(metric(self.search(query.text, self.config.retrieval_k, {left}), query.gold_doc_ids))
                right_values.append(metric(self.search(query.text, self.config.retrieval_k, {right}), query.gold_doc_ids))
                joint_values.append(metric(self.search(query.text, self.config.retrieval_k, {left, right}), query.gold_doc_ids))
            base_mean = sum(base_values) / len(base_values)
            left_gain = sum(left_values) / len(left_values) - base_mean
            right_gain = sum(right_values) / len(right_values) - base_mean
            joint_gain = sum(joint_values) / len(joint_values) - base_mean
            rows.append({
                "left_region": left, "right_region": right, "queries": len(query_ids),
                "left_gain": left_gain, "right_gain": right_gain, "joint_gain": joint_gain,
                "interaction": joint_gain - left_gain - right_gain,
            })
        interactions = [abs(float(row["interaction"])) for row in rows]
        return {
            "pairs": rows,
            "mean_absolute_interaction": sum(interactions) / len(interactions) if interactions else 0.0,
            "online_analysis_cost": self.graph_retriever.delta(before),
        }

    @property
    def full_graph_cost(self) -> float:
        """返回所有区域 token proxy 之和，仅作成本对照分母，不再用于截断。"""
        return sum(self.costs.values())

    def select(self, method: str = "warp") -> list[str]:
        """WARP 按自身规则选完全部合格区域；controls 只换公式并对齐区域个数。"""
        method = method.lower()
        if method == "warp" and self.config.selection_mode == "conditional":
            key = "full_pipeline"
            if key not in self.selection_cache:
                before = self.graph_retriever.stats()
                started = time.perf_counter()
                with audit_scope(stage="conditional_design", selection=key):
                    selected, report = select_conditional(self)
                report["retrieval_usage"] = self.graph_retriever.delta(before)
                report["wall_seconds"] = time.perf_counter() - started
                self.selection_reports[key] = report
                self.selection_cache[key] = selected
            return list(self.selection_cache[key])
        if method == "warp":
            return self.selector.select("warp", self.features, self.costs, self.estimated_gains)
        return self.selector.select(
            method, self.features, self.costs, self.estimated_gains, limit=len(self.select("warp")),
        )

    def search_multistep(self, query, k, search_once, *, trace=None):
        if self.config.retrieval_steps == 1:
            return search_once(query, k, trace if trace is not None else {})
        if self._feedback_documents is None:
            self._feedback_documents = {doc.id: doc for doc in self.bundle.documents}
        return retrieve_steps(query, k, search_once, self._feedback_documents, self.reranker,
                              candidate_k=self.config.candidate_k, steps=self.config.retrieval_steps,
                              feedback_docs=self.config.feedback_docs, feedback_chars=self.config.feedback_chars,
                              trace=trace)

    def search_base(self, query: str, k: int | None = None, method: str = "hybrid") -> list[SearchResult]:
        k = self.config.retrieval_k if k is None else k
        return self.search_multistep(query, k, lambda q, depth, trace: self._search_base_once(q, depth, method))

    def _search_base_once(self, query: str, k: int | None = None, method: str = "hybrid", *, trace=None) -> list[SearchResult]:
        """所有 Base baseline 也走与图方法相同的 candidate depth 和 CrossEncoder。"""
        k = self.config.retrieval_k if k is None else k
        retriever = {"bm25": self.base.bm25, "dense": self.base.dense, "hybrid": self.base}.get(method)
        if retriever is None:
            raise ValueError(f"Unknown base method: {method}")
        candidates = retriever.search(query, self.config.candidate_k)
        return fuse_and_rerank(
            query, [candidates], self.reranker, k=k,
            candidate_k=self.config.candidate_k, source=f"{method}_reranked", trace=trace,
        )

    def materialize(self, region_ids: list[str]) -> None:
        """按需构建尚未缓存的区域图；probe 图会被安全复用。"""
        if self.bundle is None:
            raise RuntimeError("fit must be called first")
        for region_id in region_ids:
            if region_id not in self.graphs:
                self.graphs[region_id] = self.graph_builder.build(self.region_map[region_id], self.bundle.documents)

    def search(self, query: str, k: int | None = None, selected_regions: set[str] | None = None,
               *, trace: dict[str, Any] | None = None) -> list[SearchResult]:
        k = self.config.retrieval_k if k is None else k
        return self.search_multistep(query, k, lambda q, depth, detail: self._search_once(
            q, depth, selected_regions, trace=detail), trace=trace)

    def _search_once(self, query: str, k: int | None = None, selected_regions: set[str] | None = None,
               *, trace: dict[str, Any] | None = None) -> list[SearchResult]:
        """检索显式选区；空集合表示 Base，缓存本身不代表部署选择。"""
        k = self.config.retrieval_k if k is None else k
        if selected_regions is None:
            raise ValueError("selected_regions must be explicit; pass set() for Base-only retrieval")
        selected_regions = set(selected_regions)
        unknown = selected_regions - set(self.region_map)
        if unknown:
            raise ValueError(f"Unknown selected_regions: {sorted(unknown)}")
        missing = selected_regions - set(self.graphs)
        if missing:
            raise ValueError(f"Selected regions are not materialized: {sorted(missing)}; call materialize first")
        base_results = self.base.search(query, self.config.candidate_k)
        # 路由是确定性的文档归属查表，不使用 LLM/agent 做检索决策。
        routed: list[str] = []
        for result in base_results[:self.config.routing_k]:
            region_id = self.doc_region[result.doc_id]
            if region_id in selected_regions and region_id not in routed:
                routed.append(region_id)
        emit("routing", {"query": query, "selected_regions": sorted(selected_regions),
                         "routed_regions": routed, "base_candidates": base_results})
        graph_results = [self.graph_retriever.search(query, self.graphs[region_id], self.config.candidate_k)
                         for region_id in routed]
        if trace is not None:
            trace.clear()
            trace.update({
                "routed_regions": routed,
                "base_candidate_doc_ids": [row.doc_id for row in base_results],
                "graph_candidate_doc_ids": {
                    region_id: [row.doc_id for row in rows]
                    for region_id, rows in zip(routed, graph_results)
                },
            })
        return fuse_and_rerank(
            query, [base_results] + graph_results, self.reranker,
            k=k, candidate_k=self.config.candidate_k, source="warp", trace=trace,
        )

    def materialize_full_graph(self) -> RegionalGraph:
        """Build one corpus-wide graph for the true Full Graph baseline.

        This is intentionally not represented as a union of independent regional
        graphs: the baseline must retain cross-region facts and synonymy edges.
        """
        if self.bundle is None:
            raise RuntimeError("fit must be called first")
        if self.full_graph is None:
            region = Region("__full_corpus__", [doc.id for doc in self.bundle.documents])
            self.full_graph = self.graph_builder.build_full_graph(region, self.bundle.documents)
        return self.full_graph

    def generate_next_query(self, prompt: str) -> str:
        """IRCoT 下一步查询生成；没有共享 LLM 时返回 END 以停止。"""
        self.last_generation_usage = {}
        llm = getattr(self.graph_builder, "_shared_llm", None)
        tracker = getattr(self.graph_builder, "_usage_tracker", None)
        if llm is None or not hasattr(llm, "infer"):
            return "END"
        previous = getattr(tracker, "phase", None) if tracker is not None else None
        previous_context = getattr(tracker, "audit_context", None) if tracker is not None else None
        before = tracker.get("ircot") if tracker is not None else {}
        if tracker is not None:
            tracker.phase = "ircot"
            tracker.audit_context = audit_context()
        try:
            result = llm.infer([{"role": "user", "content": prompt}])
        finally:
            if tracker is not None:
                tracker.phase = previous or "idle"
                tracker.audit_context = previous_context or {}
        if tracker is not None:
            after = tracker.get("ircot")
            self.last_generation_usage = {
                key: after.get(key, 0) - before.get(key, 0) for key in set(after) | set(before)
            }
        text = result[0] if isinstance(result, tuple) and result else result
        return str(text or "").strip()

    def search_full_graph(self, query: str, k: int | None = None) -> list[SearchResult]:
        k = self.config.retrieval_k if k is None else k
        return self.search_multistep(query, k, lambda q, depth, trace: self._search_full_graph_once(q, depth))

    def _search_full_graph_once(self, query: str, k: int | None = None, *, trace=None) -> list[SearchResult]:
        """融合 Base 与真正 corpus-wide 单图结果，并使用相同 reranker。"""
        k = self.config.retrieval_k if k is None else k
        graph = self.materialize_full_graph()
        base_results = self.base.search(query, self.config.candidate_k)
        graph_results = self.graph_retriever.search(query, graph, self.config.candidate_k)
        return fuse_and_rerank(
            query, [base_results, graph_results], self.reranker,
            k=k, candidate_k=self.config.candidate_k, source="full_graph", trace=trace,
        )

    def evaluate_full_graph(self, queries: list[Any], ks: tuple[int, ...] | None = None) -> dict[str, Any]:
        """评测 Base + Full Graph fusion。"""
        self.materialize_full_graph()
        return self.evaluate_search(queries, lambda q, k, trace: self._search_full_graph_once(q, k, trace=trace), ks)

    def evaluate_full_graph_only(self, queries: list[Any], ks: tuple[int, ...] | None = None) -> dict[str, Any]:
        """评测官方图检索通路；仍使用共享 CrossEncoder。"""
        graph = self.materialize_full_graph()
        return self.evaluate_search(queries, lambda q, k, trace: fuse_and_rerank(
            q, [self.graph_retriever.search(q, graph, self.config.candidate_k)], self.reranker,
            k=k, candidate_k=self.config.candidate_k, source="hipporag2_reranked", trace=trace), ks)

    def evaluate(self, queries: list[Any], selected_regions: list[str], ks: tuple[int, ...] | None = None) -> dict[str, Any]:
        """物化给定区域并保存同一次检索的候选轨迹，支持逐题消融审计。"""
        self.materialize(selected_regions)
        return self.evaluate_search(queries, lambda q, k, trace: self._search_once(
            q, k, set(selected_regions), trace=trace), ks)

    def evaluate_search(self, queries, search_once, ks=None):
        traces = []
        ks = tuple(self.config.retrieval_ks if ks is None else ks)
        def search(query, k):
            trace = {}
            results = self.search_multistep(query, k, search_once, trace=trace)
            traces.append(trace)
            return results
        started = time.perf_counter()
        output = evaluate_retrieval(queries, search, ks, bootstrap_seed=self.config.seed)
        emit("retrieval_evaluation_timing", {"wall_seconds_including_statistics": time.perf_counter() - started})
        eligible = [q for q in queries if q.gold_doc_ids]
        output["retrieval_traces"] = {
            q.id: {"gold_doc_ids": sorted(set(q.gold_doc_ids)), **annotate_steps(
                trace, q.gold_doc_ids, output["retrieved_doc_ids"][q.id], ks)}
            for q, trace in zip(eligible, traces)}
        output["retrieval_steps_limit"] = self.config.retrieval_steps
        emit("retrieval_diagnostics", output["retrieval_traces"])
        return output

    def evaluate_base(self, queries: list[Any], method: str, ks: tuple[int, ...] | None = None) -> dict[str, Any]:
        """评测 BM25、Dense 或 Hybrid 基础检索。"""
        normalized = "hybrid" if method.lower() == "base" else method.lower()
        return self.evaluate_search(queries, lambda q, k, trace: self._search_base_once(q, k, normalized, trace=trace), ks)

    def report(self) -> dict[str, Any]:
        """导出物理设计、probe 成本、特征和预测结果供 artifact 审计。"""
        probe_cost = aggregate_costs([outcome.graph.cost for outcome in self.probes.values()])
        observed = sorted(outcome.gain for outcome in self.probes.values())
        nonnegative = sorted(max(value, 0.0) for value in observed)
        total = sum(nonnegative)
        gini = (sum((2 * index - len(nonnegative) - 1) * value
                    for index, value in enumerate(nonnegative, 1))
                / (len(nonnegative) * total)) if total > 0 else 0.0
        return {
            "config": asdict(self.config),
            "num_documents": len(self.bundle.documents) if self.bundle else 0,
            "num_train_queries": len(self.bundle.train) if self.bundle else 0,
            "num_regions": len(self.regions),
            "probe_regions": sorted(self.probes),
            "probe_cost": probe_cost.to_dict(),
            "probe_estimated_cost": sum(self.costs[key] for key in self.probes),
            "probe_estimated_cost_fraction": sum(self.costs[key] for key in self.probes) / self.full_graph_cost,
            "design_retrieval_usage": self.design_retrieval_usage,
            "observed_gain_distribution": {
                "minimum": observed[0] if observed else None,
                "median": statistics.median(observed) if observed else None,
                "maximum": observed[-1] if observed else None,
                "mean": sum(observed) / len(observed) if observed else None,
                "positive_fraction": sum(value > 0 for value in observed) / len(observed) if observed else None,
                "nonnegative_gain_gini": gini,
            },
            "non_graph_design_wall_seconds": self.design_timings,
            "full_graph_estimated_cost": self.full_graph_cost,
            "benefit_estimator": "paired_gain_zero_prior_shrinkage",
            "selection_gain_scope": "measured_regions_only",
            "graph_implementation": type(self.graph_builder).__name__,
            "graph_retriever": type(self.graph_retriever).__name__,
            "feature_schema": "region_local_v2_with_global_context",
            "probe_interactions": self.interaction_analysis,
            "conditional_selection": self.selection_reports,
            "regions": [{
                **self.features[region.id].to_dict(),
                "doc_ids": region.doc_ids,
                "estimated_cost": self.costs[region.id],
                "estimated_gain": self.estimated_gains.get(region.id) if region.id in self.probes else None,
                "gain_status": "measured" if region.id in self.probes else "unprobed",
                "probe_gain": self.probes[region.id].gain if region.id in self.probes else None,
                "probe_recall_gain": self.probes[region.id].recall_gain if region.id in self.probes else None,
                "probe_complete_gain": self.probes[region.id].complete_gain if region.id in self.probes else None,
                "probe_query_count": self.probes[region.id].query_count if region.id in self.probes else None,
                "probe_eligible_query_count": self.probes[region.id].eligible_query_count if region.id in self.probes else None,
                "probe_per_query": self.probes[region.id].per_query if region.id in self.probes else None,
                "probe_retrieval_usage": self.probes[region.id].retrieval_usage if region.id in self.probes else None,
            } for region in self.regions],
        }
