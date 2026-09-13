"""CPU regressions for selection isolation; graph/model calls are test doubles."""

import importlib.util
import sys
import types
import unittest
from unittest.mock import patch

# FAISS is only used by index construction, which these tests do not exercise.
if importlib.util.find_spec("faiss") is None:
    sys.modules.setdefault("faiss", types.ModuleType("faiss"))

from warp.eval.retrieval import evaluate_retrieval
from warp.graph.builder import RegionalGraph
from warp.graph.hipporag2 import HippoRAG2Config, HippoRAG2GraphBuilder
from warp.models import ConstructionCost, DatasetBundle, Document, Query, Region, RegionFeatures, SearchResult
from warp.pipeline import WARPConfig, WARPG
from warp.retrieval.hybrid import HybridRetriever, fuse_and_rerank
from scripts.audit_region_results import audit_artifact, compare_rows


def results(ids):
    return [SearchResult(doc_id, 1.0 / rank, "fixture", rank)
            for rank, doc_id in enumerate(ids, 1)]


class Base:
    def search(self, query, k):
        return results(["a0", "a1", "j0", "j1"][:k])


class GraphSearch:
    def __init__(self):
        self.calls = []

    def search(self, query, graph, k):
        self.calls.append(graph.region_id)
        return results({"r0": ["x0", "a0"], "r1": ["x1", "a1"]}[graph.region_id][:k])


class Reranker:
    def rerank(self, query, candidates, k):
        order = {key: index for index, key in enumerate(["x0", "x1", "a0", "a1", "j0", "j1"])}
        return sorted(candidates, key=lambda row: order[row.doc_id])[:k]


def model_fixture():
    model = WARPG(WARPConfig(selection_mode="independent", retrieval_k=2, routing_k=2, candidate_k=4),
                  None, GraphSearch(), Reranker(), Base())
    model.regions = [Region("r0", ["a0", "x0"]), Region("r1", ["a1", "x1"]),
                     Region("r2", ["j0", "j1"])]
    model.region_map = {region.id: region for region in model.regions}
    model.doc_region = {doc_id: region.id for region in model.regions for doc_id in region.doc_ids}
    model.bundle = DatasetBundle([Document(doc_id, doc_id) for doc_id in model.doc_region],
                                [Query("train", "query", ["a0", "x0"])])
    model.graphs = {key: RegionalGraph(key, model.region_map[key].doc_ids, ConstructionCost())
                    for key in ("r0", "r1")}
    model.features = {key: RegionFeatures(key, 2, 2, 10, 0.5, 1, 0, 1, 0, 0)
                      for key in model.region_map}
    model.costs = {key: 1.0 for key in model.region_map}
    model.estimated_gains = {"r0": 1.0, "r1": 0.2, "r2": 0.1}
    return model


class RegionalRetrievalTests(unittest.TestCase):
    def test_warp_ranks_by_frequency_times_gain_not_cost(self):
        from warp.advisor.selector import RegionSelector
        features = {
            "cheap": RegionFeatures("cheap", 1, 1, 1, 0, 0, 0, 0, 0, 0),
            "hot": RegionFeatures("hot", 1, 1, 10, 0, 0, 0, 0, 0, 0),
        }
        selector = RegionSelector(42)
        ranked = selector.rank("warp", features, {"cheap": 1.0, "hot": 100.0},
                               {"cheap": 0.1, "hot": 0.1})
        self.assertEqual(ranked[0], "hot")
        self.assertEqual(selector.select("warp", features, {"cheap": 1.0, "hot": 100.0},
                                         {"cheap": 0.1, "hot": 0.1}), ["hot", "cheap"])

    def test_controls_reuse_warp_region_count(self):
        model = model_fixture()
        model.estimated_gains["r2"] = 0.0
        warp = model.select("warp")
        self.assertEqual(warp, ["r0", "r1"])
        for method in ("random_region", "frequency_only", "gain_only"):
            self.assertEqual(len(model.select(method)), len(warp))

    def test_selector_and_evaluation_do_not_alias_methods(self):
        model = model_fixture()
        model.estimated_gains = {"r0": 1.0, "r1": 0.0, "r2": 0.0}
        warp = model.select("warp")
        random = model.select("random_region")
        self.assertEqual(warp, ["r0"])
        self.assertEqual(random, ["r1"])
        query = [Query("test", "query", ["a0", "x0"])]
        left = model.evaluate(query, warp, (2,))
        right = model.evaluate(query, random, (2,))
        self.assertEqual(left["complete_evidence@2"], 1.0)
        self.assertEqual(right["complete_evidence@2"], 0.0)
        self.assertEqual(model.graph_retriever.calls, ["r0", "r1"])
        audit = compare_rows({"method": "warp", "selected_regions": warp, **left},
                             {"method": "random_region", "selected_regions": random, **right}, 2)
        self.assertFalse(audit["same_region_set"])
        self.assertEqual(audit["result_comparison"]["same_ranked_topk_queries"], 0)
        self.assertEqual(audit["candidate_stages"]["queries_with_new_graph_gold_in_topk"], 1)
        self.assertEqual(audit["reference_stages"]["queries_with_new_graph_gold_in_topk"], 0)

    def test_equal_metrics_do_not_claim_identical_results_in_old_artifacts(self):
        metrics = {"q": {"evidence_recall@10": 0.5, "complete_evidence@10": 0.0}}
        left = {"method": "warp", "selected_regions": ["r0"], "per_query": metrics}
        right = {"method": "random_region", "selected_regions": ["r1"], "per_query": metrics}
        old = compare_rows(left, right)
        self.assertIsNone(old["result_comparison"])
        left["retrieved_doc_ids"] = {"q": ["gold", "x"]}
        right["retrieved_doc_ids"] = {"q": ["gold", "y"]}
        new = compare_rows(left, right)
        self.assertEqual(new["metrics"]["evidence_recall@10"]["equal_queries"], 1)
        self.assertEqual(new["result_comparison"]["same_ranked_topk_queries"], 0)

    def test_fusion_loss_is_distinguished_from_no_graph_gain(self):
        row = {"method": "warp", "selected_regions": ["r0"],
               "per_query": {"q": {"evidence_recall@2": 0.5, "complete_evidence@2": 0.0}},
               "retrieved_doc_ids": {"q": ["a0"]},
               "retrieval_traces": {"q": {
                   "gold_doc_ids": ["a0", "x0"], "routed_regions": ["r0"],
                   "base_candidate_doc_ids": ["a0"],
                   "graph_candidate_doc_ids": {"r0": ["x0"]},
                   "fused_candidate_doc_ids": ["a0"],
               }}}
        summary = compare_rows(row, row, 2)["candidate_stages"]
        self.assertEqual(summary["queries_with_new_graph_gold"], 1)
        self.assertEqual(summary["queries_with_new_graph_gold_after_fusion"], 0)
        self.assertEqual(summary["queries_with_complete_gold_in_union"], 1)
        self.assertEqual(summary["queries_with_complete_gold_after_fusion"], 0)

    def test_method_order_does_not_change_explicit_selection(self):
        model = model_fixture()
        query = [Query("test", "query", ["a0", "x0"])]
        before = model.evaluate(query, ["r1"], (2,))
        model.evaluate(query, ["r0"], (2,))
        after = model.evaluate(query, ["r1"], (2,))
        self.assertEqual(before, after)

    def test_artifact_comparison_separates_folds(self):
        metrics = {"q": {"evidence_recall@2": 0.5, "complete_evidence@2": 0.0}}
        rows = [{"method": method, "selected_regions": [f"r{fold}"], "per_query": metrics,
                 "design_seed": 42, "fold": fold, "budget_fraction": 0.2}
                for fold in (0, 1) for method in ("warp", "random_region")]
        self.assertEqual(len(audit_artifact({"quality_cost_curve": rows}, "warp", "random_region", 2)), 2)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            audit_artifact({"quality_cost_curve": rows + [rows[0]]}, "warp", "random_region", 2)

    def test_search_requires_explicit_deployment(self):
        model = model_fixture()
        with self.assertRaisesRegex(ValueError, "selected_regions"):
            model.search("query", 2)
        self.assertEqual(model.graph_retriever.calls, [])

    def test_empty_selection_excludes_all_cached_graphs(self):
        model = model_fixture()
        self.assertEqual([row.doc_id for row in model.search("query", 2, set())], ["a0", "a1"])
        self.assertEqual(model.graph_retriever.calls, [])

    def test_unbuilt_selection_is_not_silently_ignored(self):
        model = model_fixture()
        del model.graphs["r0"]
        with self.assertRaisesRegex(ValueError, "materializ"):
            model.search("query", 2, {"r0"})

    def test_unknown_selection_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown"):
            model_fixture().search("query", 2, {"typo"})

    def test_routing_diagnostics_use_actual_candidate_depth(self):
        class Ranked:
            def __init__(self, ids):
                self.ids = ids

            def search(self, query, k, doc_ids=None):
                return results(self.ids[:k])

        model = model_fixture()
        model.config.routing_k = 1
        model.config.candidate_k = 3
        model.base = HybridRetriever(Ranked(["a", "b", "c", "d", "x"]),
                                     Ranked(["e", "f", "g", "h", "x"]))
        model.doc_region = {key: "r1" for key in "abcdefgh"}
        model.doc_region["x"] = "r0"
        shallow = model.base.search("query", 1)[0].doc_id
        actual = model.base.search("query", 3)[0].doc_id
        self.assertNotEqual(shallow, actual)
        self.assertEqual(actual, "x")
        report = model.routing_diagnostics([Query("q", "query", ["x"])])
        self.assertEqual(report["complete_gold_region_recall"], 1.0)

    def test_refit_discards_old_graphs_before_new_design(self):
        model = model_fixture()
        model.full_graph = model.graphs["r0"]

        def check_reset(documents):
            self.assertEqual(model.graphs, {})
            self.assertIsNone(model.full_graph)
            raise RuntimeError("stop before external model construction")

        with patch.object(model.base, "fit", check_reset, create=True):
            with self.assertRaisesRegex(RuntimeError, "stop before external"):
                model.fit(model.bundle)

    def test_graph_cache_fingerprint_changes_with_region_contents(self):
        builder = HippoRAG2GraphBuilder(HippoRAG2Config())
        region = Region("r0", ["a"])
        first = builder._artifact_dir(region, [Document("a", "first")], "test")
        second = builder._artifact_dir(region, [Document("a", "second")], "test")
        self.assertNotEqual(first, second)

    def test_different_graph_rankings_can_be_erased_by_reranker(self):
        base = results(["a0", "a1", "x0", "x1"])
        left = fuse_and_rerank("q", [base, results(["x0", "x1"])], Reranker(),
                               k=2, candidate_k=4, source="left")
        right = fuse_and_rerank("q", [base, results(["x1", "x0"])], Reranker(),
                                k=2, candidate_k=4, source="right")
        self.assertEqual([row.doc_id for row in left], [row.doc_id for row in right])

    def test_duplicate_query_ids_cannot_overwrite_results(self):
        queries = [Query("q", "first", ["a0"]), Query("q", "second", ["x0"])]
        with self.assertRaisesRegex(ValueError, "unique"):
            evaluate_retrieval(queries, lambda query, k: results(["a0"]), (1,), bootstrap_samples=10)


if __name__ == "__main__":
    unittest.main()
