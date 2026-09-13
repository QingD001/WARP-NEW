import types
import unittest
from dataclasses import replace
from unittest.mock import patch

from test_regional_retrieval import model_fixture, results
from warp.advisor.conditional import select_conditional
from warp.advisor.objective import utility
from warp.eval.diagnostics import annotate_steps
from warp.models import Document, Query
from warp.retrieval.multistep import retrieve_steps
from warp.run import _probe_design_costs


class PreserveOrder:
    def rerank(self, query, candidates, k):
        return candidates[:k]


class ConditionalAndMultistepTests(unittest.TestCase):
    def test_partial_evidence_has_signal_and_losses_are_negative(self):
        gold = ["a", "b"]
        self.assertEqual(utility(results(["a"]), gold, "complete_evidence"), 0)
        self.assertEqual(utility(results(["a"]), gold, "mixed", .5), .25)
        self.assertLess(utility(results(["a"]), gold) - utility(results(gold), gold), 0)
        self.assertEqual(utility(results(["a"]), gold, "mixed", 0), .5)
        self.assertEqual(utility(results(["a"]), gold, "mixed", 1), 0)

    def pair_model(self):
        model = model_fixture()
        model.config = replace(model.config, selection_mode="conditional", benefit_objective="complete_evidence")
        model.probes = {"r0": None, "r1": None}
        model.estimated_gains = {key: 0 for key in model.costs}
        model.bundle.train = [Query("design", "design only", ["x0", "x1"])]
        model.bundle.test = [Query("test", "must never be searched", ["a0"])]
        calls = []
        def search(query, k, selected):
            self.assertEqual(query, "design only")
            calls.append(frozenset(selected))
            return results(["x0" if "r0" in selected else "a0", "x1" if "r1" in selected else "a1"])
        model.search = search
        model.graph_retriever.stats = lambda: {"logical_input_tokens": len(calls) * 10}
        model.graph_retriever.delta = lambda before: {"logical_input_tokens": len(calls) * 10 - before["logical_input_tokens"]}
        return model, calls

    def test_pair_escapes_zero_singleton_and_selection_cache_avoids_reruns(self):
        model, calls = self.pair_model()
        self.assertEqual(model.select("warp"), ["r0", "r1"])
        self.assertEqual(len(calls), 4)  # Base, two singletons, their pair
        self.assertEqual(model.select("warp"), ["r0", "r1"])
        self.assertEqual(len(calls), 4)
        self.assertTrue(all("r2" not in call for call in calls))

    def test_conditional_controls_match_actual_warp_count_not_independent_score(self):
        model, _ = self.pair_model()
        warp = model.select("warp")
        self.assertEqual(warp, ["r0", "r1"])
        independent = model.selector.select("warp", model.features, model.costs, model.estimated_gains)
        self.assertEqual(independent, [])
        for method in ("random_region", "frequency_only", "gain_only"):
            self.assertEqual(len(model.select(method)), len(warp))

    def test_evaluation_cap_and_negative_marginals(self):
        model, calls = self.pair_model()
        model.config = replace(model.config, conditional_max_evaluations=2)
        chosen, report = select_conditional(model, 2)
        self.assertEqual(chosen, [])
        self.assertEqual(len(calls), 2)
        self.assertTrue(report["evaluation_limit_reached"])
        model.config = replace(model.config, conditional_max_evaluations=16)
        model.search = lambda query, k, selected: results(["x0", "x1"] if not selected else ["a0", "a1"])
        self.assertEqual(select_conditional(model, 2)[0], [])

    def test_conditional_calls_charged_only_to_warp_design(self):
        model = model_fixture()
        model.design_retrieval_usage = {"logical_input_tokens": 10}
        model.selection_reports = {"full_pipeline": {"retrieval_usage": {"logical_input_tokens": 90}}}
        self.assertEqual(_probe_design_costs(model, "warp", [], {})[0].input_tokens, 100)
        self.assertEqual(_probe_design_costs(model, "gain_only", [], {})[0].input_tokens, 10)
        self.assertEqual(_probe_design_costs(model, "random_region", [], {})[0].input_tokens, 0)

    def test_feedback_discovers_second_hop_and_reranks_original_question(self):
        calls, rerank_queries = [], []
        class Ranker:
            def rerank(self, query, candidates, k):
                rerank_queries.append(query)
                order = {"bridge": 0, "answer": 1, "noise": 2}
                return sorted(candidates, key=lambda r: order[r.doc_id])[:k]
        def search(query, k, trace):
            calls.append(query)
            return results(["answer"]) if "BridgeEntity" in query else results(["bridge", "noise"])
        trace = {}
        found = retrieve_steps("question", 2, search,
            [Document("bridge", "BridgeEntity points elsewhere"), Document("answer", "answer"), Document("noise", "noise")],
            Ranker(), candidate_k=3, steps=2, feedback_docs=1, trace=trace)
        self.assertEqual([r.doc_id for r in found], ["bridge", "answer"])
        self.assertEqual(len(calls), 2)
        self.assertEqual(set(rerank_queries), {"question"})
        diagnostic = annotate_steps(trace, ["bridge", "answer"], [r.doc_id for r in found], (2,))
        self.assertEqual(diagnostic["evidence_diagnostics"][1]["metrics_by_k"]["2"]["new_gold"], ["answer"])
        self.assertEqual(diagnostic["evidence_diagnostics"][0]["metrics_by_k"]["2"]["complete_evidence"], 0)

    def test_no_feedback_stops_and_step_one_does_not_expand(self):
        calls = []
        def search(q, k, trace):
            calls.append(q)
            return results(["d"])
        trace = {}
        retrieve_steps("q", 1, search, [Document("d", "context")], PreserveOrder(), candidate_k=1, steps=5, trace=trace)
        self.assertEqual(len(calls), 2)
        self.assertEqual(trace["stop_reason"], "no_new_feedback")
        calls.clear()
        retrieve_steps("q", 1, search, [], PreserveOrder(), candidate_k=1, steps=1)
        self.assertEqual(calls, ["q"])

    def test_multistep_reroutes_and_respects_explicit_deployment(self):
        model = model_fixture()
        model.config = replace(model.config, retrieval_steps=2, feedback_docs=1)
        model.base.search = lambda q, k: results(["a1", "a0"] if "Evidence:" in q else ["a0", "j0"])
        trace = {}
        model.search("q", 2, {"r0", "r1"}, trace=trace)
        self.assertNotIn("r1", trace["steps"][0]["routed_regions"])
        self.assertIn("r1", trace["steps"][1]["routed_regions"])
        model.graph_retriever.calls.clear()
        model.search("q", 2, {"r0"})
        self.assertNotIn("r1", model.graph_retriever.calls)

    def test_base_and_empty_deployment_share_multistep_protocol(self):
        model = model_fixture()
        model.config = replace(model.config, retrieval_steps=2)
        model.base.bm25 = model.base
        model.base.dense = model.base
        queries = model.bundle.train
        base = model.evaluate_base(queries, "hybrid", (2,))
        empty = model.evaluate(queries, [], (2,))
        self.assertEqual(base["retrieved_doc_ids"], empty["retrieved_doc_ids"])
        self.assertEqual(base["retrieval_traces"]["train"]["steps_executed"], 2)
        self.assertEqual(empty["retrieval_traces"]["train"]["steps_executed"], 2)

    def test_fit_probes_actual_multistep_path_and_records_mixed_labels(self):
        model = model_fixture()
        model.config = replace(model.config, retrieval_steps=2, selection_mode="conditional",
                               probe_fraction=1)
        regions = model.regions[:2]
        features = {r.id: model.features[r.id] for r in regions}
        graphs = dict(model.graphs)
        model.graph_builder = types.SimpleNamespace(estimate_cost=lambda r, docs: 1,
                                                    build=lambda r, docs: graphs[r.id])
        model.base.fit = lambda docs: None
        model.graph_retriever.stats = lambda: {"calls": len(model.graph_retriever.calls)}
        model.graph_retriever.delta = lambda before: {"calls": len(model.graph_retriever.calls) - before["calls"]}
        # Keep all documents assigned; only two regions are eligible for probing.
        all_regions = model.regions
        all_features = model.features
        all_features["r2"].query_freq = 0
        routed = {r.id: ["train"] for r in all_regions}
        base_rows = {"train": model.base.search("query", 4)}
        with patch("warp.pipeline.CoaccessGraphBuilder.build", return_value=types.SimpleNamespace()), \
             patch("warp.pipeline.RegionPartitioner.partition", return_value=all_regions), \
             patch("warp.pipeline.RegionFeatureExtractor.extract", return_value=(all_features, routed, base_rows)):
            model.fit(model.bundle)
        self.assertEqual(set(model.probes), {"r0", "r1"})
        self.assertGreaterEqual(model.design_retrieval_usage["calls"], 4)
        for probe in model.probes.values():
            self.assertAlmostEqual(probe.gain, .5 * probe.recall_gain + .5 * probe.complete_gain)
            self.assertIn("utility_gain", probe.per_query[0])
        model.select("warp")
        self.assertIn("full_pipeline", model.report()["conditional_selection"])

    def test_config_rejects_unbounded_or_invalid_settings(self):
        model = model_fixture()
        for kwargs in ({"conditional_max_evaluations": 0}, {"retrieval_steps": 0},
                       {"complete_weight": 2}, {"benefit_objective": "unknown"},
                       {"multistep_max_steps": -1}):
            with self.assertRaises(ValueError):
                replace(model.config, **kwargs)
