import json
from pathlib import Path
import tempfile
import types
import unittest

from test_regional_retrieval import model_fixture, results
from warp.advisor.features import RegionFeatureExtractor
from warp.advisor.probe import RegionProber
from warp.audit import audit_scope, emit
from warp.graph.hipporag2 import HippoRAG2Config, HippoRAG2GraphBuilder
from warp.models import Query, Region, RegionFeatures


class LocalFeatureTests(unittest.TestCase):
    def test_local_failure_is_not_inherited_from_another_region(self):
        model = model_fixture()
        model.base.dense = types.SimpleNamespace(vector=lambda doc_id: [1.0, 0.0])
        query = Query("q", "q", ["a0", "x1"])
        features, _, _ = RegionFeatureExtractor(routing_k=2, candidate_k=4).extract(
            model.regions, model.bundle.documents, [query], model.base,
            types.SimpleNamespace(query_edges={}))
        local = features["r0"]
        self.assertEqual(local.base_recall, 1.0)
        self.assertEqual(local.failure_rate, 0.0)
        self.assertEqual(local.multi_doc_rate, 0.0)
        self.assertEqual(local.global_base_recall, 0.5)
        self.assertEqual(local.global_failure_rate, 1.0)
        self.assertEqual(local.global_multi_doc_rate, 1.0)
        self.assertEqual(local.cross_region_gold_rate, 1.0)
        self.assertEqual(features["r1"].base_recall, 0.0)

    def test_probe_query_cap_and_local_gain_records(self):
        model = model_fixture()
        model.graph_retriever.stats = lambda: {"calls": len(model.graph_retriever.calls)}
        model.graph_retriever.delta = lambda before: {"calls": len(model.graph_retriever.calls) - before["calls"]}
        builder = types.SimpleNamespace(build=lambda region, documents: model.graphs[region.id])
        prober = RegionProber(builder, model.graph_retriever, model.reranker, .2, 2, 4,
                              "complete_evidence", 42, max_queries=2)
        queries = [Query(f"q{i}", "q", ["a0", "x0"]) for i in range(5)]
        outcome = prober.run([model.region_map["r0"]], model.bundle.documents, queries,
                             {"r0": [q.id for q in queries]}, {q.id: model.base.search(q.text, 4) for q in queries})["r0"]
        self.assertEqual(outcome.query_count, 2)
        self.assertEqual(outcome.eligible_query_count, 5)
        self.assertEqual(outcome.retrieval_usage["calls"], 2)
        self.assertEqual(outcome.per_query[0]["new_gold"], ["x0"])

    def test_probe_budget_never_silently_overruns(self):
        regions = [Region(f"r{i}", [f"d{i}"]) for i in range(10)]
        features = {r.id: RegionFeatures(r.id, 1, 10, 1, 0, 1, 0, 0, 0, 0) for r in regions}
        costs = {r.id: (100 if i < 3 else 1) for i, r in enumerate(regions)}
        prober = RegionProber(None, None, None, .2, 2, 4, "complete_evidence", 42)
        selected = prober.select_probe_regions(regions, features, costs=costs, budget=6)
        self.assertEqual(len(selected), 2)
        self.assertLessEqual(sum(costs[r.id] for r in selected), 6)
        self.assertEqual(prober.select_probe_regions(regions, features, costs=costs, budget=0), [])
        self.assertEqual(len(prober.select_probe_regions(regions[:1], features, costs=costs, budget=100)), 1)

    def test_measured_gain_shrinkage_preserves_losses_and_unknowns(self):
        from warp.advisor.estimator import estimate_benefits
        outcomes = {
            "small": types.SimpleNamespace(gain=.5, query_count=1),
            "large": types.SimpleNamespace(gain=.5, query_count=64),
            "loss": types.SimpleNamespace(gain=-.5, query_count=64),
        }
        gains = estimate_benefits(dict.fromkeys([*outcomes, "unknown"]), outcomes, 16)
        self.assertLess(gains["small"], gains["large"])
        self.assertAlmostEqual(gains["large"], .4)
        self.assertAlmostEqual(gains["loss"], -.4)
        self.assertEqual(gains["unknown"], 0)
        self.assertEqual(estimate_benefits({"r": None}, {}), {"r": 0})

    def test_empty_probe_report_is_valid(self):
        model = model_fixture()
        report = model.report()
        self.assertEqual(report["probe_regions"], [])
        self.assertIsNone(report["observed_gain_distribution"]["mean"])
        self.assertTrue(all(r["gain_status"] == "unprobed" for r in report["regions"]))

    def test_raw_llm_record_preserves_content_not_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            llm = types.SimpleNamespace(infer=lambda *args, **kwargs: ("raw answer", {"prompt_tokens": 2, "completion_tokens": 1}, False))
            builder = HippoRAG2GraphBuilder(HippoRAG2Config())
            tracker = builder._wrap_llm(llm)
            tracker.phase = "construction"
            tracker.audit_context = {"path": str(path), "region_id": "r0"}
            llm.infer([{"role": "user", "content": "raw prompt"}], api_key="DO_NOT_STORE")
            rendered = path.read_text()
            self.assertNotIn("DO_NOT_STORE", rendered)
            self.assertEqual(json.loads(rendered)["payload"]["response"], "raw answer")
            self.assertEqual(tracker.get("construction")["physical_input_tokens"], 2)
            captured = {}
            thinking_llm = types.SimpleNamespace(infer=lambda *args, **kwargs: captured.update(kwargs) or ("ok", {"prompt_tokens": 1, "completion_tokens": 0}, False))
            HippoRAG2GraphBuilder(HippoRAG2Config(disable_llm_thinking=True))._wrap_llm(thinking_llm)
            thinking_llm.infer([])
            self.assertEqual(captured["extra_body"]["thinking"], {"type": "disabled"})
            self.assertFalse(captured["extra_body"]["enable_thinking"])
            with audit_scope(path=str(path), method="warp"):
                emit("test", {"value": 1})
            self.assertEqual(len(path.read_text().splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
