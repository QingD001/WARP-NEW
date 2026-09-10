"""Experiment control-flow and accounting regressions, without model/API calls."""

import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from test_regional_retrieval import model_fixture, results
from scripts.prepare_benchmark import supporting_titles
from scripts.prepare_hipporag2 import _answers
from warp.baselines.global_graph import _keyword_knn
from scripts.export_paper_results import _read, _write_rows
from warp.data.base import load_crossfit_bundles
from warp.eval.reader import evaluate_hipporag2_reader
from warp.graph.hipporag2 import HippoRAG2GraphRetriever
from warp.models import ConstructionCost, Query
from warp.utils import write_json
from warp import run


class IntegrityTests(unittest.TestCase):
    def test_answer_aliases_are_not_dropped(self):
        self.assertEqual(_answers({"answer": "Rockland County", "answer_aliases": ["Rockland County, New York"]}),
                         ["Rockland County", "Rockland County, New York"])

    def test_bounded_keyword_neighbors_match_exhaustive_overlap(self):
        words = {"a": {"x", "y"}, "b": {"x", "z"}, "c": {"x", "y"}, "d": {"z"}}
        expected = {key: sorted([(other, len(value & other_value)) for other, other_value in words.items()
                                if other != key and value & other_value], key=lambda item: (-item[1], item[0]))[:2]
                    for key, value in words.items()}
        self.assertEqual(_keyword_knn(words, 2), expected)

    def test_explicit_evidence_ids_take_priority_over_titles(self):
        self.assertEqual(supporting_titles({"gold_doc_ids": ["d1"], "supporting_facts": [["Title", 0]]}), ["d1"])

    def test_crossfit_rejects_duplicate_ids_and_empty_folds(self):
        with tempfile.TemporaryDirectory() as directory:
            corpus, queries = Path(directory) / "corpus.jsonl", Path(directory) / "queries.jsonl"
            corpus.write_text('{"id":"d","text":"document"}\n')
            queries.write_text('{"id":"q","query":"q","gold_doc_ids":["d"]}\n' * 2)
            config = {"corpus": str(corpus), "queries": str(queries)}
            with self.assertRaisesRegex(ValueError, "unique"):
                load_crossfit_bundles(config, 2, 42)
            with self.assertRaisesRegex(ValueError, "one query"):
                load_crossfit_bundles(config, 3, 42)

    def test_probe_cost_counts_reused_graph_only_once_and_charges_gain_only(self):
        model = model_fixture()
        model.graphs["r0"].cost = ConstructionCost(input_tokens=10)
        model.probes = {"r0": types.SimpleNamespace(graph=model.graphs["r0"])}
        model.design_retrieval_usage = {"logical_input_tokens": 4}
        for method in ("warp", "gain_only"):
            total, extra = run._probe_design_costs(model, method, ["r0"], {})
            self.assertEqual(total.input_tokens, 14)
            self.assertEqual(extra.input_tokens + model.graphs["r0"].cost.input_tokens, 14)
        self.assertEqual(run._probe_design_costs(model, "random_region", [], {})[0].selection_cost, 0)

    def test_reader_summary_weights_unequal_folds(self):
        rows = [{"method": "warp", "num_queries": n, "answer_em": value, "answer_f1": value}
                for n, value in ((1, 1.0), (3, 0.0))]
        self.assertEqual(run._reader_summary(rows)[0]["answer_em_mean"], 0.25)

    def test_significance_without_warp_is_empty(self):
        self.assertEqual(run._significance([{"method": "random_region", "budget_fraction": 0.2}], 10, 42), [])

    def test_atomic_write_preserves_previous_result_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            write_json(path, {"valid": 1})
            with self.assertRaises(ValueError):
                write_json(path, {"invalid": float("nan")})
            self.assertEqual(json.loads(path.read_text()), {"valid": 1})
            self.assertEqual(len(list(Path(directory).iterdir())), 1)

    def test_backend_fallback_is_counted_separately(self):
        model = model_fixture()
        graph = model.graphs["r0"]
        solution = types.SimpleNamespace(docs=["a0"], doc_metadata=[{"source_id": "a0"}],
                                         doc_scores=[1.0], graph_seeds=[])
        graph.backend = types.SimpleNamespace(retrieve=lambda **kwargs: [solution])
        retriever = HippoRAG2GraphRetriever()
        retriever.search("q", graph)
        solution.graph_seeds = [("a", "relation", "b")]
        retriever.search("q", graph)
        self.assertEqual(retriever.stats()["dense_fallback_calls"], 1)
        self.assertEqual(retriever.stats()["graph_seeded_calls"], 1)

    def test_reader_honors_topk_and_rejects_missing_answers(self):
        module = types.ModuleType("hipporag.utils.misc_utils")
        module.QuerySolution = types.SimpleNamespace
        observed = []
        config = types.SimpleNamespace(qa_top_k=5)
        tracker = types.SimpleNamespace(phase="idle", get=lambda phase: {})

        def qa(solutions):
            observed.append(config.qa_top_k)
            return [], [], []

        graph = types.SimpleNamespace(backend=types.SimpleNamespace(global_config=config, qa=qa,
                                                                   _warp_usage_tracker=tracker))
        with patch.dict("sys.modules", {"hipporag.utils.misc_utils": module}):
            with self.assertRaisesRegex(RuntimeError, "number of answers"):
                evaluate_hipporag2_reader([Query("q", "q", ["a0"], "answer")],
                    lambda query, k: results(["a0"]), model_fixture().bundle.documents, graph, 10)
        self.assertEqual(observed, [10])
        self.assertEqual(config.qa_top_k, 5)
        self.assertEqual(tracker.phase, "idle")

    def test_complete_runner_with_stubbed_backends_export_and_resume(self):
        # Exercise all method/budget/reader/partition branches with actual
        # regional retrieval, metrics and aggregation. Only expensive backends
        # and physical-design training are replaced.
        config = run.load_config("configs/paper/hotpotqa.yaml")
        config["experiment"]["cross_fitting_folds"] = 2
        config["experiment"].pop("budgets", None)
        config["experiment"]["randomization_samples"] = 20
        config["reader"]["methods"] = list(config["experiment"]["methods"]) + ["bm25", "dense", "hybrid", "hipporag2", "full_graph"]

        def build(config, bundle, seed, partition_mode=None):
            model = model_fixture()
            model.config.candidate_k = 50
            model.config.retrieval_k = 10
            model.config.retrieval_steps = config["warp"]["retrieval_steps"]
            model.config.multistep_max_steps = int(config["warp"].get("multistep_max_steps", 0))
            if "retrieval_ks" in config["warp"]:
                model.config.retrieval_ks = tuple(config["warp"]["retrieval_ks"])
            model.config.selection_mode = config["warp"]["selection_mode"]
            model.bundle = bundle
            model.base.bm25 = model.base
            model.base.dense = model.base
            model.report = lambda: {}
            model.design_timings = {"base_index_seconds": 0, "feature_seconds": 1}
            for graph in model.graphs.values():
                graph.cost = ConstructionCost(input_tokens=100, estimated_usd=0.1)
            model.full_graph = model.graphs["r0"]
            model.probes = {"r0": types.SimpleNamespace(graph=model.graphs["r0"])}
            model.graph_builder = types.SimpleNamespace(build=lambda region, documents: types.SimpleNamespace(
                region_id=region.id, doc_ids=region.doc_ids, cost=ConstructionCost(input_tokens=10)))
            model.graph_retriever.stats = lambda: {"calls": len(model.graph_retriever.calls)}
            model.graph_retriever.delta = lambda before: {"calls": len(model.graph_retriever.calls) - before["calls"]}
            return model, model.graph_retriever

        class Factory:
            def __init__(self, documents, base, builder, retriever, reranker, candidate_k, **kwargs):
                self.base = base
                self.doc_costs = {}
                self.ket_core_fraction = kwargs.get("ket_core_fraction", 0.8)
                self.g2_core_fraction = kwargs.get("g2_core_fraction", 0.8)

            def build(self, method, *args, **kwargs):
                return types.SimpleNamespace(search=self.base.search, graph=None, lightweight=None,
                                             cost=ConstructionCost())

        def reader(queries, search, documents, graph, top_k, *, retrieved_doc_ids=None):
            self.assertIsNone(search)
            for query in queries:
                self.assertIn(query.id, retrieved_doc_ids)
            return {"answer_em": 0.5, "answer_f1": 0.5, "num_queries": len(queries)}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus, queries = root / "corpus.jsonl", root / "queries.jsonl"
            corpus.write_text("\n".join(json.dumps({"id": doc.id, "text": doc.text})
                                           for doc in model_fixture().bundle.documents))
            queries.write_text("\n".join(json.dumps({"id": f"q{i}", "query": f"query {i}",
                "gold_doc_ids": ["a0", "x0"], "answer": "answer"}) for i in range(4)))
            config["dataset"].update(corpus=str(corpus), queries=str(queries))
            metadata = {"data_sha256": {"corpus": "fixture", "queries": "fixture"}, "package_versions": {}}
            with patch.object(run, "reproducibility_metadata", return_value=metadata), \
                 patch.object(run, "_build_model", side_effect=build) as built, \
                 patch.object(run, "GlobalBaselineFactory", Factory), \
                 patch.object(run, "evaluate_hipporag2_reader", side_effect=reader):
                output = run.run_experiment(config, root / "checkpoints")
                self.assertEqual(len(output["quality_cost_curve"]), 14)
                self.assertTrue(all("evidence_recall@2" in row and "complete_evidence@3" in row
                                    for row in output["quality_cost_curve"]))
                self.assertTrue(all(row.get("multistep", {}).get("protocol") == "ircot"
                                    for row in output["quality_cost_curve"]))
                self.assertTrue((root / "checkpoints" / "multistep" / "fold-0" / "warp.jsonl").exists())
                self.assertEqual(len(output["partition_ablations"]), 6)
                self.assertTrue(all(row["num_queries"] == 4 for row in output["baselines"]))
                self.assertIn("hybrid", {row["reference"] for row in output["paired_significance"]})
                self.assertEqual(built.call_count, 8)
                snapshot = Path(output["raw_artifacts"]["input_snapshot"])
                self.assertTrue((snapshot / "dataset.json").exists())
                events = [json.loads(line) for line in Path(output["raw_artifacts"]["fold_events"][0]).read_text().splitlines()]
                self.assertIn("reader_result", {event["event"] for event in events})
                self.assertIn("retrieval_query", {event["event"] for event in events})
                self.assertIn("multistep_retrieval", {event["event"] for event in events})
                self.assertIn("conditional_selection", {event["event"] for event in events})
                path = root / "result.json"
                write_json(path, output)
                loaded = _read(path)
                _write_rows(root / "curve.csv", loaded["quality_cost_curve"])
                self.assertTrue((root / "curve.csv").stat().st_size)
                resumed = run.run_experiment(config, root / "checkpoints")
                self.assertEqual(built.call_count, 8)
                self.assertEqual(resumed["quality_cost_curve"], output["quality_cost_curve"])
                changed = copy.deepcopy(config)
                changed["warp"]["routing_k"] += 1
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    run.run_experiment(changed, root / "checkpoints")


if __name__ == "__main__":
    unittest.main()
