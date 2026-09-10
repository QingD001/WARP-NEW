import inspect
import json
import tempfile
import unittest
from pathlib import Path

from warp.eval.cutoffs import RETRIEVAL_KS, metric_names
from warp.eval.multistep import merge_ranked, run_multistep_retrieval
from warp.eval.reader import evaluate_hipporag2_reader
from warp.eval.retrieval import evaluate_retrieval, query_metrics, ranked_from_payload
from warp.models import Document, Query, SearchResult
from warp.pipeline import WARPConfig
from warp.run import _apply_run_overrides, build_parser


class RetrievalCutoffTest(unittest.TestCase):
    def test_query_metrics_include_all_cutoffs(self) -> None:
        query = Query("q1", "who", ["d1", "d2"])
        results = [
            SearchResult("d1", 1.0, "base", 1),
            SearchResult("d3", 0.5, "base", 2),
            SearchResult("d2", 0.4, "base", 3),
        ]
        values = query_metrics(results, query, RETRIEVAL_KS)
        self.assertEqual(set(values), set(metric_names()))
        self.assertEqual(values["evidence_recall@2"], 0.5)
        self.assertEqual(values["complete_evidence@2"], 0.0)
        self.assertEqual(values["complete_evidence@3"], 1.0)
        self.assertEqual(values["complete_evidence@5"], 1.0)
        self.assertEqual(values["complete_evidence@10"], 1.0)


class RetrievalThenReaderWorkflowTest(unittest.TestCase):
    def test_evaluate_retrieval_caches_ranked_results_in_one_pass(self) -> None:
        calls: list[tuple[str, int]] = []

        def search(query: str, k: int) -> list[SearchResult]:
            calls.append((query, k))
            return [SearchResult("d1", 1.0, "base", 1), SearchResult("d2", 0.5, "base", 2)]

        metrics = evaluate_retrieval(
            [Query("q1", "who", ["d1"])], search, ks=(2, 3, 5, 10), bootstrap_samples=10,
        )
        self.assertEqual(calls, [("who", 10)])
        self.assertIn("ranked_results", metrics)
        self.assertEqual(metrics["ranked_results"]["q1"][0]["doc_id"], "d1")
        self.assertEqual(metrics["retrieved_doc_ids"]["q1"], ["d1", "d2"])
        self.assertEqual(metrics["evidence_recall@2"], 1.0)
        restored = ranked_from_payload(metrics["ranked_results"]["q1"])
        self.assertEqual(restored[0].doc_id, "d1")
        self.assertEqual(len(calls), 1)

    def test_reader_still_accepts_cached_doc_ids(self) -> None:
        params = inspect.signature(evaluate_hipporag2_reader)
        self.assertIn("retrieved_doc_ids", params.parameters)


class IRCoTMultistepTest(unittest.TestCase):
    def test_merge_ranked_keeps_higher_score(self) -> None:
        merged = merge_ranked(
            [SearchResult("d1", 0.2, "base", 1)],
            [SearchResult("d1", 0.9, "graph", 1), SearchResult("d2", 0.5, "base", 2)],
        )
        self.assertEqual([row.doc_id for row in merged], ["d1", "d2"])
        self.assertEqual(merged[0].score, 0.9)
        self.assertEqual(merged[0].source, "graph")
        self.assertEqual([row.rank for row in merged], [1, 2])

    def test_step_logs_reconstruct_trajectory_and_do_not_use_gold_to_stop(self) -> None:
        queries = [Query("q1", "who wrote it", ["gold"])]
        calls: list[str] = []
        generations: list[str] = ["missing evidence", "END"]

        def search(query: str, k: int):
            calls.append(query)
            doc_id = "gold" if "missing" in query else "noise"
            return [SearchResult(doc_id, 1.0, "base", 1)], {"routed_regions": ["r0"]}

        def generate(prompt: str) -> str:
            self.assertIn("who wrote it", prompt)
            return generations.pop(0)

        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "warp.jsonl"
            summary = run_multistep_retrieval(
                queries, search, generate, max_steps=3, retrieval_k=2, ks=RETRIEVAL_KS,
                log_path=log_path, method="warp",
                documents={"noise": Document("noise", "irrelevant"), "gold": Document("gold", "the answer")},
            )
            rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
        steps = [row for row in rows if row.get("event") == "step"]
        complete = [row for row in rows if row.get("event") == "query_complete"]
        self.assertEqual(len(steps), 2)
        self.assertEqual(calls, ["who wrote it", "missing evidence"])
        self.assertEqual(steps[0]["stop_reason"], "continue")
        self.assertEqual(steps[0]["step_gold_hit"], [])
        self.assertEqual(steps[1]["stop_reason"], "model_end")
        self.assertEqual(steps[1]["gold_hit"], ["gold"])
        self.assertEqual(steps[1]["metrics"]["complete_evidence@2"], 1.0)
        self.assertEqual(steps[1]["routed_regions"], ["r0"])
        self.assertEqual(steps[1]["next_query"], "missing evidence")
        self.assertEqual(complete[0]["steps"], 2)
        self.assertEqual(summary["complete_evidence@2"], 1.0)
        self.assertEqual(summary["protocol"], "ircot")
        reconstructed = [row["step_query"] for row in steps] + [complete[0]["stop_reason"]]
        self.assertEqual(reconstructed, ["who wrote it", "missing evidence", "model_end"])

    def test_empty_generation_stops_without_gold_shortcut(self) -> None:
        prompts: list[str] = []

        def search(query: str, k: int):
            return [SearchResult("gold", 1.0, "base", 1)], {}

        def generate(prompt: str) -> str:
            prompts.append(prompt)
            return ""

        summary = run_multistep_retrieval(
            [Query("q1", "who", ["gold"])], search, generate,
            max_steps=3, retrieval_k=1, ks=(2, 3, 5, 10), method="hybrid",
        )
        self.assertEqual(len(prompts), 1)
        self.assertEqual(summary["num_queries"], 1)
        self.assertEqual(summary["complete_evidence@2"], 1.0)

    def test_skip_multistep_override_sets_zero_steps(self) -> None:
        updated = _apply_run_overrides(
            {"warp": {"retrieval_k": 10}, "reader": {"enabled": True}},
            skip_multistep=True,
        )
        self.assertEqual(updated["warp"]["multistep_max_steps"], 0)

    def test_parser_accepts_max_folds(self) -> None:
        args = build_parser().parse_args(
            ["--config", "c.yaml", "--output", "o.json", "--max-folds", "1"]
        )
        self.assertEqual(args.max_folds, 1)

    def test_config_rejects_negative_multistep_steps(self) -> None:
        with self.assertRaises(ValueError):
            WARPConfig(multistep_max_steps=-1)


if __name__ == "__main__":
    unittest.main()
