"""从 YAML 运行完整论文实验并输出自描述 JSON artifact。"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import platform
import statistics
import shutil
import sys
from dataclasses import asdict, replace
from uuid import uuid4
from warp.audit import audit_scope, emit
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
import torch

from warp.baselines import GlobalBaselineFactory
from warp.data import load_crossfit_bundles
from warp.eval.construction_cost import aggregate_costs, attach_token_efficiency, token_efficiency
from warp.eval.cutoffs import metric_names
from warp.eval.multistep import run_multistep_retrieval
from warp.eval.reader import evaluate_hipporag2_reader
from warp.eval.statistics import paired_bootstrap_interval, paired_randomization_pvalue
from warp.graph import HippoRAG2Config, HippoRAG2GraphBuilder, HippoRAG2GraphRetriever
from warp.graph.hipporag2 import SUPPORTED_API_VERSION, UPSTREAM_COMMIT, UPSTREAM_REPOSITORY
from warp.pipeline import WARPConfig, WARPG
from warp.models import ConstructionCost
from warp.retrieval import BM25Retriever, CrossEncoderReranker, DenseRetriever, HybridRetriever, fuse_and_rerank
from warp.utils import write_json


METRICS = metric_names()
TOKEN_EFFICIENCY_KEYS = (
    "total_tokens_excluding_design", "total_tokens_including_design",
    "token_efficiency_excluding_design", "token_efficiency_including_design",
)
REGIONAL_METHODS = {"warp", "random_region", "frequency_only", "gain_only", "cost_only"}
GLOBAL_METHODS = {"ket_rag", "g2cons"}


def _audited(call, **labels):
    with audit_scope(**labels):
        value = call()
        emit("evaluation_result", value)
        return value


def _apply_run_overrides(config: dict[str, Any], *, skip_multistep: bool = False) -> dict[str, Any]:
    if not skip_multistep:
        return config
    updated = copy.deepcopy(config)
    updated.setdefault("warp", {})["multistep_max_steps"] = 0
    return updated


def _trace_search(search_once: Any) -> Any:
    def wrapped(query: str, k: int) -> tuple[Any, dict[str, Any]]:
        trace: dict[str, Any] = {}
        return search_once(query, k, trace), trace
    return wrapped


def _plain_search(search: Any) -> Any:
    def wrapped(query: str, k: int) -> tuple[Any, dict[str, Any]]:
        return search(query, k), {}
    return wrapped


def _with_usage_trace(search_trace: Any, retriever: Any) -> Any:
    def wrapped(query: str, k: int) -> tuple[Any, dict[str, Any]]:
        before = retriever.stats() if retriever is not None else {}
        results, trace = search_trace(query, k)
        payload = dict(trace or {})
        if retriever is not None:
            payload["tokens"] = retriever.delta(before)
        return results, payload
    return wrapped


def _ircot_generate(model: WARPG) -> Any:
    def generate(prompt: str) -> str:
        text = model.generate_next_query(prompt)
        generate.last_usage = dict(getattr(model, "last_generation_usage", {}) or {})
        return text
    generate.last_usage = {}
    return generate


def _attach_multistep(
    metrics: dict[str, Any], *, method: str, search_trace: Any, model: WARPG,
    queries: Any, log_dir: Path | None, documents: Any,
) -> dict[str, Any]:
    """IRCoT 对照：不改写主路径 retrieved_doc_ids，问答仍用第一遍检索。"""
    steps = int(model.config.multistep_max_steps)
    if steps <= 0:
        return metrics
    log_path = None if log_dir is None else log_dir / f"{method}.jsonl"
    doc_map = {doc.id: doc for doc in documents} if documents is not None else None
    summary = run_multistep_retrieval(
        queries,
        _with_usage_trace(search_trace, getattr(model, "graph_retriever", None)),
        _ircot_generate(model),
        max_steps=steps,
        retrieval_k=int(model.config.retrieval_k),
        ks=tuple(model.config.retrieval_ks),
        log_path=log_path,
        method=method,
        documents=doc_map,
        snippet_chars=int(model.config.feedback_chars),
    )
    metrics["multistep"] = {key: value for key, value in summary.items() if key != "ranked_results"}
    return metrics


def load_config(path: str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict) or not value:
        raise ValueError(f"Experiment config must be a non-empty mapping: {path}")
    return value


def reproducibility_metadata(config: dict[str, Any]) -> dict[str, Any]:
    packages = [
        "warp-g", "hipporag", "numpy", "torch", "sentence-transformers",
        "igraph", "leidenalg", "faiss-cpu", "datasets", "tiktoken", "PyYAML",
    ]
    if not torch.cuda.is_available():
        raise RuntimeError("The formal experiment requires a CUDA device")
    dataset_paths = {key: Path(value) for key, value in config["dataset"].items()
                     if key in {"corpus", "queries"}}
    data_hashes = {}
    for key, path in dataset_paths.items():
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        data_hashes[key] = digest.hexdigest()
    device = torch.cuda.current_device()
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "configuration": config,
        "package_versions": {name: importlib.metadata.version(name) for name in packages},
        "data_sha256": data_hashes,
        "cuda": {
            "runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device_name": torch.cuda.get_device_name(device),
            "device_capability": list(torch.cuda.get_device_capability(device)),
        },
        "hipporag2_upstream": {
            "repository": UPSTREAM_REPOSITORY,
            "commit": UPSTREAM_COMMIT,
            "validated_api_version": SUPPORTED_API_VERSION,
        },
    }


def _build_model(
    config: dict[str, Any], bundle: Any, design_seed: int, partition_mode: str | None = None,
) -> tuple[WARPG, HippoRAG2GraphRetriever]:
    graph_config = dict(config["graph"])
    graph_config["dataset"] = config["dataset"]["name"]
    graph_config["seed"] = design_seed
    builder = HippoRAG2GraphBuilder(HippoRAG2Config(**graph_config))
    graph_retriever = HippoRAG2GraphRetriever()
    dense = DenseRetriever(
        encoder=builder.passage_encoder(), model_name=f"hipporag:{builder.config.embedding_model_name}",
    )
    base = HybridRetriever(BM25Retriever(), dense)
    reranker = CrossEncoderReranker(bundle.documents, **config["reranker"])
    warp_config = replace(WARPConfig(**config["warp"]), seed=design_seed)
    if partition_mode is not None:
        warp_config = replace(warp_config, partition_mode=partition_mode)
    return WARPG(warp_config, builder, graph_retriever, reranker, base).fit(bundle), graph_retriever


def _summarize(
    rows: list[dict[str, Any]], keys: tuple[str, ...], metric_names: tuple[str, ...] = METRICS,
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    output: list[dict[str, Any]] = []
    for group_key, values in sorted(groups.items()):
        item = {key: value for key, value in zip(keys, group_key)}
        item["trials"] = len(values)
        for metric in metric_names:
            samples = [float(row[metric]) for row in values]
            item[f"{metric}_mean"] = statistics.fmean(samples)
            item[f"{metric}_std"] = statistics.stdev(samples) if len(samples) > 1 else 0.0
        if "actual_cost_fraction" in values[0]:
            samples = [float(row["actual_cost_fraction"]) for row in values]
            item["actual_cost_fraction_mean"] = statistics.fmean(samples)
            item["actual_cost_fraction_std"] = statistics.stdev(samples) if len(samples) > 1 else 0.0
        for key in TOKEN_EFFICIENCY_KEYS:
            if key not in values[0] or values[0][key] is None:
                continue
            samples = [row[key] for row in values if row.get(key) is not None]
            if not samples:
                continue
            item[f"{key}_mean"] = statistics.fmean(samples)
            item[f"{key}_std"] = statistics.stdev(samples) if len(samples) > 1 else 0.0
        output.append(item)
    return output


def _crossfit_summary(
    rows: list[dict[str, Any]], keys: tuple[str, ...], seed: int,
    metric_names: tuple[str, ...] = METRICS,
) -> list[dict[str, Any]]:
    """Merge disjoint held-out folds and compute final query-level statistics."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    output: list[dict[str, Any]] = []
    for group_key, values in sorted(groups.items()):
        per_query = {
            query_id: metrics
            for row in values for query_id, metrics in row["per_query"].items()
        }
        if len(per_query) != sum(len(row["per_query"]) for row in values):
            raise ValueError("A query appeared in more than one held-out fold")
        item: dict[str, Any] = {
            **{key: value for key, value in zip(keys, group_key)},
            "folds": len(values), "num_queries": len(per_query),
        }
        for metric in metric_names:
            samples = [float(metrics[metric]) for metrics in per_query.values()]
            lower, upper = paired_bootstrap_interval(samples, seed=seed)
            item[metric] = statistics.fmean(samples)
            item[f"{metric}_ci95"] = [lower, upper]
        if "actual_cost_fraction" in values[0]:
            costs = [float(row["actual_cost_fraction"]) for row in values]
            item["actual_cost_fraction_mean"] = statistics.fmean(costs)
            item["actual_cost_fraction_std"] = statistics.stdev(costs) if len(costs) > 1 else 0.0
        if "total_tokens_excluding_design" in values[0]:
            excluding = sum(int(row.get("total_tokens_excluding_design") or 0) for row in values)
            including = sum(int(row.get("total_tokens_including_design") or 0) for row in values)
            item["total_tokens_excluding_design"] = excluding
            item["total_tokens_including_design"] = including
            evidence = item.get("complete_evidence@10")
            if evidence is not None:
                item["token_efficiency_excluding_design"] = token_efficiency(float(evidence), excluding)
                item["token_efficiency_including_design"] = token_efficiency(float(evidence), including)
        output.append(item)
    return output


def _significance(rows: list[dict[str, Any]], samples: int, seed: int) -> list[dict[str, Any]]:
    if not any(row["method"] == "warp" for row in rows):
        return []
    output: list[dict[str, Any]] = []
    references = sorted({str(row["method"]) for row in rows if row["method"] != "warp"})
    candidate_rows = [row for row in rows if row["method"] == "warp"]
    candidate_per_query = {
        query_id: values
        for row in candidate_rows for query_id, values in row["per_query"].items()
    }
    for reference_name in references:
        reference_rows = [row for row in rows if row["method"] == reference_name]
        reference_per_query = {
            query_id: values
            for row in reference_rows for query_id, values in row["per_query"].items()
        }
        query_ids = sorted(candidate_per_query)
        if set(query_ids) != set(reference_per_query):
            raise ValueError("Cross-fit significance requires identical held-out query coverage")
        for metric in METRICS:
            candidate = [candidate_per_query[query_id][metric] for query_id in query_ids]
            baseline = [reference_per_query[query_id][metric] for query_id in query_ids]
            output.append({
                "design_seed": seed,
                "cross_fitting_folds": len(candidate_rows),
                "num_queries": len(query_ids),
                "selection": "full_pipeline",
                "candidate": "warp",
                "reference": reference_name,
                "metric": metric,
                "mean_difference": statistics.fmean(left - right for left, right in zip(candidate, baseline)),
                "paired_randomization_pvalue": paired_randomization_pvalue(
                    candidate, baseline, samples=samples, seed=seed,
                ),
            })
    for metric in METRICS:
        group = [row for row in output if row["metric"] == metric]
        ordered = sorted(group, key=lambda row: float(row["paired_randomization_pvalue"]))
        running = 0.0
        for index, row in enumerate(ordered):
            adjusted = min(1.0, (len(ordered) - index) * float(row["paired_randomization_pvalue"]))
            running = max(running, adjusted)
            row["holm_adjusted_pvalue"] = running
    return output


def _quality_cost_auc(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Integrate each method's mean quality over its measured deployment-cost curve."""
    methods = sorted({str(row["method"]) for row in rows})
    output: list[dict[str, Any]] = []
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        grouping = "budget_fraction" if any("budget_fraction" in row for row in method_rows) else "selection"
        budget_points: list[tuple[float, dict[str, float]]] = []
        keys = sorted({row.get(grouping, "full_pipeline") for row in method_rows}, key=lambda value: str(value))
        for key in keys:
            trials = [row for row in method_rows if row.get(grouping, "full_pipeline") == key]
            cost = statistics.fmean(float(row["actual_cost_fraction"]) for row in trials)
            metrics = {metric: statistics.fmean(float(row[metric]) for row in trials) for metric in METRICS}
            budget_points.append((cost, metrics))
        by_cost: dict[float, list[dict[str, float]]] = {}
        for cost, metrics in budget_points:
            by_cost.setdefault(cost, []).append(metrics)
        points = [
            (cost, {metric: statistics.fmean(item[metric] for item in values) for metric in METRICS})
            for cost, values in sorted(by_cost.items())
        ]
        item: dict[str, Any] = {
            "method": method,
            "cost_axis": "actual_cost_fraction",
            "minimum_cost_fraction": points[0][0],
            "maximum_cost_fraction": points[-1][0],
            "points": [{"actual_cost_fraction": cost, **metrics} for cost, metrics in points],
        }
        for metric in METRICS:
            if len(points) < 2:
                item[f"{metric}_auc"] = None
            else:
                item[f"{metric}_auc"] = sum(
                    (right_cost - left_cost) * (left_metrics[metric] + right_metrics[metric]) / 2.0
                    for (left_cost, left_metrics), (right_cost, right_metrics) in zip(points, points[1:])
                )
        output.append(item)
    return output


def _probe_design_costs(model: WARPG, method: str, selected: list[str],
                        graph_config: dict[str, Any], selection_key: str = "full_pipeline") -> tuple[ConstructionCost, ConstructionCost]:
    """Return total design cost and incremental cost excluding deployed probe graphs."""
    if method not in {"warp", "gain_only"}:
        return ConstructionCost(), ConstructionCost()
    usage = dict(model.design_retrieval_usage)
    if method == "warp":
        extra = model.selection_reports.get(selection_key, {}).get("retrieval_usage", {})
        for key in ("logical_input_tokens", "logical_output_tokens", "wall_seconds"):
            usage[key] = usage.get(key, 0) + extra.get(key, 0)
    input_tokens = int(usage.get("logical_input_tokens", 0))
    output_tokens = int(usage.get("logical_output_tokens", 0))
    retrieval_cost = ConstructionCost(
        input_tokens=input_tokens, output_tokens=output_tokens,
        wall_seconds=float(usage.get("wall_seconds", 0)),
        estimated_usd=(input_tokens * graph_config.get("input_usd_per_million_tokens", 0.15)
                       + output_tokens * graph_config.get("output_usd_per_million_tokens", 0.60)) / 1_000_000,
    )
    all_cost = aggregate_costs([outcome.graph.cost for outcome in model.probes.values()] + [retrieval_cost])
    extra_cost = aggregate_costs([outcome.graph.cost for key, outcome in model.probes.items()
                                 if key not in set(selected)] + [retrieval_cost])
    return all_cost, extra_cost


def _reader_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Weight folds by evaluated queries, not by fold count."""
    output = _summarize(rows, ("method",), ("answer_em", "answer_f1"))
    for summary in output:
        trials = [row for row in rows if row["method"] == summary["method"]]
        count = sum(row["num_queries"] for row in trials)
        summary["num_queries"] = count
        for metric in ("answer_em", "answer_f1"):
            summary[f"{metric}_mean"] = sum(row[metric] * row["num_queries"] for row in trials) / count
    return output


def _run_fold_experiment(
    config: dict[str, Any], bundle: Any, fold: int, checkpoint_dir: Path | None = None,
) -> dict[str, Any]:
    required = ("dataset", "warp", "graph", "reranker", "experiment", "reader")
    missing = [name for name in required if not isinstance(config.get(name), dict)]
    if missing:
        raise ValueError(f"Missing experiment configuration sections: {', '.join(missing)}")
    experiment = config["experiment"]
    for required_key in (
        "methods", "seed", "randomization_samples",
        "partition_ablations",
    ):
        if required_key not in experiment:
            raise ValueError(f"experiment.{required_key} is required")
    if "budgets" in experiment:
        print(json.dumps({
            "warning": "experiment.budgets is ignored; each method runs its full native pipeline",
        }, ensure_ascii=False), flush=True)
    methods = [str(value).lower() for value in experiment["methods"]]
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("Experiment methods must be non-empty and unique")
    unknown = set(methods) - REGIONAL_METHODS - GLOBAL_METHODS
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")
    design_seed = int(experiment["seed"])

    if not bundle.test:
        raise ValueError("A paper experiment requires a non-empty test query split")
    rows: list[dict[str, Any]] = []
    baseline_trials: list[dict[str, Any]] = []
    design_reports: list[dict[str, Any]] = []
    reader_trials: list[dict[str, Any]] = []
    partition_ablation_rows: list[dict[str, Any]] = []

    for design_seed in [design_seed]:
        with audit_scope(stage="design"):
            model, graph_retriever = _build_model(config, bundle, design_seed)
        factory = GlobalBaselineFactory(
            bundle.documents, model.base, model.graph_builder, model.graph_retriever,
            model.reranker, model.config.candidate_k,
            ket_core_fraction=float(experiment.get("ket_core_fraction", 0.8)),
            g2_core_fraction=float(experiment.get("g2_core_fraction", 0.8)),
        ) if (set(methods) | (set(config["reader"]["methods"]) if config["reader"]["enabled"] else set())) & GLOBAL_METHODS else None
        design_reports.append({
            "design_seed": design_seed, "fold": fold,
            "train_routing": model.routing_diagnostics(bundle.train),
            "test_routing": model.routing_diagnostics(bundle.test),
            **model.report(),
        })
        emit("physical_design", design_reports[-1])
        reader_config = config["reader"]
        reader_methods = {str(value).lower() for value in reader_config["methods"]} if reader_config["enabled"] else set()
        reader_k = int(reader_config["top_k"])
        evaluation_ks = tuple(sorted(set(model.config.retrieval_ks) | ({reader_k} if reader_methods else set())))
        reader_done = set()
        pending_readers = {}
        multistep_dir = (
            Path(checkpoint_dir) / "multistep" / f"fold-{fold}"
            if checkpoint_dir is not None else None
        )

        def maybe_reader(method, metrics):
            if method not in reader_methods or method in reader_done:
                return
            reader_backend = next(iter(model.graphs.values()), None) or model.full_graph
            if reader_backend is None:
                pending_readers[method] = metrics
                return
            with audit_scope(method=method, selection="full_pipeline", stage="reader"):
                # QA only needs the shared LLM/config, not a corpus-wide graph.
                scored = evaluate_hipporag2_reader(bundle.test, None, bundle.documents, reader_backend, reader_k,
                                                   retrieved_doc_ids=metrics["retrieved_doc_ids"])
                reader_trials.append({"design_seed": design_seed, "fold": fold, "method": method,
                                      "selection": "full_pipeline", **scored})
                emit("reader_result", reader_trials[-1])
            reader_done.add(method)

        for method in ("bm25", "dense", "hybrid"):
            before = graph_retriever.stats()
            metrics = _audited(lambda: model.evaluate_base(bundle.test, method, evaluation_ks), method=method, stage="test")
            online = graph_retriever.delta(before)
            maybe_reader(method, metrics)
            metrics = _attach_multistep(
                metrics, method=method,
                search_trace=_trace_search(
                    lambda query, k, trace, method=method: model._search_base_once(query, k, method, trace=trace)
                ),
                model=model, queries=bundle.test, log_dir=multistep_dir, documents=bundle.documents,
            )
            baseline_trials.append({
                "design_seed": design_seed, "fold": fold, "method": method,
                "online_retrieval_cost": online, **metrics,
            })

        seed_rows: list[dict[str, Any]] = []
        for method in methods:
            before = graph_retriever.stats()
            ircot_search = None
            if method in REGIONAL_METHODS:
                selected = model.select(method)
                selected_set = set(selected)
                before = graph_retriever.stats()  # design calls are not test retrieval
                metrics = _audited(lambda: model.evaluate(bundle.test, selected, evaluation_ks),
                                   method=method, selection="full_pipeline", stage="test")
                deployment_cost = aggregate_costs([model.graphs[key].cost for key in selected])
                selected_ids: dict[str, Any] = {"selected_regions": selected}
                estimated_cost = sum(model.costs[key] for key in selected)
                ircot_search = _trace_search(
                    lambda query, k, trace, selected_set=selected_set: model._search_once(
                        query, k, selected_set, trace=trace
                    )
                )
            else:
                if factory is None:
                    raise RuntimeError("Global baseline factory was not initialized")
                baseline = factory.build(method)
                metrics = _audited(lambda: model.evaluate_search(
                    bundle.test, lambda text, depth, trace: baseline.search(text, depth), evaluation_ks),
                    method=method, selection="full_pipeline", stage="test")
                deployment_cost = baseline.cost
                selected_ids = {
                    "selected_documents": baseline.graph.doc_ids if baseline.graph is not None else [],
                    "core_fraction": (
                        factory.ket_core_fraction if method == "ket_rag" else factory.g2_core_fraction
                    ),
                }
                estimated_cost = (
                    sum(factory.doc_costs[key] for key in selected_ids["selected_documents"])
                    + (baseline.lightweight.cost.selection_cost if baseline.lightweight is not None else 0.0)
                )
                ircot_search = _plain_search(baseline.search)
            design_search_cost, incremental_design_cost = _probe_design_costs(
                model, method, selected_ids.get("selected_regions", []), config["graph"],
            )
            first_run_cost = aggregate_costs([deployment_cost, incremental_design_cost])
            if method in {"warp", "gain_only"}:
                method_design_wall = sum(model.design_timings.values()) - model.design_timings["base_index_seconds"]
            else:
                method_design_wall = 0.0
            selection_report = model.selection_reports.get("full_pipeline", {}) if method == "warp" else {}
            method_design_wall += max(0.0, selection_report.get("wall_seconds", 0) - selection_report.get("retrieval_usage", {}).get("wall_seconds", 0))
            online = graph_retriever.delta(before)
            maybe_reader(method, metrics)
            metrics = _attach_multistep(
                metrics, method=method, search_trace=ircot_search, model=model,
                queries=bundle.test, log_dir=multistep_dir, documents=bundle.documents,
            )
            seed_rows.append({
                "method": method,
                "design_seed": design_seed,
                "fold": fold,
                "selection": "full_pipeline",
                **selected_ids,
                "conditional_design": selection_report,
                "retrieval_steps_limit": model.config.retrieval_steps,
                "selected_estimated_graph_cost": estimated_cost,
                "deployment_cost": deployment_cost.to_dict(),
                "design_search_cost": design_search_cost.to_dict(),
                "incremental_design_cost": incremental_design_cost.to_dict(),
                "first_run_cost_including_probe": first_run_cost.to_dict(),
                "method_specific_design_wall_seconds": method_design_wall,
                "first_run_wall_seconds": first_run_cost.wall_seconds + method_design_wall,
                "online_retrieval_cost": online,
                **metrics,
            })
            emit("method_cost", {key: value for key, value in seed_rows[-1].items()
                                 if key not in {"per_query", "retrieval_traces", "retrieved_doc_ids",
                                                "ranked_results", "multistep"}})

        before = graph_retriever.stats()
        full_metrics = _audited(lambda: model.evaluate_full_graph(bundle.test, evaluation_ks), method="full_graph", stage="test")
        full_online = graph_retriever.delta(before)
        for pending_method, pending_metrics in list(pending_readers.items()):
            maybe_reader(pending_method, pending_metrics)
        pending_readers.clear()
        maybe_reader("full_graph", full_metrics)
        full_metrics = _attach_multistep(
            full_metrics, method="full_graph",
            search_trace=_trace_search(lambda query, k, trace: model._search_full_graph_once(query, k, trace=trace)),
            model=model, queries=bundle.test, log_dir=multistep_dir, documents=bundle.documents,
        )
        before = graph_retriever.stats()
        graph_metrics = _audited(lambda: model.evaluate_full_graph_only(bundle.test, evaluation_ks), method="hipporag2", stage="test")
        graph_online = graph_retriever.delta(before)
        maybe_reader("hipporag2", graph_metrics)
        graph = model.materialize_full_graph()
        graph_metrics = _attach_multistep(
            graph_metrics, method="hipporag2",
            search_trace=_trace_search(lambda query, k, trace, graph=graph: fuse_and_rerank(
                query, [model.graph_retriever.search(query, graph, model.config.candidate_k)],
                model.reranker, k=k, candidate_k=model.config.candidate_k,
                source="hipporag2_reranked", trace=trace,
            )),
            model=model, queries=bundle.test, log_dir=multistep_dir, documents=bundle.documents,
        )
        full_cost = model.full_graph.cost
        baseline_trials.extend([
            {"design_seed": design_seed, "fold": fold, "method": "hipporag2", "online_retrieval_cost": graph_online,
             "actual_construction_cost": full_cost.to_dict(), **graph_metrics},
            {"design_seed": design_seed, "fold": fold, "method": "full_graph", "online_retrieval_cost": full_online,
             "actual_construction_cost": full_cost.to_dict(), **full_metrics},
        ])
        denominator_tokens = full_cost.selection_cost
        denominator_usd = full_cost.estimated_usd
        if denominator_tokens <= 0:
            raise ValueError("Full graph construction token cost must be positive")
        for row in seed_rows:
            deployed = row["deployment_cost"]
            first_run = row["first_run_cost_including_probe"]
            hybrid = next(trial for trial in baseline_trials if trial["method"] == "hybrid")
            delta_ce = row["complete_evidence@10"] - hybrid["complete_evidence@10"]
            net_completed = delta_ce * row["num_queries"]
            deployed_tokens = deployed["input_tokens"] + deployed["output_tokens"] + deployed["embedding_tokens"]
            first_tokens = first_run["input_tokens"] + first_run["output_tokens"] + first_run["embedding_tokens"]
            row["incremental_efficiency"] = {
                "reference": "hybrid", "delta_complete_evidence@10": delta_ce,
                "net_completed_queries": net_completed,
                "net_completed_queries_per_million_deployment_tokens": net_completed * 1e6 / deployed_tokens if deployed_tokens else None,
                "net_completed_queries_per_million_first_run_tokens": net_completed * 1e6 / first_tokens if first_tokens else None,
                "cost_scope": "logical input + output + estimated embedding tokens; online retrieval and QA reported separately",
            }
            row["actual_cost_fraction"] = (
                deployed["input_tokens"] + deployed["output_tokens"] + deployed["embedding_tokens"]
            ) / denominator_tokens
            row["first_run_cost_fraction"] = (
                first_run["input_tokens"] + first_run["output_tokens"] + first_run["embedding_tokens"]
            ) / denominator_tokens
            row["actual_usd_fraction"] = deployed["estimated_usd"] / denominator_usd if denominator_usd > 0 else None
            attach_token_efficiency(row)
        rows.extend(seed_rows)
        for trial in baseline_trials:
            attach_token_efficiency(trial)

        # Reader-only methods still retrieve exactly once, and save the
        # corresponding evidence before QA instead of invoking a second search.
        for method in sorted(reader_methods - reader_done):
            if method in REGIONAL_METHODS:
                selected = model.select(method)
                metrics = _audited(lambda: model.evaluate(bundle.test, selected, evaluation_ks),
                                   method=method, selection="full_pipeline", stage="reader_retrieval")
            elif method in GLOBAL_METHODS:
                baseline = factory.build(method)
                metrics = _audited(lambda: model.evaluate_search(bundle.test, lambda text, depth, trace: baseline.search(text, depth), evaluation_ks),
                                   method=method, selection="full_pipeline", stage="reader_retrieval")
            else:
                raise ValueError(f"Unknown reader method: {method}")
            maybe_reader(method, metrics)

    design_reports[-1]["conditional_selection"] = model.selection_reports

    # Avoid constructing the next encoder/reranker while the main experiment's
    # model, full graph and bound search closures still retain GPU resources.
    model = factory = baseline = search = graph = graph_retriever = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    ablation_config = experiment["partition_ablations"]
    if not isinstance(ablation_config, dict):
        raise ValueError("experiment.partition_ablations is required")
    if "budget" in ablation_config:
        print(json.dumps({
            "warning": "partition_ablations.budget is ignored; WARP runs its full pipeline",
        }, ensure_ascii=False), flush=True)
    ablation_seed = int(ablation_config["seed"])
    for mode in [str(value) for value in ablation_config["modes"]]:
        model, graph_retriever = _build_model(config, bundle, ablation_seed, mode)
        selected = model.select("warp")
        before = graph_retriever.stats()
        metrics = model.evaluate(bundle.test, selected)
        ablation_design_cost, ablation_extra_cost = _probe_design_costs(model, "warp", selected, config["graph"])
        ablation = {
            "design_search_cost": ablation_design_cost.to_dict(),
            "incremental_design_cost": ablation_extra_cost.to_dict(),
            "conditional_design": model.selection_reports.get("full_pipeline", {}),
            "partition_mode": mode,
            "design_seed": ablation_seed,
            "fold": fold,
            "selection": "full_pipeline",
            "selected_regions": selected,
            "deployment_cost": aggregate_costs([model.graphs[key].cost for key in selected]).to_dict(),
            "online_retrieval_cost": graph_retriever.delta(before),
            "routing": model.routing_diagnostics(bundle.test),
            "num_regions": len(model.regions),
            **metrics,
        }
        attach_token_efficiency(ablation)
        partition_ablation_rows.append(ablation)
        model = graph_retriever = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "design_trials": design_reports,
        "baseline_trials": baseline_trials,
        "quality_cost_curve": rows,
        "partition_ablations": partition_ablation_rows,
        "reader_evaluation_trials": reader_trials,
    }


def run_experiment(
    config: dict[str, Any],
    checkpoint_dir: Path | None = None,
    max_folds: int | None = None,
) -> dict[str, Any]:
    """Run deterministic cross-fitting and merge all held-out query results."""
    if checkpoint_dir is None:
        checkpoint_dir = Path(config["graph"]["artifact_root"]) / "experiment_records"
    experiment = config.get("experiment")
    if not isinstance(experiment, dict) or "cross_fitting_folds" not in experiment:
        raise ValueError("experiment.cross_fitting_folds is required")
    folds = int(experiment["cross_fitting_folds"])
    seed = int(experiment["seed"])
    metadata = reproducibility_metadata(config)
    bundles = load_crossfit_bundles(config["dataset"], folds, seed)
    if max_folds is not None:
        if int(max_folds) < 1:
            raise ValueError("max_folds must be a positive integer")
        bundles = bundles[: int(max_folds)]
    source_hash = hashlib.sha256()
    for source in sorted(Path(__file__).parent.rglob("*.py")):
        source_hash.update(str(source.relative_to(Path(__file__).parent)).encode())
        source_hash.update(source.read_bytes())
    metadata["warp_source_sha256"] = source_hash.hexdigest()
    signature = hashlib.sha256(json.dumps({
        "config": config, "data": metadata["data_sha256"],
        "packages": metadata["package_versions"], "source": metadata["warp_source_sha256"],
    }, sort_keys=True).encode()).hexdigest()
    if checkpoint_dir is not None:
        snapshot = checkpoint_dir / "inputs" / signature
        if not (snapshot / "dataset.json").exists():
            snapshot.mkdir(parents=True, exist_ok=True)
            project_root = Path(__file__).resolve().parent.parent
            for directory in ("warp", "scripts", "configs"):
                for source in (project_root / directory).rglob("*"):
                    if source.is_file() and source.suffix in {".py", ".yaml"}:
                        target = snapshot / "source" / source.relative_to(project_root)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source, target)
            for key in ("corpus", "queries"):
                source = Path(config["dataset"][key])
                shutil.copy2(source, snapshot / f"{key}{source.suffix}")
            manifest = Path(config["dataset"]["corpus"]).parent / "split_manifest.json"
            if manifest.exists():
                shutil.copy2(manifest, snapshot / manifest.name)
                for key, filename in json.loads(manifest.read_text()).get("source_files", {}).items():
                    source = Path(filename)
                    if source.is_file():
                        shutil.copy2(source, snapshot / f"raw_{key}{source.suffix}")
            write_json(snapshot / "dataset.json", {"signature": signature, "configuration": config,
                       "metadata": metadata, "documents": [asdict(doc) for doc in bundles[0].documents],
                       "queries": [asdict(query) for bundle in bundles for query in bundle.test],
                       "fold_test_ids": [[query.id for query in bundle.test] for bundle in bundles]})
    fold_results: list[dict[str, Any]] = []
    for fold, bundle in enumerate(bundles):
        checkpoint = checkpoint_dir / f"fold-{fold}.json" if checkpoint_dir is not None else None
        if checkpoint is not None and checkpoint.exists():
            saved = json.loads(checkpoint.read_text(encoding="utf-8"))
            if saved.get("signature") != signature or saved.get("fold") != fold:
                raise ValueError(f"Checkpoint configuration/data/code mismatch: {checkpoint}; use a new checkpoint directory")
            result = saved["result"]
        else:
            audit_path = checkpoint_dir / "raw" / f"fold-{fold}-{uuid4().hex}.jsonl" if checkpoint_dir else None
            with audit_scope(path=str(audit_path) if audit_path else None, fold=fold,
                             signature=signature, stage="experiment"):
                result = _run_fold_experiment(config, bundle, fold, checkpoint_dir=checkpoint_dir)
            result["raw_audit_path"] = str(audit_path) if audit_path else None
            if checkpoint is not None:
                write_json(checkpoint, {"signature": signature, "fold": fold, "result": result})
        fold_results.append(result)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    baseline_trials = [row for result in fold_results for row in result["baseline_trials"]]
    quality_rows = [row for result in fold_results for row in result["quality_cost_curve"]]
    design_trials = [row for result in fold_results for row in result["design_trials"]]
    partition_rows = [row for result in fold_results for row in result["partition_ablations"]]
    reader_trials = [row for result in fold_results for row in result["reader_evaluation_trials"]]
    held_out_ids = [query.id for bundle in bundles for query in bundle.test]
    if len(held_out_ids) != len(set(held_out_ids)):
        raise ValueError("Cross-fitting must evaluate each query exactly once")
    metadata["cross_fitting"] = {
        "folds": folds,
        "executed_folds": len(bundles),
        "max_folds": None if max_folds is None else int(max_folds),
        "executed_fold_indices": list(range(len(bundles))),
        "design_queries_per_fold": [len(bundle.train) for bundle in bundles],
        "test_queries_per_fold": [len(bundle.test) for bundle in bundles],
        "total_unique_held_out_queries": len(held_out_ids),
        "assignment": "sha256(seed:query_id) ordering followed by round-robin folds",
    }
    return {
        "run_metadata": metadata,
        "raw_artifacts": {"input_snapshot": str(checkpoint_dir / "inputs" / signature),
                          "fold_events": [result.get("raw_audit_path") for result in fold_results]},
        "design_trials": design_trials,
        "baseline_trials": baseline_trials,
        "baselines": _crossfit_summary(baseline_trials, ("method",), seed),
        "quality_cost_curve": quality_rows,
        "quality_cost_summary": _crossfit_summary(
            quality_rows, ("method",), seed,
        ),
        "quality_cost_auc": _quality_cost_auc(quality_rows),
        "paired_significance": _significance(
            quality_rows + baseline_trials,
            int(experiment["randomization_samples"]), seed,
        ),
        "partition_ablations": partition_rows,
        "partition_ablation_summary": _crossfit_summary(
            partition_rows, ("partition_mode",), seed,
        ),
        "reader_evaluation_trials": reader_trials,
        "reader_evaluation": _reader_summary(reader_trials) if reader_trials else [],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run WARP-G selective GraphRAG experiments")
    parser.add_argument("--config", required=True, help="YAML experiment configuration")
    parser.add_argument("--output", default="outputs/results.json", help="Result JSON path")
    parser.add_argument("--checkpoint-dir", type=Path, help="Resume completed folds; default: <output>.folds")
    parser.add_argument("--max-folds", type=int, default=None,
                        help="Run only the first N cross-fitting folds; the yaml fold split is unchanged")
    parser.add_argument("--skip-multistep", action="store_true",
                        help="Disable the IRCoT对照; QA still uses the first-pass retrieval cache")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    checkpoint_dir = args.checkpoint_dir or Path(args.output + ".folds")
    config = _apply_run_overrides(load_config(args.config), skip_multistep=args.skip_multistep)
    result = run_experiment(config, checkpoint_dir, max_folds=args.max_folds)
    write_json(args.output, result)
    print(json.dumps({"output": args.output, "rows": len(result["quality_cost_curve"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
