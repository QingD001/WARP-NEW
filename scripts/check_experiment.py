#!/usr/bin/env python3
"""Read-only experiment preflight; does not load model weights or call an API."""

import argparse
import importlib.metadata
import importlib.util
import json
from collections import Counter
from pathlib import Path

import yaml


def inspect_config(path):
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    blockers = []
    counts = {}
    records = {}
    for key in ("corpus", "queries"):
        source = Path(config["dataset"][key])
        if not source.is_file():
            blockers.append(f"Missing {key}: {source}")
            continue
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
        counts[key] = len(rows)
        records[key] = rows
        if not rows:
            blockers.append(f"Empty {key}: {source}")
        ids = [str(row.get("id")) for row in rows]
        if len(set(ids)) != len(ids) or any(row.get("id") is None for row in rows):
            blockers.append(f"Missing or duplicate IDs in {key}")
    if set(records) == {"corpus", "queries"}:
        known = {str(row["id"]) for row in records["corpus"]}
        queries = records["queries"]
        contents = [(str(row.get("title", "")) + "\n" + str(row.get("text", ""))).strip()
                    for row in records["corpus"]]
        counts["duplicate_document_contents"] = len(contents) - len(set(contents))
        counts["empty_document_contents"] = sum(not value for value in contents)
        counts["empty_gold_queries"] = sum(not row.get("gold_doc_ids") for row in queries)
        counts["queries_with_missing_gold"] = sum(bool(set(row.get("gold_doc_ids", [])) - known) for row in queries)
        counts["missing_answers"] = sum(row.get("answer") is None or row.get("answer") == [] for row in queries)
        counts["gold_count_distribution"] = dict(Counter(len(set(row.get("gold_doc_ids", []))) for row in queries))
        for key in ("duplicate_document_contents", "empty_document_contents", "empty_gold_queries", "queries_with_missing_gold"):
            if counts[key]:
                blockers.append(f"Invalid data: {key}={counts[key]}")
        folds = config["experiment"]["cross_fitting_folds"]
        if not 2 <= folds <= len(queries):
            blockers.append("Invalid cross-fitting fold count")
        if config["reader"]["enabled"] and counts["missing_answers"]:
            blockers.append("Reader enabled but answers are missing")
    return {"config": str(path), "counts": counts, "blockers": blockers}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=Path("configs/paper"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    modules = ["numpy", "torch", "faiss", "hipporag", "sentence_transformers",
               "igraph", "leidenalg", "tiktoken"]
    available = {name: importlib.util.find_spec(name) is not None for name in modules}
    cuda = False
    if available["torch"]:
        import torch
        cuda = torch.cuda.is_available()
    packages = {}
    for name in ("warp-g", "hipporag"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    configurations = [inspect_config(path) for path in sorted(args.config_dir.glob("*.yaml"))]
    blockers = [f"Missing dependency: {name}" for name, present in available.items() if not present]
    if not cuda:
        blockers.append("CUDA is unavailable; the formal runner requires CUDA")
    if not packages["warp-g"]:
        blockers.append("warp-g is not installed; runner requires its package metadata")
    if packages["hipporag"] != "2.0.0a4":
        blockers.append("The pinned adapter requires hipporag 2.0.0a4")
    if not configurations:
        blockers.append("No experiment configurations found")
    output = {"modules": available, "packages": packages, "cuda_available": cuda,
              "global_blockers": blockers, "configurations": configurations,
              "ready_for_runtime_smoke_test": not blockers and all(not row["blockers"] for row in configurations),
              "not_verified": ["model weights and revisions", "API connectivity and credentials",
                               "GPU memory capacity", "real graph indexing and retrieval", "official baseline installations"]}
    rendered = json.dumps(output, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    raise SystemExit(0 if output["ready_for_runtime_smoke_test"] else 1)


if __name__ == "__main__":
    main()
