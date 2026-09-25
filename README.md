# WARP-G

WARP-G is a workload-aware regional graph materialization system for GraphRAG.
There is no separate training loop. Physical design (partition, probe, region
selection) and held-out evaluation both run inside `python -m warp.run`.

This repository is intended as anonymous supplementary code. It does **not**
bundle processed corpora, HippoRAG indexes, LLM caches, or paper result JSON.

Paper datasets: **HotpotQA, 2Wiki, MuSiQue, NQ**.

## What the paper runner does

On a shared corpus the runner:

1. builds BM25 + NV-Embed-v2 + RRF base retrieval;
2. partitions documents from the design-query workload (`seed=42`);
3. probes a subset of regions with the same HippoRAG2 retrieval path used at
   deployment;
4. selects regions with WARP-G or a control rule (no deployment-token cutoff);
5. evaluates BM25 / Dense / Hybrid, HippoRAG2 graph-only, Base + Full Graph,
   KET-RAG, G2ConS, and the four region-selection methods;
6. writes retrieval metrics, a shared-cache reader, IRCoT, and token costs.

LinearRAG is a separate official end-to-end run. It is not a row in the WARP
region-selection table.

## Repository layout

```text
configs/paper/              main-table YAML
configs/ablations/          optional method / probe-fraction YAML
configs/official_baselines.yaml
scripts/prepare_hipporag2.py
scripts/run_paper_suite.py
scripts/export_paper_results.py
scripts/prepare_official_baselines.py
scripts/run_official_baseline.py
scripts/run_official_suite.py
scripts/export_official_results.py
warp/run.py                 paper experiment entry (`warp-g`)
tests/                      unit tests that do not call the paper LLM
```

## Environment

Needs Python 3.10+, a CUDA GPU (the paper runner refuses CPU-only execution),
Hugging Face access for the embedding/reranker weights, and an
OpenAI-compatible LLM endpoint.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
export OPENAI_API_KEY=YOUR_KEY
export OPENAI_BASE_URL=https://api.openai.com/v1   # or any compatible gateway
export HF_HOME=$PWD/.hf-cache
export CUDA_VISIBLE_DEVICES=0
```

Pinned backends in the paper YAML:

| Role | Setting in this repo |
|---|---|
| Graph / dense encoder | `nvidia/NV-Embed-v2` |
| OpenIE + reader LLM | `deepseek-v4-flash`, thinking disabled |
| Reranker | `BAAI/bge-reranker-v2-m3` @ `b5160aeac3c6c8fe7beaaaf04c9e0142826b58d1` |
| Graph implementation | HippoRAG 2.0.0a4, commit `c617143f01477243992a63b2e2151cc003dd3b21` |
| Seeds | `warp.seed: 42`, `experiment.seed: 42` |
| Retrieval cutoffs | Evidence Recall / Complete Evidence @2/3/5/10 |
| Reader | top-5, 3 repeats, shared retrieval cache |

`pip install -e .` pulls HippoRAG from the pinned Git commit in
`pyproject.toml`. A GPU is required for NV-Embed-v2 and the reranker. LLM
OpenIE and QA need a paid or self-hosted endpoint. This packaging pass did
**not** re-run the paper suite, so wall-clock and dollar cost are not restated
here.

Set `OPENAI_BASE_URL` to the same OpenAI-compatible service used in your
experiment. The YAML does not embed a lab-specific gateway.

## Data

### HippoRAG2 release (HotpotQA, 2Wiki, MuSiQue)

```bash
python3 -m pip install -U huggingface_hub
mkdir -p data/raw/hipporag2
hf download osunlp/HippoRAG_2 \
  hotpotqa.json hotpotqa_corpus.json \
  2wikimultihopqa.json 2wikimultihopqa_corpus.json \
  musique.json musique_corpus.json \
  --repo-type dataset \
  --revision 5ec05b38deecc3318bb432c69865959c56058990 \
  --local-dir data/raw/hipporag2

python3 scripts/prepare_hipporag2.py --datasets hotpotqa 2wiki musique
```

The converter writes `data/processed/{hotpotqa,2wiki,musique}/` with
`corpus.jsonl`, `queries.jsonl`, and `split_manifest.json`. Passage IDs are
`doc-` plus a hash of title and text. Every gold evidence passage must exist
in the shared corpus.

Raw HippoRAG2 JSON and the processed JSONL are **not** shipped. Hugging Face
gated weights and the HippoRAG2 dataset license still apply.

### Natural Questions

`configs/paper/nq.yaml` is a paper config, but `scripts/prepare_hipporag2.py`
does not convert NQ. There is no in-repo download or schema adapter. Place
`data/processed/nq/corpus.jsonl` and `data/processed/nq/queries.jsonl` in the
schema below before running NQ.

### Expected JSONL schema

Corpus:

```json
{"id":"doc-1","title":"Title","text":"Passage text"}
```

Queries:

```json
{"id":"q-1","query":"Question?","gold_doc_ids":["doc-1","doc-2"],"answer":["alias"]}
```

## Main experiments

There is no checkpointed model to load. Design artifacts are HippoRAG indexes
under `outputs/indexes/<dataset>/`. Documented paper commands use
`--max-folds 1`.

Single dataset (unverified end-to-end; needs data, GPU, and LLM):

```bash
python3 -m warp.run \
  --config configs/paper/hotpotqa.yaml \
  --output outputs/paper/hotpotqa.json \
  --max-folds 1
```

All four paper datasets:

```bash
python3 scripts/run_paper_suite.py
```

Useful flags (verified via `--help` only):

```bash
python3 -m warp.run --help
# --max-folds 1     paper reproduction command used in this README
# --skip-multistep  keep first-pass QA; skip the IRCoT protocol
# --checkpoint-dir  default is <output>.folds
```

Export tidy CSV from completed JSON (unverified without result files):

```bash
python3 scripts/export_paper_results.py \
  --input-dir outputs/paper \
  --output-dir outputs/paper/tables
```

### Command / config map

| Artifact | Command | Config |
|---|---|---|
| Main retrieval / reader / cost table | `python -m warp.run --config configs/paper/<ds>.yaml --max-folds 1` | `configs/paper/{hotpotqa,2wiki,musique,nq}.yaml` |
| Four-dataset suite | `python3 scripts/run_paper_suite.py` | same four YAML files |
| Partition ablations | included in the paper YAML (`partition_ablations.modes`) | query / semantic / random |
| Probe-fraction 40% (WARP + controls only) | `python -m warp.run --config configs/ablations/<ds>/probe40-warp.yaml --max-folds 1 --skip-multistep` | NQ and HotpotQA |
| LinearRAG | `python3 scripts/prepare_official_baselines.py` then `python3 scripts/run_official_suite.py` | `configs/official_baselines.yaml` |
| Official EM/F1 CSV | `python3 scripts/export_official_results.py` | `outputs/official/` |

### Output files and metrics

`outputs/paper/<dataset>.json` is self-describing. Important sections:

| Section | Contents |
|---|---|
| `run_metadata` | config snapshot, data SHA-256, package versions, CUDA, HippoRAG commit |
| `baselines` / `baseline_trials` | BM25, Dense, Hybrid, HippoRAG2, Full Graph |
| `quality_cost_curve` / `quality_cost_summary` | WARP and graph methods, with costs |
| `paired_significance` | WARP vs each reference, Holm-adjusted paired randomization |
| `partition_ablations` | query / semantic / random partitions |
| `reader_evaluation` | Answer EM / F1 on the shared top-5 cache |

Retrieval metrics: **Evidence Recall** and **Complete Evidence** at k in
{2, 3, 5, 10}, with query-level paired bootstrap 95% CIs. Tokens are
`input + output + embedding`. `actual_cost_fraction` is deployed tokens
divided by the Full Graph construction tokens. IRCoT is stored under
`multistep` and does not overwrite first-pass `retrieved_doc_ids`.

CSV export writes `outputs/paper/tables/{baselines,quality_cost_trials,quality_cost_summary,reader,paired_significance,quality_cost_auc,partition_ablations}.csv`.
With a single full-pipeline point, `quality_cost_auc` is a placeholder, not a
main-table curve.

## Official LinearRAG

```bash
python3 scripts/prepare_official_baselines.py
# clones into external/official/linearrag at the pinned commit
python3 scripts/run_official_suite.py
python3 scripts/export_official_results.py
```

The pinned commit is in `configs/official_baselines.yaml`. LinearRAG uses the
same `deepseek-v4-flash` paper LLM and no generation token cap. It is an
official end-to-end API, not a WARP region-selection row.

## Smoke checks (no paper LLM, no full run)

```bash
python3 -m warp.run --help
python3 scripts/prepare_hipporag2.py --help
python3 -c "import warp, warp.run, warp.pipeline"
python3 -m unittest discover -s tests -v
```

Those commands check the CLI surface and local unit tests. They do **not**
reproduce paper numbers.

## Anonymous supplement

When you zip this code for submission, include source, configs, tests, and
this README. Do **not** include:

- `.git/` (commit metadata is identifying)
- `data/` or `outputs/`, including local-path symlinks
- `HippoRAG/`, `external/`, `vendor/`, `.venv/`, `.hf-cache/`
- API keys, `.env`, logs, or machine-specific check files

Do not publish this tree or change remotes as part of packaging.

## What this pass does not claim

- No paper table was regenerated here.
- Full `warp.run` jobs were not executed (GPU + LLM + multi-hour OpenIE).
- NQ preprocessing is not specified in-repo.
- Exact GPU model and wall-clock hours are not recorded in the tracked
  configs, so they are omitted.
