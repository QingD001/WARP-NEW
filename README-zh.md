# WARP-G

审稿以英文 [`README.md`](README.md) 为准，本文是对照译本。

WARP-G 是面向 GraphRAG 的 workload-aware 区域图物化系统。没有单独的训练循环。物理设计（分区、探测、选区）和 held-out 评测都在 `python -m warp.run` 里完成。

论文数据集：**HotpotQA、2Wiki、MuSiQue、NQ**。

## 复现路径（先看这里）

1. 安装依赖，并设置 `OPENAI_API_KEY` / `OPENAI_BASE_URL`（环境）。
2. 下载并转换 HotpotQA / 2Wiki / MuSiQue（数据）。NQ 的 JSONL 需自行放好。
3. 运行 `python -m warp.run --config configs/paper/<ds>.yaml --max-folds 1`。
4. 从结果 JSON 导出 CSV。只有需要官方 LinearRAG 那一行时才跑官方脚本。

## 论文 runner 做什么

在共享语料上，runner 会：

1. 建立 BM25 + NV-Embed-v2 + RRF 基础检索；
2. 按设计 query 的 workload 划分文档（`seed=42`）；
3. 用与上线相同的 HippoRAG2 检索路径探测一部分区域；
4. 用 WARP-G 或对照规则选区（部署阶段不再按 token 预算截断）；
5. 评测 BM25 / Dense / Hybrid、HippoRAG2 仅图、Base + Full Graph、KET-RAG、G2ConS，以及四种选区方法；
6. 写出检索指标、共享缓存上的 reader、IRCoT 和 token 成本。

LinearRAG 是单独的官方端到端实验，不是 WARP 选区表里的一行。

## 仓库结构

```text
configs/paper/              主表 YAML
configs/ablations/          探测比例 40% 的 YAML
configs/official_baselines.yaml
scripts/prepare_hipporag2.py
scripts/run_paper_suite.py
scripts/export_paper_results.py
scripts/prepare_official_baselines.py
scripts/run_official_baseline.py
scripts/run_official_suite.py
scripts/export_official_results.py
warp/run.py                 论文实验入口（`warp-g`）
```

## 环境

需要 Python 3.10+、一块 CUDA GPU（论文 runner 拒绝纯 CPU）、Hugging Face 上 embedding / reranker 权重的访问权限，以及 OpenAI 兼容的 LLM 接口。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
export OPENAI_API_KEY=YOUR_KEY
export OPENAI_BASE_URL=https://api.openai.com/v1   # 或其它兼容网关
export HF_HOME=$PWD/.hf-cache
export CUDA_VISIBLE_DEVICES=0
```

论文 YAML 中锁定的后端：

| 角色 | 本仓库设置 |
|---|---|
| 图 / dense encoder | `nvidia/NV-Embed-v2` |
| OpenIE + reader LLM | `deepseek-v4-flash`，关闭 thinking |
| Reranker | `BAAI/bge-reranker-v2-m3` @ `b5160aeac3c6c8fe7beaaaf04c9e0142826b58d1` |
| 图实现 | HippoRAG 2.0.0a4，commit `c617143f01477243992a63b2e2151cc003dd3b21` |
| 随机种子 | `warp.seed: 42`，`experiment.seed: 42` |
| 检索截断 | Evidence Recall / Complete Evidence @2/3/5/10 |
| Reader | top-5，3 次 repeat，共享检索缓存 |

`pip install -e .` 会按 `pyproject.toml` 拉取锁定的 HippoRAG commit。NV-Embed-v2 和 reranker 需要 GPU。OpenIE 与 QA 需要付费或自建接口。

把 `OPENAI_BASE_URL` 设成实验使用的 OpenAI 兼容服务。YAML 里不写站点专用网关。

精确的 GPU 型号、卡数和墙钟小时**没有**写在已跟踪的配置里。资源数字以论文为准。本 README 不重述表格数字。

## 数据

### HippoRAG2 发布包（HotpotQA、2Wiki、MuSiQue）

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

转换脚本写出 `data/processed/{hotpotqa,2wiki,musique}/`，内含 `corpus.jsonl`、`queries.jsonl` 和 `split_manifest.json`。passage ID 为 `doc-` 加上 title 与 text 的哈希。每条 gold evidence 都必须存在于共享语料中。

原始 HippoRAG2 JSON 和处理后的 JSONL **不随代码发布**。Hugging Face 门控权重和 HippoRAG2 数据集许可仍然适用。

### Natural Questions

`configs/paper/nq.yaml` 是论文配置，但 `scripts/prepare_hipporag2.py` 不转换 NQ。仓库内没有下载或 schema 适配脚本。跑 NQ 前请自行放入符合下方 schema 的 `data/processed/nq/corpus.jsonl` 和 `data/processed/nq/queries.jsonl`。

### 期望的 JSONL schema

Corpus：

```json
{"id":"doc-1","title":"Title","text":"Passage text"}
```

Queries：

```json
{"id":"q-1","query":"Question?","gold_doc_ids":["doc-1","doc-2"],"answer":["alias"]}
```

## 主实验

没有可加载的训练好的模型。设计产物是 `outputs/indexes/<dataset>/` 下的 HippoRAG 索引。文档中的论文命令使用 `--max-folds 1`。

单个数据集（需要数据、GPU 和 LLM）：

```bash
python3 -m warp.run \
  --config configs/paper/hotpotqa.yaml \
  --output outputs/paper/hotpotqa.json \
  --max-folds 1
```

四个论文数据集：

```bash
python3 scripts/run_paper_suite.py
```

常用参数（仅通过 `--help` 核对过）：

```bash
python3 -m warp.run --help
# --max-folds 1     本 README 使用的论文复现命令
# --skip-multistep  只做第一遍 QA，跳过 IRCoT
# --checkpoint-dir  默认是 <output>.folds
```

从已完成的 JSON 导出 CSV：

```bash
python3 scripts/export_paper_results.py \
  --input-dir outputs/paper \
  --output-dir outputs/paper/tables
```

### 命令 / 配置对照

| 产物 | 命令 | 配置 |
|---|---|---|
| 主检索 / reader / 成本表 | `python -m warp.run --config configs/paper/<ds>.yaml --max-folds 1` | `configs/paper/{hotpotqa,2wiki,musique,nq}.yaml` |
| 四数据集套件 | `python3 scripts/run_paper_suite.py` | 同上四个 YAML |
| 分区消融 | 写在论文 YAML 的 `partition_ablations.modes` | query / semantic / random |
| 探测比例 40%（仅 WARP + 对照） | `python -m warp.run --config configs/ablations/<ds>/probe40-warp.yaml --max-folds 1 --skip-multistep` | `configs/ablations/{hotpotqa,2wiki,musique,nq}/probe40-warp.yaml` |
| LinearRAG | 先 `python3 scripts/prepare_official_baselines.py`，再 `python3 scripts/run_official_suite.py` | `configs/official_baselines.yaml` |
| 官方 EM/F1 CSV | `python3 scripts/export_official_results.py` | `outputs/official/` |

### 输出文件与指标

`outputs/paper/<dataset>.json` 自描述。主要字段：

| 字段 | 内容 |
|---|---|
| `run_metadata` | 配置快照、数据 SHA-256、包版本、CUDA、HippoRAG commit |
| `baselines` / `baseline_trials` | BM25、Dense、Hybrid、HippoRAG2、Full Graph |
| `quality_cost_curve` / `quality_cost_summary` | WARP 与图方法，含成本 |
| `paired_significance` | WARP 对各对照的 Holm 校正配对随机化检验 |
| `partition_ablations` | query / semantic / random 分区 |
| `reader_evaluation` | 共享 top-5 缓存上的 Answer EM / F1 |

检索指标：k ∈ {2, 3, 5, 10} 的 **Evidence Recall** 与 **Complete Evidence**，带 query-level paired bootstrap 95% CI。tokens 为 `input + output + embedding`。`actual_cost_fraction` 是部署 tokens 除以 Full Graph 构图 tokens。IRCoT 存在 `multistep` 下，不覆盖第一遍的 `retrieved_doc_ids`。

CSV 导出路径为 `outputs/paper/tables/{baselines,quality_cost_trials,quality_cost_summary,reader,paired_significance,quality_cost_auc,partition_ablations}.csv`。只有一个完整 pipeline 点时，`quality_cost_auc` 只是占位，不是主表曲线。

## 官方 LinearRAG

```bash
python3 scripts/prepare_official_baselines.py
# 按锁定 commit 克隆到 external/official/linearrag
python3 scripts/run_official_suite.py
python3 scripts/export_official_results.py
```

锁定 commit 在 `configs/official_baselines.yaml`。LinearRAG 使用与论文相同的 `deepseek-v4-flash`，且不设 generation token 上限。

## 冒烟检查（不调论文 LLM，不跑完整实验）

```bash
python3 -m warp.run --help
python3 scripts/prepare_hipporag2.py --help
python3 -c "import warp, warp.run, warp.pipeline"
```

## 本发布不包含

- 处理后的语料、索引、缓存或结果 JSON。
- NQ 的下载 / 转换脚本。
- GPU 型号、卡数或墙钟小时（未写入已跟踪配置）。
- 重新生成的论文表。
