# WARP-G：最终研究设计

## 1. 研究问题

WARP-G 研究共享语料上的 workload-aware regional graph materialization。所有方法拥有完全相同的
BM25、NV-Embed-v2 和 CrossEncoder。昂贵的 HippoRAG2 图是附加物理结构，不是基础数据库。

给定 corpus regions `R`、历史设计 workload `Q_train` 和检索指标 `M`，WARP 按方法自身规则选出
物化集合 `S ⊆ R`（独立模式 `score>0` 自然停，条件模式边际 gain>0 自然停），**不施加部署 token 预算**。
实验比较各方法完整 pipeline 的 `E[M(q; S)]` 与实际消耗 tokens。论文仍研究同一种 Graph representation
应该在哪些 regions 物化，不引入 Tree、Summary、Agent、RL 或在线更新。

研究问题固定为：

- RQ1：HippoRAG2 相对 Base 的收益是否在 corpus regions 之间显著不均匀？
- RQ2：仅使用探测区实测收益（不对未探测区域外推）时，条件边际选区是否优于独立 score 排序与个数对齐对照？
- RQ3：在各自完整 pipeline 下，WARP-G 是否优于个数对齐的 region-selection controls（random / frequency / gain）、原生 core
  比例的 KET-RAG/G2ConS，并以更低实际构图 tokens 接近 Full Graph？
- RQ4：收益排序对 partition method 的变化是否稳定，query routing 对系统上限有多大影响？
- RQ5：construction savings 是否会被策略搜索成本或在线区域图调用抵消？

## 2. 数据边界

每个数据集使用 HippoRAG2 官方发布的共享 corpus 和完整 1,000 条 query。所有配置在正式运行前冻结，
不使用 query 自动调参。固定 `seed=42` 做五折交叉拟合：

- 每折 800 条 design query 用于 partition、features、probe labels 和 materialization selection；
- 另 200 条 held-out query 只用于 retrieval、reader 和 routing evaluation；
- 五折分别重新执行完整 physical-design 流程；
- 每条 query 恰好作为一次 held-out test，最终统计在合并后的 1,000 条逐题结果上计算。

`base_recall`、`failure_rate` 和 `multi_doc_rate` 使用 design-fold gold evidence，因此方法被明确界定为
supervised workload-aware physical design，不宣称适用于完全无标注的线上日志。

数据集为 HotpotQA、2WikiMultiHopQA、MuSiQue 和 PopQA。前三个测量多跳完整证据，PopQA 是
single-hop control。所有文档 ID 和内容必须唯一，所有 gold evidence 必须存在于共享 corpus。

## 3. Base 与统一排序路径

全语料 Base 为：

```text
BM25 + NV-Embed-v2 -> RRF -> pinned BGE CrossEncoder -> top-k
```

Base、区域 probe、WARP、corpus-level baselines 和 Full Graph 使用相同 `candidate_k`、RRF、reranker
和最终 `k`。因此不部署任何区域图时，每个选择方法严格退化为同一个 Base，不会把 reranker 收益误记成图收益。

## 4. Workload-aware partition

对每个 train query 取得 Base top-`coaccess_k`。共同出现的文档形成 query coaccess edge：

```text
w_query(i,j) = #{q: d_i,d_j ∈ TopK_base(q)}
```

使用 FAISS HNSW 为每篇文档取得少量 semantic neighbors：

```text
w(i,j) = w_query(i,j) + λ × cosine(i,j)
```

Leiden 在该稀疏图上生成 regions。正式消融包括：combined、query-only、semantic-only 和在相同
region size multiset 下随机分配文档。Embedding dispersion 只采样固定数量文档对，不产生二次内存。

## 5. Region features

构图前特征为：

- `num_docs`、`num_tokens`；
- `query_freq`；
- `base_recall`、`failure_rate`；
- `avg_retrieval_entropy`；
- `multi_doc_rate`；
- `embedding_dispersion`；
- `coaccess_density`。

构图前特征只用于探测优先级和报告，不训练回归器去预测未探测区域。

## 6. Probe labels 与选择信号

Probe regions 从有 workload 覆盖的区域中按局部缺失证据/成本排序，并穿插随机探索；数量约为覆盖区域的 20%（至少一个），累计估算建图成本不超过全图估算成本的 10%。每个 probe graph 使用正式 HippoRAG2 构建。

标签走与最终系统完全一致的 candidate generation、RRF 和 CrossEncoder。paper 默认设计目标为混合效用

```text
U = (1 - complete_weight) * EvidenceRecall + complete_weight * CompleteEvidence
```

`complete_weight=0.5`。同时保存 Evidence Recall 与 Complete Evidence 的增益。标签是整题证据上的净变化，允许为负。

对探测区做 `estimated_gain = n * g / (n + 16)` 向零收缩。未探测区域不外推，估计值为 0，报告中标记为 unprobed。不对未探测区域训练回归器。

区域图集合的真实效用不被假设为严格可加。paper 默认 `interaction_pairs=0`，不把二阶交互作为主表。条件选区在探测区内直接测量单区与有限双区组合的边际增益，作为对独立可加假设的运行时替代。

## 7. WARP selection 与 baselines

独立模式 WARP score：

```text
score_i = query_freq_i × max(estimated_gain_i, 0)
```

区域不可拆分。独立模式选出全部 `score_i>0` 的区域后停止；条件模式按实测边际 gain>0 停止。
条件选区的候选短名单同样按 `query_freq × max(estimated_gain, 0)` 排序，边际比较用原始 gain，不再除 cost。
两者都不再用 token proxy 做探测或部署截断。对照只换 frequency / gain / random 排序公式，并取与 WARP 相同的区域个数。
选区公式不乘常数、不另加 ε 门槛；表中增益另报百分点（`*_pp = gain × 100`），方便阅读。

正式比较方法：

- BM25、Dense 与 Hybrid；
- Random-region、Frequency-only 与 Gain-only：复用完全相同的 regions 和检索后端，
  依次检验随机选择、workload frequency 和预测收益的单独作用；
- KET-RAG：lexical/semantic KNN PageRank core chunks、HippoRAG2 KG skeleton、全语料 keyword
  bipartite retrieval；
- G2ConS：sentence-level concept embeddings、semantic-filtered co-occurrence concept graph、PageRank
  core chunks、HippoRAG2 core-KG 和 dual-path retrieval；
- HippoRAG2 graph-only；
- Base + corpus-wide Full HippoRAG2。

KET-RAG 与 G2ConS 的昂贵 KG 都使用与 WARP 相同的 HippoRAG2 builder，避免 Graph backend 能力差异。
它们按各自论文的文档比例选取 core（正式配置 `ket_core_fraction`/`g2_core_fraction`=0.8），不再套用
WARP 的 token 预算；轻量 keyword/concept 结构始终计入 deployment cost。

region-selection controls 是 selector ablation，不是独立构图系统：它们复用同一折已经完成的 WARP
physical-design state，只替换最后的区域排序公式，并对齐 WARP 的选区个数。LinearRAG 与 LightRAG 另用锁定的作者官方仓库，在相同完整
corpus 和 1,000 queries 上运行独立 end-to-end 表；由于构图单元和成本维度不相同，不强行映射到本节实际 token 主表。
完整方法差异、官方代码状态和 commit 见 `related_work.md` 与 `configs/official_baselines.yaml`。

## 8. Query routing

Test query 先运行 Base，取 top-`routing_k` 文档并查表得到 regions。只查询已物化且被 Base 命中的图。
论文同时报告 any-gold-region recall 和 complete-gold-region recall。该指标界定系统上限：Base 没有命中
正确区域时，区域图不能修复该 query。

## 9. 成本口径

成本严格分账：

1. `deployment_cost`：最终保留图及 KET/G2ConS 轻量结构的冷构建成本；
2. `design_search_cost`：WARP 获得 probe labels 与条件选区测量的图构建/检索成本；
3. `method_specific_design_wall_seconds`：partition、feature 等非图设计时间；
4. `first_run_cost`：deployment + 方法专属 design cost；
5. `online_retrieval_cost`：每个 method/trial 独立的 query-time tokens、calls 和 wall time。

原始维度包括 LLM input/output tokens、embedding tokens、wall time、nodes、edges、storage 和按配置中
固定价格快照计算的 USD。主报这些实测 tokens 以及 CE@10 / tokens 的 token efficiency。
相对 Full Graph 的 `actual_cost_fraction` 只作描述，不是选择时的截断约束。
探测按 `probe_fraction` 抽样实测，不再用构图 token proxy 做探测或部署截断；独立选区分数只使用 frequency × gain。

## 10. 评价与统计

Retrieval 指标：Evidence Recall 与 Complete Evidence 统一报告 **@2 / @3 / @5 / @10**。
主路径检索与 Reader 使用第一遍结果；IRCoT 多步是另开的对照，不覆盖问答缓存。
Reader 固定为同一 HippoRAG2 QA LLM
（正式配置为 `deepseek-v4-flash`），报告 Answer EM/F1；Reader 复用该方法那一轮检索的文档，不另设预算档。
正式配置 `reader.repeats=3`，同一检索缓存连跑三次并报告 mean/std；温度 0 且命中 QA 缓存时三次可以完全相同。
每折检索列表写到 `<checkpoint>/retrieval/fold-<k>/<method>.jsonl`，每次 Reader 回答写到
`<checkpoint>/reader/fold-<k>/<method>-repeat-<i>.jsonl`，便于举题级例子。

每个方法只跑一轮完整 pipeline。全部正式实验固定 `seed=42`。
每个方法通过 query-level paired bootstrap 报告 95% CI。WARP 与所有参考方法做 paired
randomization test，并对同一指标的比较应用 Holm correction。

主表同时给出实际 deployment / first-run / online tokens 与 token efficiency。不再扫描
`{0, 0.1, 0.2, 0.4, 0.6, 1.0}` 预算点，也不把 quality-cost AUC 作为主结论。

## 11. 可复现性

Artifact 保存：完整 YAML、所有包版本、HippoRAG commit/API version、数据 SHA-256、Python/平台、
CUDA/cuDNN/GPU、固定 seed、每题指标、每次构建 manifest 和多维成本。正式运行要求 CUDA、
锁定模型 revision、官方 HippoRAG API、完整 usage metadata 和稳定 `Chunk.source_id`；任何不一致直接失败。
