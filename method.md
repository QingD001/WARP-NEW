# WARP-G 区域划分方法与现有方法对比

## 1. 方法概览

WARP-G 的区域划分不是对已经构建好的知识图谱做社区发现，而是先利用历史查询构造一张廉价的文档共访问图，再在这张图上运行 Leiden，将完整语料划分为若干互不重叠的 GraphRAG 物化单元。

```text
历史设计查询
    ↓
BM25 + Dense + RRF
    ↓
每个查询的 Top-20 文档
    ↓
查询共访问边 + 弱语义近邻边
    ↓
文档级加权无向图
    ↓
Leiden 社区发现
    ↓
互不重叠的语料区域
    ↓
区域探测、收益估计和区域选择
```

这里需要区分两张用途完全不同的图：

- **划分图**：节点是 passage，边来自查询共访问和语义近邻。它是一张廉价的临时设计图，只用于确定区域边界。
- **区域知识图**：在选中区域内部使用 HippoRAG2 构建的实体—事实—段落图。它是昂贵的持久化检索结构，用于最终图检索。

因此，WARP-G 的区域划分发生在昂贵知识抽取和图构建之前，其作用是确定“哪些文档应该作为一个可独立构建、计费和选择的部署单元”。

## 2. 当前区域划分的完整过程

### 2.1 仅使用设计查询

每个数据集采用五折交叉拟合。每一折只允许设计查询参与区域划分，测试查询不能影响共访问边、区域边界、区域特征、探测区域和区域选择。

系统首先在完整语料上建立共享基础检索器：

\[
H=\operatorname{RRF}(\mathrm{BM25},\mathrm{Dense}).
\]

Dense 检索器使用 NV-Embed-v2。用于划区的是 BM25 和 Dense 经 Reciprocal Rank Fusion 得到的排序结果，尚未经过 CrossEncoder，也没有使用 HippoRAG2。基础索引及划分入口见 [`warp/pipeline.py`](warp/pipeline.py#L108)。

这意味着区域边界来自廉价基础检索器观察到的历史访问模式，而不是直接由 gold evidence、CrossEncoder 或昂贵图后端划定。

### 2.2 根据查询结果建立文档共访问边

对每个设计查询 \(q\)，基础检索器取前 \(k_c=20\) 篇文档：

\[
L_q=\operatorname{Top20}_H(q).
\]

然后将同一结果列表中的文档两两连接。若 \(d_i\) 和 \(d_j\) 同时出现在一个查询的 Top-20 中，它们之间的边权增加 1：

\[
w_{\mathrm{query}}(i,j)
=
\sum_{q\in Q_{\mathrm{des}}}
\mathbb{I}[d_i\in L_q\land d_j\in L_q].
\]

例如：

```text
查询 q1 → {A, B, C}
查询 q2 → {A, B, D}
查询 q3 → {A, B}
```

对应的部分边权为：

```text
w(A,B) = 3
w(A,C) = 1
w(B,C) = 1
w(A,D) = 1
w(B,D) = 1
```

因此，A 和 B 会被强烈倾向于分入同一区域。该边表达的不是单纯的文本相似性，而是：

> 在当前查询工作负载下，这两篇文档经常被同一查询共同访问。

每个查询的 Top-20 最多形成

\[
\binom{20}{2}=190
\]

个文档对。不同查询产生的相同文档对会累积权重。具体实现见 [`warp/partition/coaccess_graph.py`](warp/partition/coaccess_graph.py#L46)。

### 2.3 使用弱语义边补充未覆盖文档

只使用历史查询会使从未进入设计查询 Top-20 的文档缺少共访问边，容易成为孤立节点。WARP-G 因此使用 Dense 向量为每篇文档寻找 \(k_s=5\) 个语义近邻。当前实现通过 FAISS HNSW 近似搜索余弦近邻：

\[
N_s(d_i)=\operatorname{Top5}_{d_j}\cos(e_i,e_j).
\]

语义边定义为：

\[
w_{\mathrm{sem}}(i,j)=\cos(e_i,e_j).
\]

最终边权为：

\[
w(i,j)
=
w_{\mathrm{query}}(i,j)
+
\lambda w_{\mathrm{sem}}(i,j),
\qquad \lambda=0.05.
\]

一次查询共现贡献 1，而一条语义边最多约贡献 0.05。因此，查询共访问关系是主要信号，语义关系只是弱补充：

- 有历史查询覆盖时，由查询共访问关系决定主要结构；
- 没有工作负载覆盖时，语义关系避免文档完全孤立；
- 语义相似度不会轻易覆盖真实查询产生的共访问结构。

如果两个方向都找到同一语义近邻，当前代码保留最大相似度，不重复累加。具体实现见 [`warp/partition/coaccess_graph.py`](warp/partition/coaccess_graph.py#L57)，FAISS HNSW 近邻搜索见 [`warp/retrieval/ann.py`](warp/retrieval/ann.py)。

### 2.4 使用 Leiden 发现文档社区

得到加权无向图

\[
G_{\mathrm{part}}=(D,E,w)
\]

后，WARP-G 使用 Leiden 的 `RBConfigurationVertexPartition` 进行社区发现：

\[
\mathcal R=\operatorname{Leiden}(G_{\mathrm{part}}).
\]

当前默认参数为：

\[
\mathrm{resolution}=1.0,
\qquad
\mathrm{seed}=42.
\]

Leiden 倾向于将内部连接较强、外部连接较弱的文档组织为同一社区。与 K-means 不同，WARP-G 不预先指定区域数量；区域数量和大小由加权图结构与 resolution 共同决定。实现见 [`warp/partition/leiden.py`](warp/partition/leiden.py#L29)。

### 2.5 合并过小区域

Leiden 可能产生很小的社区。当前正式配置要求每个区域至少包含 5 篇文档。若区域 \(r_i\) 的大小小于 5，系统计算它与每个合格区域之间的总连接权重：

\[
a(r_i,r_j)
=
\sum_{u\in r_i}
\sum_{v\in r_j}
w(u,v).
\]

小区域被并入连接最强的合格区域：

\[
r_i\rightarrow
\arg\max_{r_j:\lvert r_j\rvert\geq 5}a(r_i,r_j).
\]

如果小区域与其他区域完全没有连接，当前实现会确定性地选择编号最小的候选区域。合并逻辑见 [`warp/partition/leiden.py`](warp/partition/leiden.py#L42)。

最终区域满足：

\[
r_i\cap r_j=\varnothing,
\qquad
\bigcup_i r_i=D.
\]

因此：

- 每篇文档只属于一个区域；
- 所有文档都被某个区域覆盖；
- 区域是后续探测、成本估计、区域选择和图构建的不可拆分单元。

### 2.6 当前正式参数

四个数据集的正式配置采用相同的区域划分参数：

| 参数 | 当前值 | 含义 |
|---|---:|---|
| `coaccess_k` | 20 | 每个设计查询用于建立共访问边的文档数 |
| `semantic_k` | 5 | 每篇文档的语义近邻数 |
| `semantic_lambda` | 0.05 | 语义边相对查询共访问边的权重 |
| `partition_resolution` | 1.0 | Leiden 默认 resolution |
| `partition_mode` | `combined` | 同时使用查询边和语义边 |
| `min_region_size` | 5 | 区域的最小文档数 |
| `seed` | 42 | Leiden 和其他随机过程的固定种子 |

配置示例见 [`configs/paper/hotpotqa.yaml`](configs/paper/hotpotqa.yaml#L5)。

### 2.7 区域形成后的查询—区域映射

区域划分完成后，系统重新为每个设计查询取得 Base 前 50 个候选，并读取前 20 个结果所属的区域。如果查询 \(q\) 的前 20 个结果中至少有一篇属于区域 \(r_i\)，系统就认为该查询会访问该区域：

\[
q\rightarrow r_i
\iff
r_i\cap\operatorname{Top20}_H(q)\neq\varnothing.
\]

一个查询可以映射到多个区域，但在同一区域内只计一次。该关系用于计算：

- `query_freq`；
- `base_recall`；
- `failure_rate`；
- 区域探测使用的查询集合；
- 最终在线路由。

这一步不再改变区域边界。实现见 [`warp/advisor/features.py`](warp/advisor/features.py#L38)。

gold evidence 不参与区域边界的形成，但会在区域形成后用于计算 `base_recall`、`failure_rate` 和 `multi_doc_rate`。因此，当前划分结构本身来自查询日志与语料表示，后续区域收益建模则属于有监督的物理设计。

## 3. 区域的实际含义

WARP-G 的区域不是严格意义上的主题簇，也不是实体知识图谱社区。它更接近：

> 在历史工作负载和当前基础检索器下，未来可能被同一批查询共同访问的一组文档。

同一区域中的文档可能因为三种原因聚集：

1. 它们经常被同一个查询同时召回；
2. 它们经常与同一批其他文档共同竞争；
3. 它们缺少查询覆盖，但在语义空间中彼此接近。

第一种对应理想的多跳证据关系，第二种也包含基础检索器的歧义和错误。因此，“共访问”不等于“共同正确证据”。它同时刻画了真实需求和基础检索器产生的候选结构。

这与当前系统的部署目标一致：区域不仅应当包含可能共同有用的证据，还应尽量包含在线路由时容易被一起命中的文档。

## 4. 与现有方法的总体比较

| 方法 | 划分或选择对象 | 主要信号 | 发生时间 | 主要目的 |
|---|---|---|---|---|
| 语义聚类/K-means | 文档向量 | 文本相似度 | 构图前 | 获得语义或主题簇 |
| Microsoft GraphRAG 社区 | 已抽取的实体知识图 | 实体关系与图拓扑 | 昂贵构图后 | 生成社区摘要、支持全局理解 |
| HippoRAG/HippoRAG2 | 实体、事实和段落节点 | 抽取关系、同义与相似关系 | 构图后 | 图传播和多跳证据检索 |
| KET-RAG | 全局文本块 | KNN 图中心性/PageRank | 部分构图前 | 选择核心文本构建 KG 骨架 |
| G2ConS | 概念和文本块 | 概念共现与全局显著性 | 部分构图前 | 选择核心概念和文本 |
| WawPart/AWAPart | 已有 RDF 三元组 | SPARQL 查询模式 | KG 已存在后 | 减少跨节点连接与通信 |
| WARP-G | 原始 passage | 查询共访问为主、语义近邻为辅 | 昂贵 GraphRAG 构图前 | 产生可选择、可计费的区域图物化单元 |

## 5. 与纯语义聚类的区别

纯语义聚类通常根据

\[
\operatorname{sim}(d_i,d_j)=\cos(e_i,e_j)
\]

把内容相似的文档放在一起。这种方法优化的是主题纯度或向量空间紧致性，但语义相似不一定意味着两篇文档会被同一问题共同需要。

例如，“法国总统”和“法国历史”在语义空间中可能接近，却可能服务于完全不同的查询；“某人的出生地”和“该地点所属国家”表面语义未必高度相似，却经常构成同一个多跳问题的两段证据。

WARP-G 的主信号是：

\[
\#\{\text{共同召回 }d_i,d_j\text{ 的历史查询}\},
\]

因此它优化的是工作负载局部性，而不是单纯的主题纯度。语义边主要承担冷文档平滑作用。

当前代码已经实现四种划分消融：

- `combined`：查询共访问边与语义边；
- `query`：仅查询共访问边；
- `semantic`：仅语义边；
- `random`：保持 combined 区域大小 multiset 不变，随机分配文档。

相关实现见 [`warp/pipeline.py`](warp/pipeline.py#L120)。这些消融应当用于证明共访问结构是否确实优于纯语义聚类，以及真实区域成员关系是否优于仅匹配区域大小的随机划分。

## 6. 与 Microsoft GraphRAG 社区发现的区别

Microsoft GraphRAG 先使用 LLM 从完整语料抽取实体和关系，再对实体知识图进行社区划分，并为社区生成摘要，以支持全局问题和查询聚焦总结。参见 [From Local to Global: A Graph RAG Approach to Query-Focused Summarization](https://www.microsoft.com/en-us/research/publication/from-local-to-global-a-graph-rag-approach-to-query-focused-summarization/)。

Microsoft GraphRAG 的流程可以概括为：

```text
完整语料
→ LLM 实体关系抽取
→ 完整实体知识图
→ 知识图社区发现
→ 社区摘要
```

WARP-G 的流程则是：

```text
原始文档
→ 廉价 Base 检索
→ 文档共访问图
→ Leiden 区域
→ 只在部分区域构建昂贵知识图
```

二者虽然都可以使用 Leiden，但输入和目标不同：

- GraphRAG 社区是在昂贵构图之后产生的知识组织结构；
- WARP-G 区域是在昂贵构图之前产生的物理部署结构；
- GraphRAG 社区优化实体图的层次组织和全局总结；
- WARP-G 区域用于决定哪些文本值得承担 GraphRAG 构建成本。

因此，关键区别不是社区发现算法，而是“对什么图进行社区发现”和“得到的社区被用于什么决策”。

## 7. 与 HippoRAG/HippoRAG2 的区别

HippoRAG2 构建的是实际检索图，包含实体、事实、段落节点以及相似或同义关系，其目标是通过图传播找回相关证据。

WARP-G 的共访问图不参与最终答案检索，只负责决定：

- 哪些 passage 应当共同进入一个 HippoRAG2 实例；
- 哪些区域值得真正构建 HippoRAG2；
- 查询时应调用哪些已经部署的区域图。

因此，WARP-G 不是 HippoRAG2 的替代方法，而是位于图后端之外的部署层：

\[
\text{WARP-G 选择部署范围}
\quad+\quad
\text{HippoRAG2 提供区域内部图能力}.
\]

## 8. 与 KET-RAG 的区别

KET-RAG 从文本块 KNN 图中识别关键文本，只对核心文本使用 LLM 构建知识图谱骨架，并通过廉价的文本—关键词二部图保持剩余语料的可检索性。参见 [KET-RAG](https://arxiv.org/abs/2502.09304)。

二者的主要区别是：

- KET-RAG 选择全局核心文本，WARP-G 先形成互不重叠的区域，再选择完整区域；
- KET-RAG 的选择信号主要来自语料图中的全局中心性，WARP-G 的区域边界主要来自历史查询共访问；
- KET-RAG 构建一个全局稀疏 KG 骨架和轻量覆盖结构，WARP-G 构建多个隔离、可独立部署的区域图；
- KET-RAG 不直接估计图相对共享 Base 的区域级边际收益；
- WARP-G 在少量区域上真实运行最终图后端，以实测证据增益训练收益预测器。

例如，一个主题在完整语料中比较边缘，但真实工作负载经常查询它，而且 Base 经常漏掉多跳证据：KET-RAG 可能因为其全局中心性较低而不选择相关文本，WARP-G 则可能因为访问频率和区域图收益较高而选择该区域。

## 9. 与 G2ConS 的区别

G2ConS 构建概念共现图，识别重要概念及其关联文本，选择部分文本构建昂贵知识图，同时使用无需 LLM 的概念图补足覆盖。参见 [Graph-Guided Concept Selection for Efficient Retrieval-Augmented Generation](https://arxiv.org/abs/2510.24120)。

G2ConS 主要回答：

> 哪些概念或文本在语料内部最重要？

WARP-G 主要回答：

> 对当前查询分布而言，哪些语料区域最需要图能力，而且图能力能否真正改善共享基础检索器？

因此，G2ConS 的主要选择信号是 corpus salience，WARP-G 的主要决策信号是 workload-conditioned marginal utility。二者都减少昂贵构图范围，但决策依据和最终物理结构不同。

## 10. 与工作负载感知知识图谱分区的区别

WawPart 和 AWAPart 同样根据查询工作负载划分知识图谱。它们根据 SPARQL 查询特征组织已有 RDF 三元组，以减少跨分片连接和分布式通信；AWAPart 还会随工作负载变化调整分片。参见 [WawPart](https://arxiv.org/abs/2203.14888) 和 [AWAPart](https://arxiv.org/abs/2203.14884)。

WARP-G 与它们的区别在于：

- WawPart/AWAPart 的输入是已经存在的结构化知识图谱；
- WARP-G 的输入是原始文本，此时昂贵知识图尚未构建；
- WawPart/AWAPart 优化 SPARQL 执行时间、跨分片连接与通信；
- WARP-G 优化证据召回、答案质量与图构建成本之间的权衡；
- WawPart/AWAPart 决定已有图如何分布；
- WARP-G 决定哪些区域的知识图应该被创建和长期部署。

因此，“工作负载感知划分”本身不是 WARP-G 可以单独宣称的新颖点。WARP-G 的区别在于：

> 使用自然语言查询产生的文档共访问区域作为候选物化单元，通过少量真实 GraphRAG 探测学习区域级质量收益，再按边际收益选择区域图，并用实际构图 tokens 评价部署成本。

## 11. 当前区域划分的优势

### 11.1 与真实查询分布对齐

热门查询以及经常共同访问的文档会直接影响区域边界。区域由系统实际观察到的检索轨迹形成，而不是只依赖静态语料结构。

### 11.2 发生在昂贵构图之前

区域划分只依赖共享基础检索器和文档向量，不需要先为完整语料抽取实体关系。因此，选择性部署不会为了决定“构建哪里”而先支付一次全量 GraphRAG 成本。

### 11.3 能覆盖冷文档

弱语义边让没有设计查询覆盖的文档仍能进入区域体系，避免工作负载图中出现大量无归属或完全孤立的文档。

### 11.4 区域可以直接作为物理部署单元

每个区域可以独立完成：

- 构图探测；
- 收益预测；
- 成本估计；
- 区域选择；
- 图索引构建；
- 在线路由与调用。

区域划分因此不是单独的聚类预处理，而是后续物理设计闭环的候选生成步骤。

## 12. 当前实现的局限与建议实验

### 12.1 Top-20 clique 可能引入伪关系

同一查询召回的两篇文档不一定共同有用，它们也可能只是同时成为候选。因此，共访问图混合了：

- 真正的多跳证据关系；
- 同主题关系；
- Base Retriever 的检索混淆。

当前所有 Top-20 文档对统一增加 1，没有体现它们在结果列表中的排名差异。可以考虑排名衰减边权：

\[
w_q(i,j)
=
\frac{1}{\log(1+\operatorname{rank}_q(i))}
\frac{1}{\log(1+\operatorname{rank}_q(j))}.
\]

建议将统一共现计数与排名衰减版本进行消融比较。

### 12.2 原始共现次数容易被高频主题和通用文档支配

当前查询边直接累加，没有按查询频率、结果列表大小或文档流行度归一化。频繁出现在大量结果列表中的通用文档可能成为高连接枢纽。

可以比较以下边权：

- 原始共现次数；
- Jaccard；
- PMI/NPMI；
- IDF 式文档流行度校正；
- 排名衰减共访问。

### 12.3 语义权重较弱且固定

当前查询边的最小增量是 1，语义边最多约为 0.05。只要存在查询共现，语义信号通常不会改变强边结构。这符合 workload-first 的设计，但仍需要通过 \(\lambda\) 敏感性实验说明 0.05 是否合理。

建议至少测试：

\[
\lambda\in\{0,0.01,0.05,0.1,0.2\}.
\]

### 12.4 缺少最大区域大小和成本平衡约束

当前只限制区域至少包含 5 篇文档，没有限制最大区域大小，也没有让 Leiden 直接考虑预计构图成本。部分热点可能形成很大区域，导致：

- 单个区域成本过高；
- 过大区域可能进不了 10% 探测构图 proxy；
- 区域内部仍包含价值差异较大的子区域。

可以进一步研究：

- size-constrained Leiden；
- 对过大区域递归划分；
- 按 token 数而不是文档数约束区域；
- 在社区目标中加入构图成本或负载均衡项。

### 12.5 硬划分会切断跨区域关系

每篇文档只能属于一个区域。如果多跳证据落在不同区域，独立 HippoRAG2 图之间无法进行一次连续图传播。当前系统可以同时路由到多个区域并融合检索结果，但不能在区域图之间遍历实体或事实关系。

应通过以下指标判断该问题的严重程度：

- any-gold-region routing recall；
- complete-gold-region routing recall；
- Full Graph 与 Regional Graph 的质量差距；
- 区域对二阶交互；
- overlapping-region 或边界文档复制消融。

### 12.6 区域随工作负载变化

五折交叉拟合中，每折使用不同设计查询重新划分区域，因此区域组成可能发生变化。这能够避免测试泄漏，但也需要报告区域稳定性，例如：

- 不同设计样本下区域大小分布；
- Adjusted Rand Index 或 NMI；
- 关键文档对共同分区的稳定性；
- 不同随机种子下区域选择结果的变化。

## 13. 新颖性边界

当前区域划分不能单独宣称为新的图划分算法，因为其组成部分——共访问图、语义 kNN 和 Leiden——都有明确先例。论文不应宣称：

- 首次提出工作负载感知图划分；
- 首次使用共访问图；
- 首次使用 Leiden 划分 GraphRAG；
- 首次选择性构建知识图谱。

更准确的贡献在于区域的角色、输入信号和后续闭环：

1. 区域在昂贵知识图构建之前形成；
2. 区域边界主要由自然语言查询产生的检索共访问关系决定；
3. 区域是可独立构建、计费和选择的 GraphRAG 物理结构；
4. 少量区域被真实构图以测量相对共享 Base 的边际收益；
5. 系统按边际收益选择区域（独立模式 score>0，条件模式边际 gain>0），对照对齐 WARP 的选区个数，并报告实际构图 tokens。

WARP-G 真正需要强调的是完整闭环：

\[
\text{查询共访问划区}
\rightarrow
\text{少量真实 GraphRAG 探测}
\rightarrow
\text{区域收益预测}
\rightarrow
\text{需求—收益—成本联合选择}
\rightarrow
\text{区域图部署与在线路由}.
\]

## 14. 推荐的论文表述

### 14.1 中文表述

> WARP-G 根据历史查询诱导的文档共访问关系构建轻量划分图，并以弱语义边补充工作负载未覆盖的文档，由此形成可独立构建和选择的语料区域。与基于语料相似性的聚类或构图后的知识图社区不同，这些区域不是语义摘要单元，而是面向 GraphRAG 选择性部署的物理设计单元。

### 14.2 英文表述

> WARP-G constructs a pre-index document graph from workload-induced co-access and weak semantic links, and partitions it into independently materializable regions. Unlike corpus-centric clustering or post-construction knowledge-graph communities, these regions serve as workload-aligned physical design units for selective GraphRAG construction.

### 14.3 最核心的区别

一句话概括：

> 现有方法通常根据语料结构决定文档如何组织，WARP-G 则根据历史查询决定哪些文档应共同成为一个图部署单元，并进一步学习这个部署单元是否值得构图。
