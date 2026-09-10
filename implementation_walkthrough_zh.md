# WARP-G 当前实现详解：从历史问题到预算化区域构图

> 阅读对象：希望理解当前方法如何具体执行、各模块如何衔接的读者。  
> 对照版本：2026-09-08 工作区中的 Python 实现与正式实验配置。  
> 本文中的演示数据均为假设，不代表实验结果。代码片段中明确标为“示意”的内容用于说明流程，不是可独立运行的完整程序。

本文按数据流讲解当前实现。第一次阅读建议按顺序读；之后可以通过目录定位公式、参数或对应代码。

## 目录

- [1. 方法到底要解决什么问题](#s1)
- [2. 先区分数据对象和两张图](#s2)
- [3. 输入数据与五折设计](#s3)
- [4. 全量 Base：BM25、Dense 与 RRF](#s4)
- [5. 从历史问题建立文档共访问图](#s5)
- [6. 用 Leiden 将文档划分为区域](#s6)
- [7. 为区域计算九维构图前特征](#s7)
- [8. 选择少量区域做真实 probe](#s8)
- [9. 一个区域图具体怎样建立](#s9)
- [10. 怎样测量区域图的真实收益](#s10)
- [11. LightGBM 怎样学习和预测收益](#s11)
- [12. 怎样在预算内选择区域](#s12)
- [13. 区域交互分析做了什么](#s13)
- [14. 选中区域怎样物化和复用](#s14)
- [15. 新问题到来后的完整在线流程](#s15)
- [16. 用一个多跳问题串起全过程](#s16)
- [17. 从检索证据到生成答案](#s17)
- [18. 预算和实际成本怎样核算](#s18)
- [19. Full Graph、对照实验与验证](#s19)
- [20. 参数、调用链和代码阅读顺序](#s20)
- [21. 容易混淆的实现细节](#s21)

<a id="s1"></a>
## 1. 方法到底要解决什么问题

**WARP-G 的核心决策是：给定有限的构图预算，哪些文档区域值得建立图索引？**

它联合考虑三个因素：

1. 这个区域被历史问题访问得多不多。
2. 给这个区域构图，能够改善多少检索质量。
3. 给这个区域构图，要付出多少成本。

困难主要在第二项：没有建图之前，通常不知道建图有没有用；但如果为测收益而把所有区域图都建了，就已经付出了想避免的全量投入。

当前方法先试建少量区域图，测出真实收益，再训练模型预测其他区域的收益，最后做预算选择。

```text
全量文档 + 带正确证据标注的历史问题
                  │
                  ▼
      为全部文档建立 BM25 和 Dense 索引
                  │
                  ▼
    用历史问题检索，统计文档共同出现的情况
                  │
                  ▼
       文档共访问图 + 弱语义近邻连接
                  │
                  ▼
          Leiden 划分文档区域
                  │
                  ▼
       为每个区域计算九维廉价特征
                  │
                  ▼
   少量区域真实构图并试检索，得到收益标签
                  │
                  ▼
      LightGBM 预测其余区域的构图收益
                  │
                  ▼
    按“访问频率 × 收益 ÷ 成本”预算选择
                  │
                  ▼
           建立选中区域的图索引
                  │
                  ▼
新问题 → Base 候选 → 相关且选中的区域图
                  │
                  ▼
       RRF 融合 → CrossEncoder 重排
                  │
                  ▼
          返回证据，按需生成答案
```

整体上有两个阶段：**离线设计决定哪些图可用，在线检索决定当前问题访问其中哪些图。**

总编排代码：[warp/pipeline.py](warp/pipeline.py)。

<a id="s2"></a>
## 2. 先区分数据对象和两张图

### 2.1 Document：一个可检索文本块

代码里的 `Document` 是一个 passage，不一定是一整篇原始文章：

```python
Document(
    id="d1",
    title="某部电影",
    text="该电影由导演甲执导……",
)
```

BM25、Dense 和图后端统一使用 `Document.content`，即标题加正文：

```text
某部电影
该电影由导演甲执导……
```

文档 ID 用于跨模块匹配、结果映射和证据评测。当前流程检查 ID 唯一性和内容唯一性，因为图后端按内容哈希管理 chunk，重复内容可能破坏不同文档 ID 的对应关系。

### 2.2 Query：问题、正确证据与答案

```python
Query(
    id="q1",
    text="这部电影的导演出生在哪个城市？",
    gold_doc_ids=["d1", "d2"],
    answer="城市乙",
)
```

假设文档内容为：

```text
d1：电影由导演甲执导。
d2：导演甲出生于城市乙。
```

`gold_doc_ids` 表示标准证据集合。`answer` 用于生成答案评测；收益学习主要依赖证据标签。

### 2.3 Region：不可拆分的物化单元

```python
Region(
    id="r0001",
    doc_ids=["d1", "d2", "d8", "d15"],
)
```

选中该区域，就为其中所有文档建立一个区域图索引。选择器不会只取其中一半文档来填满剩余预算。

### 2.4 SearchResult：统一的检索结果

不同检索器都返回统一对象，包含 `doc_id`、`score`、`source`、`rank` 和可选的 `region_id`。这使 BM25、Dense、图结果能够进入相同融合和评测接口。

### 2.5 分区图与检索图是不同对象

| 对比项 | 文档共访问图 | 区域 HippoRAG2 图 |
|---|---|---|
| 建立时间 | 分区之前 | probe 或正式物化时 |
| 基础对象 | 文档 / passage | 官方后端组织的实体、事实、passage 等表示 |
| 连接依据 | 检索共现与向量相似性 | 官方知识抽取与图构建过程 |
| 是否需要 LLM 抽取 | 不需要 | 需要 |
| 用途 | 决定文档怎样分组 | 提供图增强检索候选 |
| 覆盖范围 | 全量语料 | 当前构建的单个区域 |

**先划分区域不要求先建全量知识图谱。**否则节省构图成本的前提就不成立了。

数据结构定义：[warp/models.py](warp/models.py)。

<a id="s3"></a>
## 3. 输入数据与五折设计

正式配置针对每个包含 1,000 个问题的数据集做五折交叉拟合：

```text
每折 800 个问题：设计系统
每折 200 个问题：测试系统
```

划分时先按下式的哈希排序：

```python
sha256(f"{seed}:{query.id}")
```

然后根据排序下标对 5 取模分折，固定 `seed=42`。五折合并后，每个问题作为测试问题出现一次。

每折的设计问题参与：

- 共访问图建立和区域划分。
- 区域访问频率、Base 表现等特征计算。
- probe 收益测量。
- LightGBM 训练和预算选择。

测试问题只参与评测和路由诊断，不参与上述设计决策。全量文档语料由各折共享，区域划分和收益学习则根据各折的设计问题重新完成。

> **数据前提：当前方法需要带正确证据标注的历史问题。**
>
> 只有问题文本、没有 `gold_doc_ids` 的普通访问日志，不能直接完成当前特征与 probe 标签计算。

实现位置：[warp/data/base.py](warp/data/base.py) 中的 `load_crossfit_bundles()`。

<a id="s4"></a>
## 4. 全量 Base：BM25、Dense 与 RRF

### 4.1 BM25 建立关键词检索能力

`BM25Retriever.fit()` 对所有文档内容进行小写化和正则分词，统计词频、倒排表、IDF 与平均文档长度。

默认参数：

```text
k1 = 1.5
b  = 0.75
```

直观上，查询词在某文档中越明显、在全语料中越稀有，对该文档的贡献就越大，同时校正文档长度影响。

### 4.2 Dense 建立语义检索能力

文档由 `nvidia/NV-Embed-v2` 编码成向量并缓存。查询使用官方 passage 检索对应的 query instruction 编码，再与文档向量计算余弦相似度。

**当前常规 Dense 检索是遍历向量的精确余弦检索。**后面给文档寻找语义邻居时才使用 FAISS HNSW。

Base 的 Dense 和 HippoRAG2 复用同一个 embedding 后端，使文档表示空间保持一致。

### 4.3 RRF 按排名融合两条列表

BM25 分数与向量相似度不直接相加，代码使用 Reciprocal Rank Fusion：

$$
\operatorname{RRF}(d)=\sum_{\ell}\frac{1}{60+\operatorname{rank}_{\ell}(d)}.
$$

文档不在某条列表中，就没有该列表的贡献。例如 BM25 第 1、Dense 第 10 的文档得到：

$$
\frac{1}{61}+\frac{1}{70}.
$$

`base.search(query, 50)` 实际执行：

```text
BM25 取最多 100 条 ─┐
                   ├→ RRF → 保留 50 条
Dense 取最多 100 条 ┘
```

### 4.4 Base 候选与 Base 最终结果不同

`base.search()` 到 RRF 为止，没有 CrossEncoder。

评测 Base 基线时，还会执行统一的最终排序路径：

```text
Base 50 条候选 → 统一融合函数 → CrossEncoder → 最终前 5 / 10 条
```

这一区别决定了后面哪些结果用于特征、哪些结果用于收益标签。

实现位置：[bm25.py](warp/retrieval/bm25.py)、[dense.py](warp/retrieval/dense.py)、[hybrid.py](warp/retrieval/hybrid.py)。

<a id="s5"></a>
## 5. 从历史问题建立文档共访问图

### 5.1 每个问题的 top-20 文档两两连边

对每个设计问题运行：

```python
base.search(query.text, k=20)
```

假设某题返回 4 篇文档，演示如下：

```text
[d1, d2, d3, d4]

形成文档对：
d1—d2  d1—d3  d1—d4
d2—d3  d2—d4  d3—d4
```

每对权重增加 1。另一道题再次共同召回 `d1`、`d2`，该边再加 1：

$$
w_{\mathrm{query}}(i,j)=
\sum_{q\in Q_{\mathrm{design}}}
\mathbb{I}[d_i,d_j\in \operatorname{Top20}_{\mathrm{Base}}(q)].
$$

注意：

- 按是否共同出现计数，不按文档名次加权。
- 连边没有直接使用 gold evidence。
- 共访问表示“共同被检索到”，不保证“共同构成正确证据”。
- 一个完整 top-20 列表最多产生 190 个文档对。

### 5.2 补充弱语义连接

为了给历史问题未覆盖的文档补充连接，代码复用文档向量建立 FAISS HNSW，给每篇文档找 5 个近邻，只保留相似度大于 0 的连接。

HNSW 默认参数为 `M=32`、`efConstruction=200`、`efSearch=128`。向量归一化后以内积近似查找余弦近邻。

近邻关系转为无向文档对；同一对被两个方向发现时保留最大相似度，不重复相加。

最终：

$$
w(i,j)=w_{\mathrm{query}}(i,j)+0.05\,w_{\mathrm{semantic}}(i,j).
$$

例如共同召回 3 次、相似度 0.8，权重为 `3.04`；只有语义连接则为 `0.04`。

查询共访问是主要信号，语义边提供平滑连接，但不能保证所有文档都连通。

输出的 `CoaccessGraph` 保留总边、查询边、语义边和逐题结果，便于后续特征和消融复用。

实现位置：[coaccess_graph.py](warp/partition/coaccess_graph.py)、[ann.py](warp/retrieval/ann.py)。

<a id="s6"></a>
## 6. 用 Leiden 将文档划分为区域

代码将共访问图转换为 igraph，调用：

```python
la.find_partition(
    graph,
    la.RBConfigurationVertexPartition,
    weights=weights,
    resolution_parameter=1.0,
    seed=42,
)
```

Leiden 根据连接结构形成社区，倾向于将内部连接较强的文档组织在一起。区域数量和大小不是预先固定的。

输出满足：

$$
r_i\cap r_j=\varnothing,\qquad \bigcup_i r_i=D.
$$

即每篇文档只属于一个区域，全量语料全部有归属。

正式配置 `min_region_size=5`。小于 5 的社区优先并入与它连接总权重最大的、大小已经达到阈值的社区。如果没有这种目标，回退到其他非空社区。因此不是所有边界情况都严格按最强相邻连接合并。

最终按稳定顺序生成 `r0000`、`r0001` 等 ID，并建立：

```python
region_map[region_id] = Region对象
doc_region[doc_id] = region_id
```

后续路由使用 `doc_region` 查表。

实现位置：[leiden.py](warp/partition/leiden.py)。

<a id="s7"></a>
## 7. 为区域计算九维构图前特征

### 7.1 先确定哪些问题访问了区域

对每个设计问题获取 Base 前 50 条候选，取其中前 20 条作为路由结果：

```python
candidates = base.search(query.text, 50)
routing_results = candidates[:20]
```

只要这些结果里有一篇属于区域 $r_i$，该问题就计入 $Q_i$。同一问题在同一区域只计一次，即使召回多篇区域内文档。

$$
F_i=|Q_i|.
$$

一个问题可以关联多个区域，因此所有区域的频率之和可以超过设计问题总数。

### 7.2 九维特征一览

| 特征 | 实际内容 | 作用 |
|---|---|---|
| `num_docs` | 区域文档数 | 描述规模 |
| `num_tokens` | 区域文本正则分词总长度 | 描述文本规模和成本 |
| `query_freq` | 路由命中的设计问题数 | 描述需求 |
| `base_recall` | 关联问题在 Base 前 20 条中的整题证据召回均值 | 描述 Base 表现 |
| `failure_rate` | 关联问题未找全证据的比例 | 描述改进空间 |
| `avg_retrieval_entropy` | 区域内候选分数归一化熵的均值 | 描述分数集中程度 |
| `multi_doc_rate` | 符合条件的关联问题中多文档问题比例 | 描述多文档需求 |
| `embedding_dispersion` | 区域内文档平均余弦距离 | 描述语义分散程度 |
| `coaccess_density` | 归一化区域内部共访问边权 | 描述共访问内聚程度 |

这些特征在区域知识图构建前即可获得；其中一部分需要设计问题的 gold evidence，因而“廉价”不意味着“无监督”。

### 7.3 Base recall 和失败率

若正确证据为 `{d1,d2}`，Base 前 20 条只包含 `d1`，recall 为 `1/2`。找全为 1，都没找到为 0，然后对 $Q_i$ 求平均。

**这里计算整道题的全部正确证据，不只计算区域内部证据。**

失败率对每题赋值：找全为 0，否则为 1。需要 3 篇却只找到 2 篇时，recall 为 `2/3`，failure 为 1；二者不总是互补。

没有 gold 的问题不会进入这两项的平均；没有可用值时特征取 0。probe 对 gold 的要求更严格，见后文。

### 7.4 检索熵

只取前 20 条中属于当前区域的候选分数，将非负分数归一化为 $p_j$：

$$
p_j=\frac{s_j}{\sum_t s_t},\qquad
H=-\frac{\sum_jp_j\log(p_j+10^{-12})}{\log m}.
$$

$m$ 为这些区域内候选的数量。分数相近时熵更高，少数候选占优时更低。只有一个区域内候选，或总分不为正时，不将该次访问加入熵均值；无可用项时取 0。

这里使用 RRF 分数，因此它是启发式特征，不是校准后的模型置信度。

### 7.5 多文档比例

先筛选“访问当前区域，且区域中包含该题至少一篇正确证据”的问题，再判断整题是否需要多篇证据。

正确证据跨两个区域、当前区域只包含其中一篇，该题仍然属于多文档问题。

### 7.6 向量离散度

$$
V_i=\operatorname{mean}_{a,b\in r_i}[1-\cos(\mathbf e_a,\mathbf e_b)].
$$

文档对数量不超过 4,096 时全部计算；更多时用依赖 seed 和区域 ID 的随机种子抽取 4,096 个不同文档对。没有文档对时取 0。

### 7.7 共访问密度

$$
C_i=\frac{\sum_{a<b,\ a,b\in r_i}w_{\mathrm{query}}(a,b)}
{\binom{|r_i|}{2}\,|Q_{\mathrm{design}}|}.
$$

代码对分母做至少为 1 的保护。分子只使用查询共访问边，不使用语义边。

最终输出三份数据：

```python
features        # 每个区域的九维特征
region_queries  # 每个区域关联的问题 ID
base_results    # 每个设计问题的 Base 50 条候选
```

实现位置：[features.py](warp/advisor/features.py)。

<a id="s8"></a>
## 8. 选择少量区域做真实 probe

probe 是实际试建和试检索，用来获得“区域特征 → 真实收益”的监督数据。

### 8.1 哪些区域有资格

只有 `query_freq > 0` 的区域能进入候选，因为需要关联设计问题来测量收益。

### 8.2 probe 数量

正式比例为 `0.2`，实际计算：

```python
count = min(
    eligible_region_count,
    max(6, round(eligible_region_count * 0.2)),
)
```

| 有访问的区域数 | probe 数 |
|---:|---:|
| 100 | 20 |
| 20 | 6 |
| 6 | 6 |
| 少于 6 | 报错 |

**20% 是区域数量比例，不是文档量或构图费用比例。**

### 8.3 具体采样过程

按以下键排序：

```python
(
    query_freq / num_tokens,
    failure_rate,
    region_id,
)
```

需要选 $p$ 个区域时，构造：

```python
buckets = [ordered[index::p] for index in range(p)]
```

再从每桶固定种子随机选一个。这里是排序后交错分桶，不是连续分位区间分桶。

选中后对整个区域构图，没有再在区域内部抽取少量文档来替代完整区域。

实现位置：[probe.py](warp/advisor/probe.py) 中的 `select_probe_regions()`。

<a id="s9"></a>
## 9. 一个区域图具体怎样建立

区域文档转换为官方 `Chunk`：

```python
Chunk(
    content=doc.content,
    source_id=doc.id,
    metadata={"warp_doc_id": doc.id, "title": doc.title},
)
```

构图器创建区域自己的 HippoRAG 实例，然后调用：

```python
rag.index(chunks)
```

知识抽取与图算法由锁定版本的官方 HippoRAG2 完成。WARP-G 适配层负责选择文档、管理独立索引、保存 ID、复用模型和计量成本，不在本仓库重新实现上游图算法。

### 9.1 什么共享，什么隔离

| 内容 | 区域之间的关系 |
|---|---|
| LLM 实例及相关共享调用资源 | 共享 |
| embedding 模型权重 | 共享 |
| 图实例 | 隔离 |
| chunk / entity / fact 向量存储 | 隔离 |
| 区域产物目录 | 隔离 |

两个区域都提到同一实体，不会自动形成跨区域的统一图节点。各区域可以各自返回候选，但当前没有跨多个区域图联合传播。

### 9.2 产物和身份映射

产物路径类似：

```text
outputs/indexes/hotpotqa/regions/r0000-<fingerprint>/
```

指纹基于区域、文档内容哈希、配置和后端版本等信息生成。manifest 保存文档集合、配置、构图成本和元信息。

`source_id` 保证图结果可以映射回 WARP 的文档 ID。适配层还检查重复内容、后端 API 和版本，并在非空区域抽取到零事实时报告错误。

当前锁定 HippoRAG 版本为 `2.0.0a4`，依赖 commit 为 `c617143f01477243992a63b2e2151cc003dd3b21`。

实现位置：[hipporag2.py](warp/graph/hipporag2.py) 中的 `HippoRAG2GraphBuilder.build()`。

<a id="s10"></a>
## 10. 怎样测量区域图的真实收益

对于区域 $r_i$，在关联问题 $Q_i$ 上比较两条检索路径。

### 10.1 两条路径保持最终排序方式一致

```text
Base 路径：
Base 50 条 → 统一融合 → CrossEncoder → 前 10 条

加入区域图：
Base 50 条 ───────────┐
                     ├→ RRF → 保留 50 条 → CrossEncoder → 前 10 条
本区域图最多 50 条 ────┘
```

图增强路径仍保留全量 Base 候选。它额外加入单个区域图候选，然后走同样的融合和重排。

### 10.2 主指标是 Complete Evidence@10

$$
M(q)=\mathbb I[\text{最终前 10 条包含全部 gold evidence}].
$$

若 gold 为 `{d1,d2}`：

| 最终前 10 条情况 | Complete Evidence |
|---|---:|
| 只有 d1 | 0 |
| 只有 d2 | 0 |
| d1、d2 都有 | 1 |
| 两篇都没有 | 0 |

probe 查询没有 gold evidence 时，指标函数报错。

### 10.3 区域标签是平均增量

$$
g_i=\frac{1}{|Q_i|}\sum_{q\in Q_i}
\left[M(q;\mathrm{Base}+G_i)-M(q;\mathrm{Base})\right].
$$

假设四道题：

| 问题 | Base | Base + 区域图 | 差值 |
|---|---:|---:|---:|
| q1 | 0 | 1 | +1 |
| q2 | 1 | 1 | 0 |
| q3 | 0 | 0 | 0 |
| q4 | 1 | 0 | −1 |

区域收益为 0。它同时改善了一题、损害了一题，净收益抵消。

负收益是允许的：新增候选可能挤掉原先的候选，CrossEncoder 也可能排序失误。

代码同时保存 Evidence Recall@10 的变化，但正式配置使用 Complete Evidence@10 的变化训练预测器。

> **特征与标签的阶段不同：**
>
> `base_recall` 特征看未重排 Base 前 20 条；probe 标签看统一融合与 CrossEncoder 后的最终前 10 条。

每个 `ProbeOutcome` 保存收益、两种指标的增量、前后效用、问题数和已构建图。

实现位置：[probe.py](warp/advisor/probe.py) 中的 `run()`。

<a id="s11"></a>
## 11. LightGBM 怎样学习和预测收益

每个 probe 区域贡献一条训练样本：

```text
X：该区域的九维特征
y：该区域的实测平均收益
```

probe 20 个区域，对应 `20 × 9` 的输入矩阵和 20 个标签。**学习单位是区域，不是问题或单篇文档。**

当前参数：

```python
LGBMRegressor(
    n_estimators=100,
    learning_rate=0.05,
    num_leaves=7,
    min_child_samples=2,
    random_state=42,
    verbosity=-1,
)
```

模型用回归树学习哪些特征组合对应较高收益，没有硬编码“高失败率一定值得构图”等规则。

预测所有区域：

$$
\hat g_i=f_\theta(\mathbf x_i).
$$

然后执行：

```python
predicted_gains.update(measured_gains)
```

因此已 probe 区域用实测值，其他区域用预测值。模型输出没有强制限制在理论收益区间，选择器只过滤非正收益。

模型只在离线设计中使用。在线问题不需要再经过 LightGBM。

实现位置：[predictor.py](warp/advisor/predictor.py)。

<a id="s12"></a>
## 12. 怎样在预算内选择区域

### 12.1 先估成本，再选择

构图前成本代理为：

$$
c_i=\sum_{d\in r_i}\operatorname{len}(\operatorname{tokenize}(d)).
$$

`tokenize()` 是正则分词，所以这是文本长度代理，不是精确 LLM token 或美元账单。

给定预算比例 $b$：

$$
B=b\sum_i c_i.
$$

总代理成本为 1,000,000、预算比例 0.2，则允许选择的代理成本和最多为 200,000。

### 12.2 排序公式

$$
s_i=\frac{F_i\max(\hat g_i,0)}{c_i}.
$$

| 因素 | 含义 |
|---|---|
| $F_i$ | 多少历史问题访问该区域 |
| $\hat g_i$ | 访问该区域的问题平均能改善多少 |
| $c_i$ | 获得改善需要多大投入 |

收益乘频率，是因为收益标签是对 $Q_i$ 的条件平均。例如 100 道题平均提升 0.1，比 10 道题平均提升 0.1 对整体工作负载更有价值。

若换算到全部历史问题上的平均贡献，可以再除以设计问题总数；该分母对所有区域相同，不影响排序。

### 12.3 一个选择例子

| 区域 | 频率 | 收益 | 成本 | 分数 |
|---|---:|---:|---:|---:|
| A | 100 | 0.10 | 1,000 | 0.010 |
| B | 20 | 0.30 | 1,000 | 0.006 |
| C | 80 | 0.10 | 4,000 | 0.002 |
| D | 100 | −0.02 | 500 | 0 |

预算 2,500 时先选 A、B，C 放不下，D 收益非正。最终剩余预算 500。

选择伪代码：

```python
for region in ranked_regions:
    if gain[region] <= 0:
        continue
    if spent + cost[region] <= budget:
        selected.append(region)
        spent += cost[region]
```

分数相同优先低成本，再按区域 ID。放不下的区域跳过后继续扫描，不会直接结束。

### 12.4 这个选择器的实际性质

- 使用收益成本比贪心，不是精确背包求解。
- 不保证全局最优。
- 区域不可拆分，预算可以剩余。
- 预算为 1 也跳过非正收益区域，因此不一定建立全部区域图。
- 频率为 0 的区域分数为 0；代码没有额外将其一律排除。如果预测收益为正且预算仍够，它仍可能在排序末尾被选中。

实现位置：[selector.py](warp/advisor/selector.py)、[hipporag2.py](warp/graph/hipporag2.py) 的 `estimate_cost()`。

<a id="s13"></a>
## 13. 区域交互分析做了什么

单独加入 A 的收益与单独加入 B 的收益，不一定能够直接相加。

互补例子：A 找到第一篇证据，B 找到第二篇，两张图一起用才能找全。冲突例子：两张图引入大量干扰候选，挤占其他证据位置。

代码在 probe 区域中选择区域对，最多抽取 30 对，在两个区域关联问题的并集上计算：

$$
I_{ij}=U(\{i,j\})-U(\{i\})-U(\{j\})+U(\varnothing).
$$

所有 $U$ 都使用同一批问题、相同在线检索规则下的平均效用。

**该量只用于诊断，没有进入选择公式。**选中 A 后，代码不会重新估算 B 的条件边际收益。

实现位置：[pipeline.py](warp/pipeline.py) 的 `_probe_interactions()`。

<a id="s14"></a>
## 14. 选中区域怎样物化和复用

物化就是为选中区域实际建立并保存图索引：

```python
for region_id in selected_regions:
    if region_id not in self.graphs:
        self.graphs[region_id] = builder.build(...)
```

probe 已建的图可以复用，未建的才新建。

必须区分：

| 对象 | 含义 |
|---|---|
| `self.graphs` | 当前对象已经建立或缓存的图，包括 probe 和此前其他预算构建的图 |
| `selected_regions` | 当前方法、当前预算允许使用的图 |

正式评测显式传入选中集合，所以没有选中的缓存图不会参与该次评测。

但直接调用 `search()` 且省略 `selected_regions` 时，默认使用 `self.graphs` 中所有图。单独调用 API 时需要显式传入预算选择集合，才能保持该预算的检索范围。

实现位置：[pipeline.py](warp/pipeline.py) 的 `materialize()` 和 `evaluate()`。

<a id="s15"></a>
## 15. 新问题到来后的完整在线流程

### 15.1 Base 获取全量候选

```python
base_results = base.search(query, 50)
```

### 15.2 根据前 20 条候选查表路由

假设选中集合和本题触达区域分别为：

```text
S    = {r0001, r0003, r0008}
A(q) = {r0001, r0002, r0008}
```

本题调用交集：

```text
{r0001, r0008}
```

路由就是“文档 ID → 区域 ID → 判断选中 → 去重”，WARP-G 没有用 LLM 做区域路由。

### 15.3 每个被激活区域执行图检索

适配器调用：

```python
rag.retrieve(queries=[query], num_to_retrieve=50)
```

接口注释描述的上游路径包含 query-to-fact、fact filtering、个性化和 PPR。内部算法由锁定的官方后端执行。

返回文档必须带 `source_id`，并且属于该区域，否则适配器报错。

代码按区域逐个调用图检索。前 20 篇路由文档最多涉及 20 个不同区域，但没有额外设置更小的图调用数量上限。

### 15.4 合并候选并统一重排

```text
Base 50 条 ──────────┐
区域图 1 最多 50 条 ───┼→ RRF → 总计最多 50 条 → CrossEncoder → 前 10 条
区域图 8 最多 50 条 ───┘
```

各列表默认等权。BM25 和 Dense 已在 Base 内融合，外层把 Base 当作一条列表。

CrossEncoder 为 `BAAI/bge-reranker-v2-m3`，固定 revision，`batch_size=32`、`max_length=512`。它对问题与候选文档内容成对打分，再生成最终排名。

CrossEncoder 无法找回从未进入候选集合的文档。

### 15.5 没有图可调用时

如果路由区域与选中集合没有交集，就只对 Base 候选执行相同的最终排序。

未构图区域仍可通过 BM25 和 Dense 检索；但已构图区域如果未被 Base 前 20 条触达，也不会被当前问题调用。

实现位置：[pipeline.py](warp/pipeline.py) 的 `search()`、[reranker.py](warp/retrieval/reranker.py)。

<a id="s16"></a>
## 16. 用一个多跳问题串起全过程

假设语料有：

```text
d1：电影《远行》由导演甲执导。
d2：导演甲出生于城市乙。
d3：电影《远行》的上映时间和演员信息。
```

问题为“《远行》的导演出生在哪里？”，gold 为 `{d1,d2}`。

1. 历史问题产生文档共访问关系，语义近邻补充连接，Leiden 可能将 d1、d2 分到同一区域。这里没有 gold 强制保证一定分在一起。
2. 区域特征记录规模、访问次数、Base 是否经常缺证据等信息。
3. 若该区域被 probe，系统真实建立区域图。
4. Base 最终前 10 条只有 d1 时，Complete Evidence 为 0。
5. 加入区域图后，如果通过相关事实和图传播把 d2 带入候选，且最终 d1、d2 都在前 10 条，指标变为 1。
6. 本题对区域收益贡献 +1，再与其他关联问题的差值一起求平均。
7. LightGBM 学习这些区域特征与收益的关系，预测未 probe 区域。
8. 预算选择器决定哪些区域值得部署。
9. 新问题的 Base 前 20 条只要命中已选区域里的某篇文档，就能激活整个区域图，无需已经找全正确证据。

Base 提供进入相关区域的入口，区域图补充证据候选。如果 Base 完全未触达该区域，该图不会被激活。

<a id="s17"></a>
## 17. 从检索证据到生成答案

reader 评测默认取最终前 5 篇证据：

```text
检索结果前 5 篇 → 固定 QA prompt 和 LLM → 答案 → EM / F1
```

配置使用 `gpt-4o-mini-2024-07-18`。各检索方法共享 reader，以保持生成器一致。

工程上复用 Full Graph 对象中的 QA 组件，但传入文档由当前被比较的检索方法提供，不会因此自动加入 Full Graph 的检索证据。

预测器训练用 Complete Evidence@10，reader 评测看答案 EM/F1；前者改善并不保证后者必然改善。

多答案问题分别计算与各标准答案的得分，取最佳 EM/F1。

实现位置：[reader.py](warp/eval/reader.py)、[qa.py](warp/eval/qa.py)。

<a id="s18"></a>
## 18. 预算和实际成本怎样核算

### 18.1 需要区分的成本

| 口径 | 含义 |
|---|---|
| 选择时成本 | 区域文本正则分词长度 |
| 部署成本 | 当前选中区域图的构建成本之和 |
| probe 成本 | 获取标签所试建区域图的成本 |
| 在线图检索成本 | 区域图调用中的 token 和耗时 |
| reader 成本 | 生成答案的 token 用量 |

**20% 代理预算不等于 20% 的真实支出。**真实知识抽取输出、embedding 工作量和缓存情况都会影响成本。

### 18.2 构图后记录什么

- LLM 输入与输出 token。
- embedding 文本的 tokenizer 估计量。
- 构图耗时、节点数、边数、存储量。
- 按配置单价折算的 LLM 美元费用。

embedding token 是对写入 chunk、entity、fact store 的文本使用 `cl100k_base` 计数得到的估计。美元字段不代表已包含全部本地 GPU 等成本。

逻辑 token 包含缓存命中的调用用量；物理 token 记录未命中缓存的用量。manifest 复用时可保留原有构建成本记录，避免直接将复用索引解释成零构建成本。

### 18.3 首次成本字段的当前实现

`run.py` 对 WARP-G 使用：

```python
first_run_cost = deployment_cost + probe_cost
```

probe 图后来被选中时，实际对象可以复用，但这个字段没有对两个集合的重合部分去重。因此 `first_run_cost_including_probe` 不能直接当成去重后的首次实际付费总额。

该字段也没有自动包含 probe 检索等所有设计计算开销。当前 `design_search_cost` 字段保存的是 probe 构图成本；名称不意味着所有设计阶段动作都已汇总进去。

`online_retrieval_cost` 来自图检索器统计，不包含完整 Base 和外层 CrossEncoder 时间，不能当成端到端响应延迟。

正式 runner 在预算循环之前完成 `fit()`，所以即使某个预算点是 0，当前整套实验也已经进行 probe。零预算表示该点在线使用零区域图，不意味着整个设计过程零构图开销。

### 18.4 实测成本曲线

runner 以真正 Full Graph 的输入、输出和 embedding token 总量作为分母，计算部署的 `actual_cost_fraction`。这与选择时使用的文本代理预算是两种口径。

公共 Base 始终覆盖全量语料，不计入选择性图预算；它本身并不免费。

实现位置：[run.py](warp/run.py)、[hipporag2.py](warp/graph/hipporag2.py)、[construction_cost.py](warp/eval/construction_cost.py)。

<a id="s19"></a>
## 19. Full Graph、对照实验与验证

### 19.1 所有区域图不等于 Full Graph

```text
全部区域图：区域 1 → 图 1；区域 2 → 图 2；……
Full Graph：全量文档 → 一张全局图
```

全局图可以保留跨区域连接，独立区域图之间没有联合传播。当前代码为 Full Graph 单独构建全语料索引。

`full_graph` 对照融合 Base 与全局图候选；`hipporag2` 对照只使用全局图候选。当前评测两者都经过共享 CrossEncoder。

### 19.2 区域选择器消融

| 方法 | 规则 |
|---|---|
| Random-region | 固定种子随机顺序 |
| Frequency-only | 高频优先 |
| Gain-only | 高收益优先，跳过非正收益 |
| Cost-only | 低成本优先 |
| WARP-G | 频率 × 正收益 ÷ 成本 |

它们共享区域和预算框架，用于区分频率、收益和成本信号的贡献。

### 19.3 分区消融

- `combined`：查询边与弱语义边共同分区。
- `query`：只使用查询共访问边。
- `semantic`：只使用语义边。
- `random`：先获得 combined 分区的区域大小，再打乱文档，按同样大小切分。

随机分区因此保留参考分区的区域大小分布。

### 19.4 收益模型验证

区域留一验证每次留出一个 probe 区域，用其他 probe 训练，再预测它，计算 MAE、RMSE 和 Spearman。

特征消融每次将一项特征在全部区域置零，再进行区域留一验证。学习曲线在已有 probe 标签中抽取不同训练规模，重复评估误差，不会为了学习曲线再构建新的区域图。

### 19.5 最终评测

检索指标包括 Evidence Recall@5/10、Complete Evidence@5/10；reader 指标包括答案 EM/F1。runner 合并五折 held-out 问题，并输出置信区间、配对显著性分析和质量—成本曲线等结果。

正式实验还包括 KET-RAG、G2ConS 等对照。是否改善未见问题，必须根据最终测试结果判断，不能只根据公式或预测器训练拟合得出结论。

实现位置：[pipeline.py](warp/pipeline.py)、[predictor.py](warp/advisor/predictor.py)、[run.py](warp/run.py)。

<a id="s20"></a>
## 20. 参数、调用链和代码阅读顺序

### 20.1 正式配置中的关键参数

| 参数 | 值 | 含义 |
|---|---:|---|
| `coaccess_k` | 20 | 建共访问图时每题使用的文档数 |
| `routing_k` | 20 | 从 Base 候选读取多少篇文档来确定区域 |
| `candidate_k` | 50 | Base、区域图候选深度及融合后候选上限 |
| `retrieval_k` | 10 | probe 默认最终检索深度 |
| `semantic_k` | 5 | 每文档语义近邻数 |
| `semantic_lambda` | 0.05 | 语义边权系数 |
| `min_region_size` | 5 | 小社区合并阈值 |
| `partition_resolution` | 1.0 | 未在 YAML 覆盖时使用的 Leiden 默认参数 |
| `probe_fraction` | 0.2 | 有访问区域的 probe 比例，至少 6 个 |
| `benefit_objective` | `complete_evidence` | 收益学习目标 |
| `dispersion_pairs` | 4096 | 离散度最多使用的文档对数 |
| `interaction_pairs` | 30 | 最多检查的 probe 区域对数 |
| `seed` | 42 | 固定种子 |
| 预算点 | 0、0.1、0.2、0.4、0.6、1.0 | 文本代理成本比例 |
| reader `top_k` | 5 | 生成答案使用的证据数 |

图后端另外配置 `retrieval_top_k=200`、`linking_top_k=5`、`damping=0.5`、`passage_node_weight=0.05` 和同义连接参数等。这些是后端参数，不应与 WARP 外层的 50 条候选上限混淆；外层图检索明确传入 `num_to_retrieve=50`。

参考：[configs/paper/hotpotqa.yaml](configs/paper/hotpotqa.yaml)。其他三个正式配置也使用上述核心 WARP 参数。

### 20.2 哪些组件学习参数

| 组件 | 当前流程中的角色 |
|---|---|
| BM25 | 统计语料词频等信息 |
| NV-Embed-v2 | 使用既有模型，不在此流程微调 |
| Leiden | 根据图结构划分区域 |
| LightGBM | 用 probe 数据训练收益回归模型 |
| 预算选择器 | 固定贪心规则 |
| WARP 路由 | 固定查表规则 |
| HippoRAG2 | 调用既定构图与检索流程 |
| CrossEncoder | 使用既有模型，不在此流程微调 |
| QA LLM | 使用既有模型生成答案 |

### 20.3 核心调用链

```text
warp.run._build_model()
    └── WARPG.fit(bundle)
        ├── base.fit(documents)
        ├── CoaccessGraphBuilder.build()
        ├── RegionPartitioner.partition()
        ├── RegionFeatureExtractor.extract()
        ├── graph_builder.estimate_cost()
        ├── RegionProber.select_probe_regions()
        ├── RegionProber.run()
        ├── BenefitPredictor.fit()
        ├── BenefitPredictor.predict()
        ├── 用实测收益覆盖 probe 区域预测
        └── _probe_interactions()

WARPG.select(budget_fraction)
    └── BudgetSelector.select()

WARPG.evaluate(queries, selected_regions)
    ├── materialize(selected_regions)
    └── 对每题调用 search(query, k, selected_regions)
        ├── Base 取候选
        ├── 查文档归属并取选中区域交集
        ├── 调用各区域图
        ├── RRF 融合并截断
        └── CrossEncoder 重排
```

### 20.4 建议阅读顺序

1. [models.py](warp/models.py)：认识数据对象。
2. [pipeline.py](warp/pipeline.py)：先读 `fit()`、`select()`、`search()`，把握主线。
3. [hybrid.py](warp/retrieval/hybrid.py)：理解 Base 与两层 RRF。
4. [coaccess_graph.py](warp/partition/coaccess_graph.py) 和 [leiden.py](warp/partition/leiden.py)：理解区域来源。
5. [features.py](warp/advisor/features.py)：理解九维输入。
6. [probe.py](warp/advisor/probe.py)：理解真实收益标签。
7. [predictor.py](warp/advisor/predictor.py) 和 [selector.py](warp/advisor/selector.py)：理解预测与预算决策。
8. [hipporag2.py](warp/graph/hipporag2.py)：理解后端调用、缓存和成本。
9. [run.py](warp/run.py)：理解整套实验和结果字段。

阅读时优先追踪五份状态：`regions`、`features`、`probes`、`predicted_gains`、`selected_regions`。

<a id="s21"></a>
## 21. 容易混淆的实现细节

| 容易产生的理解 | 当前实际行为 |
|---|---|
| 先建全量知识图，再切分 | 先建廉价文档共访问图，再决定哪些区域建立知识图 |
| 区域等于固定长度文本块 | 文本块是 Document，Region 是多篇 Document 的集合 |
| 普通无标签查询日志就够 | 部分特征和 probe 收益需要证据标签 |
| Base 检索已经重排 | `base.search()` 只到 BM25 + Dense 的 RRF |
| 特征和收益都看相同 top-k | 特征看未重排 top-20，收益看最终 top-10 |
| probe 是区域的一小部分文档 | 选少量区域，每个区域完整构图 |
| 20% probe 等于 20% 费用 | 它是有访问区域的数量比例，并有至少 6 个的下限 |
| 20% 预算等于 20% 实际账单 | 预算使用文本长度代理，实际费用另计 |
| 预算选择保证最优 | 当前是静态收益成本比贪心 |
| 选择会考虑区域互补 | 交互仅用于诊断，没有进入选择公式 |
| 试建过的图都会用于评测 | 正式评测只使用当前显式选中集合 |
| 在线使用 LightGBM 选择图 | 在线按 Base 文档归属查表，模型只用于离线收益预测 |
| 选中图每题都调用 | 还必须被本题 Base 前 20 条触达 |
| 未构图文档不可检索 | 全量 Base 仍然覆盖这些文档 |
| 所有区域图等于全局图 | 独立区域图没有全局图的跨区域传播 |
| 首次成本字段就是去重实付 | 当前部署成本加 probe 成本，没有对重合图去重 |
| 在线图检索耗时就是端到端延迟 | 不包含完整 Base、外层重排和 reader 耗时 |

还有一个更细的候选深度差异：建共访问图直接调用 `base.search(q, 20)`，内部两路各取 40 条；特征与在线路由调用 `base.search(q, 50)` 后取前 20 条，内部两路各取 100 条。RRF 的输入深度不同，因此这两个“前 20 条”不保证完全相同。当前路由诊断也直接调用 `base.search(q, routing_k)`，与实际在线候选深度存在这一差异。

这些区别不改变主流程，但会影响对特征、路由诊断和成本结果的精确解释。
