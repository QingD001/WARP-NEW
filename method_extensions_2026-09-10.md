# 三项方法改进：实现、实验口径与限制

本次改动接续 measured_advisor_2026-09-09.md，替换其中“条件增益、多步检索尚未实现”的描述。
LightGBM 仍已删除。以下描述代码实际行为，尚无真实数据实验支持效果提升。

## 1. 稀疏收益：部分补证与整题完成共同提供信号

新增共享设计目标：`U = (1 - complete_weight) * EvidenceRecall + complete_weight * CompleteEvidence`。
默认 complete_weight=0.5，同时保留 evidence_recall、complete_evidence 两种纯目标作为对照。
两篇 gold 只找到一篇时，ER=0.5、CE=0、混合 U=0.25；补齐第二篇后 U=1。
因此部分补证有连续信号，整题完成另有奖励。收益始终为同题干预后的 U 减干预前 U，丢失证据记负值。
逐题、逐区仍分别保存 ER、CE 和 utility_gain，最终报表仍使用 ER、CE、Reader EM/F1，不用混合分掩盖退化。

区域内 gold 的缺失比例用于便宜的探测优先级。设计目标使用整题证据，避免某个区域内召回提升却挤掉其他区域
必要证据的情况被误计为成功。单区估计保留零先验收缩 n/(n+16)，这不是显著性保证。
权重 0.5 与收缩强度 16 都是待验证的默认值；不能在测试集上选出最高分组合后作为无偏结果汇报。

## 2. 条件边际收益与双区试探

selection_mode=conditional 为新默认 WARP 选区方式；independent 保留上一版收益/成本排序用于消融。
候选限定为已探测建图、在当前预算内可负担的区域。按旧估计收益/成本优先截取至多 6 区，
并保留其中零/负单区收益候选参与联合测量，而不是预先删除。

从整个 train/design workload 均匀抽取至多 32 题，对所有候选集合用相同问题测量：

`Δ(A | S) = Σ_q [U(retrieve(q, S∪A)) - U(retrieve(q, S))] / (n+16)`

S 为当前已选区域，A 为一个区域或有限数量的双区域组合。每轮选正净收益/新增估算成本最高的 A。
因为同一批全局 workload 问题已经包含访问频率和旁路损失，不再额外乘一次区域访问频率。
以 CE 为目标时，两个区域单独都零增益但一起补齐证据，能够通过双区提案入选。
每轮都重新评估相对于当前 S 的收益，允许已选区域改变下一轮的最优选择。

边界：每预算最多 3 轮，每轮最多 6 个双区提案，每预算最多实际测量 16 个不同集合（包含 Base）。
每集合最多 32 题，最多检索 retrieval_steps 步；不新增探测图。同一次预算选择内缓存集合结果，
同一模型同一预算再次 select 不重复检索。不同预算独立测量和计费，不假设跨预算复用。
零预算或无可负担候选直接返回 Base，不产生条件测量调用。

这些上限防止无界枚举，但意味着不保证找到全局最优组合：候选截断、双区截断、低探测预算、三阶互补
都可能导致漏选。不能把上限触发后的“没有选更多区域”解释为其余区域确实没有收益。
报告保存 sample_query_ids、candidate_regions、全部评估提案逐题净变化、每轮选择、评估集合数及上限触发标记；
原始 JSONL 另存每个实际测量集合的检索文档、ER、CE 和效用。

## 3. 多步证据反馈检索

paper 配置默认 retrieval_steps=2，WARPConfig 通用默认仍为 1，便于显式建立单步对照。
每步调用同一个底层检索通路。第一步使用原问题；后续步从累计最终排序中选择至多 2 篇尚未反馈的文档，
每篇最多 400 个字符，将内容拼接到原问题后作为下一轮检索查询。
每步区域路由随查询重新执行，但始终只允许显式选中的区域图参与。
累计各步候选以 RRF 融合，并始终按原问题最终重排。第一步复用底层已完成的排序，不重复 CrossEncoder。
无新反馈文档时提前停止，否则在步数上限停止。没有 gold 访问，没有基于测试标签的停止规则。

这是确定性的 passage-feedback retrieval，不是 IRCoT 的 LLM 推理链与子问题生成。
它不增加查询生成的 LLM 调用，但增加 Base 检索、可能的图检索与重排开销。反馈片段可能带来查询漂移，
特别是实体不在开头截断片段内、第一轮命中错误文档时；是否有效必须由单步/两步对照判断。

Probe、条件设计、测试阶段共享同一实际检索函数。BM25、Dense、Hybrid、所有区域方法、Base+FullGraph、
HippoRAG graph-only 及本仓库 KET-RAG/G2ConS 统一应用相同步数和反馈上限。
独立运行的作者官方 baseline 脚本不受此包装影响，不能将其结果直接标为本多步协议下的匹配对照。

每题保存每步查询、反馈文档、返回文档、路由、候选、累计最终文档、实际步数和停止原因。
评测阶段在检索完成后追加每个 k 下的 ER/CE、新增/丢失 gold、进入候选却未进入最终结果的 gold。
这些 gold 诊断仅用于事后分析，检索与反馈模块不接收 Query.gold_doc_ids 或参考答案。
已有后端 graph_seeded_calls / dense_fallback_calls 继续记录；routing_diagnostics 仍描述初始 Base 路由，
多步路由应查看每题步骤轨迹。离线区域审计兼容多步聚合轨迹，并计入重复区域调用次数。

## 成本归属与 Reader

WARP 的 design_search_cost 包含探测构图、探测检索和当前预算的条件选区检索。
Gain-only 只承担它所使用的探测成本；Random/Frequency/Cost 不承担条件选区成本。
计算 test online_retrieval_cost 的起点移到 select 之后，避免把条件设计算进测试在线开销或重复收费。
条件测量 wall time 扣除已计入检索的部分后纳入方法设计时间。partition ablations 同样保存新增设计成本。

10% 探测预算仍是构图成本 proxy 上限，不包含新增条件测量，也不是实际 API tokens 硬限额。
最大集合数限制的是检索工作量，不能换算为严格 tokens 上限。不同方法步数相同也不代表实际图调用数或 tokens 相同，
仍需同时比较实际成本。Raw JSONL 保存实际多步过程及调用记录。
Reader 继续使用测试检索已保存的最终文档列表，不为回答重新检索。

## 可复现消融

已生成 configs/ablations/hotpotqa 下 12 份配置：3 种目标 × 2 种选区方式 × 1/2 步检索。
各配置共享数据、fold、seed、探测预算、部署预算及 Reader 口径，关闭额外 partition ablations 以免扩增无关成本。
生成其他数据集配置：

```sh
python scripts/make_method_ablations.py --config configs/paper/musique.yaml --output-dir configs/ablations/musique
```

每个配置必须用独立 output/checkpoint 目录，例如：

```sh
python -m warp.run --config configs/ablations/hotpotqa/mixed-conditional-2step.yaml --output outputs/method_ablations/hotpotqa/mixed-conditional-2step.json
```

默认 checkpoint 由 output 路径派生，配置签名不匹配时会拒绝复用旧结果。图内容/后端配置不变时可安全复用构图缓存；
逻辑设计成本仍需计入。没有自动启动这 12 组真实 API 实验。

优先做三组配对：固定单步和独立选择比较目标；固定目标和步数比较选区；固定目标和选区比较步数。
再检查交互项。需按 query/fold 对齐比较净改善/退化题数及区间，不能仅凭均值上升或相同断言成功或有 bug。

## 验证

41 项 CPU 单元及模拟集成测试通过。新增覆盖混合目标、负收益、联合零单区增益、设计样本隔离、候选预算、
集合评估上限、选择缓存、成本归属、第二跳发现、原问题重排、动态路由、未部署图隔离、提前停止及真实 fit/probe 路径。
完整 runner 模拟测试已启用条件选区和两步检索，覆盖多个方法/预算、Reader、导出、原始日志、快照及 checkpoint 恢复。
实际 GPU/FAISS/HippoRAG 和付费 LLM 端到端实验尚未运行，不保证真实收益。
