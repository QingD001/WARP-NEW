# 原始记录、Reader 复用、局部特征与设计成本

## 本次实现

主实验默认保存 `<output>.folds/inputs/<signature>/` 输入快照与 `raw/fold-<fold>-<attempt>.jsonl` 逐事件日志。最终 JSON 的 `raw_artifacts` 给出路径。输入快照包含规范 corpus/queries、manifest、manifest 能定位的原始来源文件、完整 Document/Query、fold 分配、配置、版本和项目源码。HippoRAG 原有索引/抽取 artifacts 继续保存在 graph.artifact_root，构图事件记录其路径。

事件包括逐题 query/gold、各路候选和分数、路由、RRF 截断前后、CrossEncoder 全候选分数、图返回原始文档/metadata/事实种子、LLM 消息/原始响应/usage/cache_hit、probe 逐题净增益、构图成本、Reader 输入和原始输出。LLM 日志不记录 API key 或认证 kwargs。记录在计算过程中逐条追加，不等整个 fold 结束；已完成 fold 可恢复，未完成 fold 的事件保留但尚不支持自动逐题续算。未启用 IRCoT，现有日志记录当前实际调用，不能恢复历史实验未保存的原始响应。

Reader 在对应方法、指定 reader budget 的检索完成后立即运行，按 query ID 复用保存的 Top-k 文档，不再次查询图。QA 使用已有 probe 图对象提供的同一套 QA LLM/config；它只读取给定文档，无需为了 Reader 提前构建 Full Graph。Reader-only 方法/预算会单独检索一次并保存。普通检索和 QA 的费用分别统计。

## 特征定义

令 Q_r 为 Base 路由访问区域 r 的设计查询，G_q 为整题 gold，G_qr = G_q ∩ r，T_q 为 Base 前 routing_k 的文档集合。

现在 `base_recall` = mean(|G_qr ∩ T_q| / |G_qr|)，`failure_rate` = mean(1[G_qr 不包含于 T_q])，`multi_doc_rate` = mean(1[|G_qr| > 1])。这三个平均只在 Q_r 中确有局部 gold 的题上计算；无样本时数值设零，并用 `gold_query_rate` 显式表示覆盖程度。

旧整题定义保留为 global_base_recall、global_failure_rate、global_multi_doc_rate。新增 cross_region_gold_rate 表示局部有 gold 的题中仍有区域外 gold 的比例。feature_schema 为 region_local_v2_with_global_context，共十四维；旧预测器需要重训，旧 checkpoint 会因源码哈希变化被拒绝。

局部特征仍测量廉价 Base 路由深度的表现，不等同于最终 CrossEncoder Top-10 表现。它们是预测特征，不能称为已经观测到的局部构图增益。

例：gold={A∈r1,B∈r2}，Base 已找到 A 但没找到 B。旧代码把两个被访问区域都标记为整题失败；新代码 r1 的局部 recall=1/failure=0，r2 的局部 recall=0/failure=1。每区只有一个 gold，因此 local multi_doc_rate=0，global_multi_doc_rate=1。

## 为什么没有直接把收益目标换成局部 Recall

局部召回是有用的辅助信号，但独立区域之间的 CE 互补、跨区域证据丢失、区域大小和局部 gold 数差异会影响最终收益。区域内新增一篇 gold，同时挤掉区域外一篇 gold，可以局部变好、整题不变甚至变差。

因此当前部署选择仍使用 probe 的整题净增益；新增逐题 local_recall_gain、new_gold、lost_gold 供诊断。query_freq × 条件平均整题增益仍对应访问工作负载上的贡献近似。如果改为只在有局部 gold 的题上估计收益，却继续乘全部访问频率，会高估收益。

后续可以在设计/验证集比较 CE 与 ER 的混合净增益，局部收益作为辅助目标。局部收益不要直接替代最终 CompleteEvidence；也不要用测试集选择权重。

## 用户成本表的计算

原表及重算指标保存于 `user_cost_table_2026-09-09.json`，来源是用户提供的数字，未与真实 run artifact 对齐。

- WARP 部署 tokens：688,931。
- WARP 含设计 tokens：23,775,735。
- 差额：23,086,804，为部署成本的 33.51 倍，占总量 97.10%。
- Gain-only 两列相同，但当前算法使用同一收益预测器，不能在端到端比较中把学习收益成本记零；前一轮代码已修复。

若表来自最初版本 runner，含设计差额对应 `aggregate_costs(probe graphs)`，即 probe 图的构建成本（input+output+embedding），而不是 LightGBM 训练 tokens，也不包含完整的 probe 查询/交互分析成本。不能把差额直接解释成 IRCoT/多轮推理开销，当前根本没有启用 IRCoT。

旧 probe 数量为 max(6, round(eligible_regions × 0.2))：若有 10 个 eligible regions，会抽 6 个，即 60%；若有 20 个，会抽 6 个，即 30%。即使区域数量比例真是 20%，大区域的 token 比例仍可能远高于 20%。设计时构建的大区域可能最终一个都未选入部署。

旧首次计费还可能重复计算已部署 probe 图；但重复部分最多为部署图成本。单靠去重不能解释或消除约 2,300 万 tokens 的差额。

## 已采取的成本控制

1. 四份 paper YAML 设置 probe_max_queries=64，按固定 seed 在每区 routed 设计题中均匀抽样；保存完整 eligible 数量、抽中 query ID、每题结果和 usage。Base 重排结果在 probe 区域间复用。
2. 四份 YAML 设置 interaction_pairs=0。旧二阶分析对每个区域对的查询分别评测 Base、左区、右区、联合，最多会产生每题四次区域图调用；它只用于诊断，不影响当前 selector。若启用，单独记录 analysis usage，不再混入收益学习的必要成本。
3. 增加可选 probe_budget_fraction：约束 probe 区域的预计构图成本，而不是仅约束区域个数。选择时预留至少六个可用于当前预测器训练的区域；预算连最便宜六区都覆盖不了，会在构图前报出最低 proxy cost，要求细分区域或明确增加预算。
4. 此预算默认不设置硬数值，因为尚未观测实际区域成本，任意 0.2 可能无法支持六个区域。可在配置中设置 `probe_budget_fraction: 0.2` 验证可行性。它约束预计 token proxy，不是 API 实际费用的严格上限。

限制每区查询数主要减少 probe 检索开销，不能减少已构建区域的图成本；关闭 interaction 也不会直接消除旧表那 2,300 万构图 tokens。真正控制这部分需要成本预算、小区域/子图探测或可复用的抽取缓存。采样更少还会增加收益标签噪声，64 是可配置的起点，不是精度保证。

## Efficiency 的问题

原表与 CE / tokens 一致，例如 0.140/688931≈2.03e-7。此指标把 Base 已完成的题也归功于增量图；Cost-only 的 0.135 与 Hybrid 相同，却能因成本小而显得非常高效。

新结果新增 incremental_efficiency：相对 Hybrid 的 ΔCE、净多完成题数，以及每百万部署/首次 tokens 的净新增完成题数。WARP 相对 Hybrid 是 ΔCE=0.005，即 200 题净多一题。零成本处不做除法，保留 null；负增益不截断为零。

应一起报告绝对质量、修复与损伤题数、质量—实际成本曲线、预算利用率。部署和学习是一次性开销，online retrieval、Reader 是随服务查询量增长的开销。input/output/embedding tokens 单价与计算代价不同，token 总和仅是约定指标，不能等同现金费用。

总成本可表示为 C_design + C_deploy + Q × (C_retrieval + C_reader)。不能只说设计成本可摊薄就断言最终更便宜：如果 WARP 一次性成本已高于对照，且在线每题成本也不低，增加查询量不会自动产生交叉点。

## 后续方法修改的优先顺序

1. 先从日志确认新增 gold 的来源和流失位置，以及 Dense fallback 比例。没有新增证据时不要先优化收益回归器。
2. 给 probe 独立预算，并让物化单元有最大 token 大小；当前只有最小区域规模，可能无法用小预算购买有价值的部分。
3. 用渐进采样代替所有区域固定消耗：先少量题，优先补测可能入选且收益不确定的区域。只在设计数据上做采样决策，保留独立验证。
4. 用更密集的整题证据净增益辅助稀疏 CE；局部特征指出瓶颈，最终整题指标负责验收。十四维特征不意味着六个 region 标签足够，应比较简单平滑基线与 LightGBM，避免小样本过拟合。
5. 如果 gold 不在候选中，再测试统一两轮检索/IRCoT，并允许每轮重新路由到已部署区域。Hybrid、Random、WARP 使用相同轮数和候选预算，probe 也改用相同部署协议。
6. 如果 gold 已在候选中，先做候选来源名额保留或证据条件重排。只有表明第二轮有帮助，才把它引入主方法。
7. 保留单次 shared-backend 基线和完整质量—成本曲线。相同“两个区域”不代表相同 tokens；相同最高预算也不代表实际花费相同。

## 验证边界

局部与整题特征的反例、probe 抽样与硬预算、原始日志保留/认证字段排除、Reader 复用以及全流程模拟测试已覆盖。没有真实 GPU/API 实验，因此不声称方法收益已提高。日志逐事件保存，但不是逐题自动恢复；完整 fold 恢复沿用原 checkpoint 协议。接口和特征语义发生变化，新实验应使用新的输出目录。
