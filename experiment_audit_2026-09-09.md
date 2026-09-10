# 完整实验链路审计（2026-09-09）

## 判断

当前可以确认：数据准备可执行、真实数据身份检查通过、修复后的实验编排在模拟后端下完成全流程、25 项测试通过。不能确认真实模型全量实验跑通：当前检查的 Python 环境缺少 FAISS、LightGBM、HippoRAG、igraph、leidenalg 和 warp-g 安装元数据，`torch.cuda.is_available()` 为 False；官方基线 external 仓库与实验 outputs 目录也不存在。

预检机器可读结果：`experiment_preflight_2026-09-09.json`。这份报告不声称已验证 API、模型下载、显存容量或真实图质量。

## 查阅范围

- `warp/data/`、`scripts/prepare_benchmark.py`、`scripts/prepare_hipporag2.py`：schema、证据 ID、答案、cross-fitting。
- `warp/retrieval/`：BM25、Dense、ANN、RRF、CrossEncoder 与接口。
- `warp/partition/`、`warp/advisor/`：共访问图、Leiden、特征、probe、收益预测与选择。
- `warp/graph/`：图缓存身份、官方参数映射、构图、检索、usage。
- `warp/baselines/global_graph.py`：KET/G2ConS 本地适配基线。
- `warp/eval/`、`warp/run.py`：检索/QA 指标、成本、统计、五折编排、reader、分区消融。
- `scripts/run_*`、`scripts/export_*`：主实验与官方基线执行、结果汇总与导出。
- 四份 paper YAML、官方 baseline YAML、pyproject 依赖、现有测试与方法实现说明。
- 额外核对锁定 commit 的 HippoRAG、LinearRAG、LightRAG 上游源码；没有安装这些后端运行真实模型。

## 真实数据检查

| 数据集 | 文档 | 查询 | gold 文档数分布 |
|---|---:|---:|---|
| HotpotQA | 9,811 | 1,000 | 全部 2 |
| 2Wiki | 6,119 | 1,000 | 2：765；4：235 |
| MuSiQue | 11,655 | 1,000 | 2：518；3：316；4：166 |
| PopQA 发布版本 | 8,676 | 1,000 | 全部 2 |

所有查询 gold ID 都能在对应 corpus 找到；没有空 gold、重复文档内容或缺失答案；五折各 200 测试题。

从本地 raw 文件重新执行四份转换，修复答案别名之前生成的 corpus/queries 与原 processed 文件逐字节一致。这验证转换可复现，不证明每份 gold 在语义上都是回答问题的必要证据。

发现 MuSiQue 276 题有 `answer_aliases`，旧 `_answers()` 在发现主答案后立即返回，丢失全部别名。已修复转换器并重新生成 MuSiQue processed 文件。比对确认只有 276 个 answer 字段改变，问题 ID、文本、gold、顺序与 corpus 保持一致。此前 QA EM/F1 应按修正后的答案重新评分；检索指标不受该答案修复影响。

此前“数据目录为空”的判断是普通 rg 搜索忽略 gitignored 数据造成的错误，已更正。

## 新增修复和审计能力

| 问题 | 影响 | 处理 |
|---|---|---|
| 部署图与 probe 图重叠时首次成本重复计费 | 高估 WARP 首次构建成本 | 按部署集合与 probe 集合并集计费；保留总设计成本和增量设计成本 |
| Gain-only 使用 probe 收益却没有计设计成本 | 成本比较偏向 Gain-only | 与 WARP 一样计入收益学习所需设计成本 |
| probe/interaction 的检索 LLM 成本未计入首次成本 | 低估设计开销 | 记录 design retrieval usage，并计入逻辑 token、估算费用与检索时间 |
| QA 返回少于请求数量时 zip 静默截断 | 不完整结果被当作有效统计 | Reader 和 LinearRAG 校验返回数量与问题顺序 |
| Reader top_k 被后端 qa_top_k 再次截断 | 配置与真正阅读证据数量不一致 | QA 调用期间应用 top_k，并在成功/异常后恢复配置 |
| 不等长 fold 的 QA 平均数直接等权平均 | 总体 EM/F1 加权不正确 | 以实际 query 数加权；保留折间离散性字段 |
| 没有 WARP 的配置仍执行 WARP 配对比较 | 后处理失败 | 无 WARP 时返回空配对比较 |
| 配对检验没有 Hybrid/Dense 等基础参照 | 无法直接检验图相对 Base 的价值 | 将五种 Base/Full-graph 参照纳入各预算的配对检验和 Holm 校正 |
| gold 显式 ID 与 supporting title 被混合 | 通用转换器误报缺失证据 | 显式 ID 优先，并支持字符串 supporting entry |
| cross-fit 全局重复 ID 或 fold 多于查询 | 可能泄漏或产生空测试折 | 在拆分前拒绝 |
| 结果写入并非原子操作 | 失败可能破坏已有 JSON | 临时文件写入、fsync、原子替换；拒绝 NaN/Infinity |
| 全部五折结束才保存结果 | 后续失败丢失已完成折 | CLI 默认逐折 checkpoint；按配置、数据、包版本和 WARP 源码哈希验证恢复 |
| ablation 新模型创建时旧模型仍被引用 | GPU 峰值内存偏高 | 清理旧模型、工厂和绑定搜索引用后再构建下一模型 |
| KET 先保存全语料共享关键词文档对 | Python 字典内存可能趋近二次规模 | 改为每文档累计并保留 top-k；小型穷举验证结果一致，计算时间仍可能较大 |
| LightRAG 配置模型名没有应用到固定 wrapper | 元数据可能与实际请求模型不一致 | 对当前适配器支持的固定模型组合进行显式校验 |
| 官方基线索引路径只区分数据集名字 | 更换语料/配置后可能复用旧索引 | 路径加入 corpus、commit、settings 指纹 |

前一轮修复的显式选区、缓存状态、漏构图、路由诊断和重复 query ID 问题详见 `retrieval_bug_audit_2026-09-09.md`。

## HippoRAG 的 Dense fallback

锁定版本的 [`retrieve()`](https://raw.githubusercontent.com/OSU-NLP-Group/HippoRAG/c617143f01477243992a63b2e2151cc003dd3b21/src/hipporag/HippoRAG.py) 在事实过滤结果为空时执行 dense passage retrieval。返回的 `graph_seeds` 可用于区分这一路径与有事实种子的图搜索。

适配器现记录 `dense_fallback_calls` 和 `graph_seeded_calls`，它们随 `online_retrieval_cost` 输出。它们是后端调用次数，不是去重后的 query 数，因为一个 query 可以访问多个区域。图已被调用不意味着它执行了有效图传播，后续应重点检查退回比例；当前没有真实实验计数，不能认定此前实验确实发生了大量 fallback。

还核对了 [LinearRAG](https://raw.githubusercontent.com/DEEP-PolyU/LinearRAG/bcc94e66c221f798801255efba09311d6fbcd8d6/src/LinearRAG.py) 的 QA 返回结构和 [LightRAG](https://raw.githubusercontent.com/HKUDS/LightRAG/d49112fb7548ee14cb727d43bd68e34da0a2c942/lightrag/llm/openai.py) 的固定模型 wrapper。上游接口核对不是运行验证。

## 测试证据

25 项 unittest 全部通过，所有 warp/scripts/tests Python 编译通过。其中一个集成测试实际执行 runner 控制流，使用两折、两档预算、七种方法、全部 reader 分支、三种分区消融、Full Graph/graph-only、检索指标、配对检验、JSON 和 CSV 导出，并验证 checkpoint 恢复不重复运行、配置变化拒绝旧 checkpoint。

集成测试替换了昂贵的模型、物理设计训练、全局图基线后端和 QA reader。区域搜索/融合、选区、指标、统计、成本与 runner 运行真实项目代码。该测试证明编排与接口能衔接，不证明真实 HippoRAG、LightGBM、FAISS 或模型质量。

```bash
/home/fuyj/.miniconda/envs/test/bin/python -m unittest discover -s tests -v
python3 -m compileall -q warp scripts tests
/home/fuyj/.miniconda/envs/test/bin/python scripts/check_experiment.py \
  --output experiment_preflight_2026-09-09.json
```

最后一条当前返回 1，表示预检发现真实运行阻塞；这不是测试失败。

## 正式运行仍需验证的事项

1. 在有 CUDA 的实际运行环境安装本项目及锁定后端，检查模型 revision 和 API。当前预检只检查依赖可发现性，不代表所有依赖能正确导入或模型能加载。
2. 在同一生产配置下做真实构图和检索 smoke test，确认 source_id、图事实数量、fallback 比例、usage 以及显存峰值；小样本需仍满足至少六个有工作负载覆盖的区域，不能任意缩到几篇文档后期待收益预测正常训练。
3. 真实跑完至少一折后，再恢复执行五折与四数据集 suite。checkpoint 粒度是完整 fold，折内失败仍需重跑该折。
4. 当前 KET/G2ConS 是共享 HippoRAG 后端的本地适配版本，不能把这里的结果直接称为作者官方实现结果；官方端到端脚本覆盖的是 LinearRAG 与 LightRAG。
5. 实际构图成本与选择预算的 token proxy 不是同一个量。AUC 还需结合已报告的成本覆盖区间解读，不能直接比较不同覆盖范围的面积。设计阶段 CPU/重排时间未被完整拆分计量，本次修复没有把这些计时宣称为全系统端到端账单。
6. 当前方法仍是一次区域检索与融合，没有实现讨论中的证据驱动两轮检索。完整执行不等于一定获得预期提升。

正式 CLI 例子（环境就绪后）：

```bash
python -m warp.run --config configs/paper/hotpotqa.yaml --output outputs/paper/hotpotqa.json
```

默认保存/恢复 `outputs/paper/hotpotqa.json.folds/`。代码、配置或数据改变时请使用新的输出/checkpoint 目录，避免混合不同实验版本。
