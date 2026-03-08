# CoreTok Training Pipeline

`models/coretok` 是 owner-domain token 化的主训练入口，目标不是继续手写语义词表，而是用可扩展、可监控、可回滚的训练链路，从真实视频文本中学习稳定的 token 词表和重要性统计。

## Current design

1. `CoreTagTokenizer`
   从 tag 学词表，控制 token budget 和词表膨胀。
2. `CoreTexTokenizer`
   在 title / desc 上复用并扩展词表。
3. `CoreCorpusStats`
   根据训练语料学习高覆盖 stop candidates，替代固定的低信息词常量表。
4. `CoreImpEvaluator`
   用 tag/text 文档频次估计 token importance。
5. `train.py`
   提供分规模训练、holdout owner retrieval 评估、自动调参、JSONL 监控日志。

## Training loop

建议按 scale 从小到大推进：

1. `tiny`
   验证词表是否收敛、query coverage 是否足够、holdout retrieval 是否明显优于随机。
2. `small`
   观察词表增长、stop candidate 学习是否稳定、Recall/MRR 波动是否可控。
3. `medium`
   只有在上一档 scale 的最佳配置稳定后才继续放大。

每档 scale 都会：

1. 构建 owner holdout 数据集
2. 训练多组阈值配置
3. 记录每轮指标到 `data/coretok/runs/<run>/events.jsonl`
4. 保存当前 scale 的最佳 bundle
5. 计算当前 scale 的稳定性统计

## Evaluation target

当前主指标不是人工标签分类，而是更贴近线上 owner search 的 holdout retrieval：

1. 用 owner 的一部分历史视频构造 support profile
2. 用保留视频构造 query
3. 用 token overlap 检索 support profiles
4. 评估 `MRR`, `Recall@1`, `Recall@5`, `Recall@10`, `query_coverage`

这种做法的目的，是尽量减少手工先验标签和固定词典对训练结论的污染。

## Run examples

小规模起跑：

```bash
python -m models.coretok.train --scales tiny --stop-on-unstable
```

从 `tiny` 扩到 `small`：

```bash
python -m models.coretok.train --scales tiny,small --start-date "2026-01-01 00:00:00"
```

输出目录：

1. `data/coretok/runs/<run>/events.jsonl`
2. `data/coretok/runs/<run>/<scale>/summary.json`
3. `data/coretok/runs/<run>/<scale>/best_bundle.json`