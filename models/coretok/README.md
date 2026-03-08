# CoreTok Training Skeleton

`models/coretok` 是新的 owner-domain token 化主线入口。当前阶段先提供可运行、可测试的训练骨架，不再继续沿用旧的 `top_tags / topic_phrases / domain_text / vector_bucket_*` 方案。

## Stage 1: CoreTagTokenizer

目标：从高质量 tag 里学习一套小而稳的核心 token 词表。

当前实现包含：

1. 混合长度规则：每个中文字符记 1 个单位；每 3 个西文数字字符记 1 个单位。
2. tag 过滤规则：总长度超过 8 单位丢弃；中文超过 8 字丢弃；西文数字超过 24 字符丢弃。
3. token budget：
   - `<= 3` 单位 -> `1` token
   - `4` 单位 -> `1` token
   - `5-6` 单位 -> `2` tokens
   - `7-8` 单位 -> `3` tokens
4. 候选生成避免无约束 bigram，优先 whole-form 和平衡切分片段。

## Stage 2: CoreTexTokenizer

目标：在 title / desc 上微调已有词表，并通过更高的新词阈值控制词表膨胀。

当前实现包含：

1. 复用 stage1 词表
2. 更高 `novelty_threshold`
3. 优先复用相似已有 token，再决定是否创建新 token

## Bypass: CoreImpEvaluator

目标：估计 token 的信息量和重要性。

当前实现先提供一个轻量 baseline：

1. tag/text 双文档频次统计
2. 稀有度 `idf` 风格打分
3. tag-dominant token 额外加权

## What is intentionally not implemented yet

这轮没有把线上 encoder/decoder 和 owner 索引写回完全打通，当前只先把训练骨架和 Mongo 数据流入口补齐，避免继续在旧噪声字段上投入。