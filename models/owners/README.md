## Owners 搜索模型训练笔记

这份文档记录 owners 搜索相关模型的训练思路、探索尝试、实测结果和经验教训。目标不是追求一个离线分数最高的分类器，而是为 owner search / owner ranking 提供一组在真实数据规模下仍然可解释、可复现实验、可快速迭代的低成本特征与模型基线。

## 0. 明确废弃的旧思路

`daily_life`、`other_game`、`douga_anime` 这类粗标签分组，只适合最早期为了快速备份和做一个极低成本 baseline 时临时使用，不适合继续作为 owners 搜索模型的主目标、主标签体系或长期评估体系。

这里明确记录三条结论，后续不要再走回头路：

1. 不要再把这类粗标签当成 owner 搜索的产品语义空间。它们过粗、边界松散、跨域严重，无法准确表达“这个 UP 主到底在做什么”。
2. 不要再把 coarse label 分类准确率当作 owner 搜索质量的核心目标。它最多只能当早期 sanity check。
3. 后续模型应转向 owner profile / topic profile 路线，也就是直接建模 owner 的主题画像、长期内容分布、近期变化和可检索特征，而不是强行把 owner 塞进几个粗桶里。

## 1. 问题定义

当前 owners 搜索的核心问题不是“给每个 UP 主打一个最终标签”这么简单，而是要回答两个更实际的问题：

1. 当用户搜一个主题词时，这个 owner 是否真的长期稳定地生产该主题相关内容。
2. 当搜索服务把 owner 召回出来后，是否可以利用 owner 的主题倾向、影响力和活跃度做更合理的排序。

因此，这里的模型训练重点是验证 owner profile 里的哪些字段最有用，尤其是：

1. `owner_name`
2. `top_tags`
3. `sample_titles`
4. `desc_samples`

## 2. 数据构造思路

早期训练数据不是人工标注集，而是从 Mongo `videos` 集合里构造 pseudo-label owner-domain 样本。做法是：

1. 先按 `owner.mid` 聚合同一个 owner 的视频。
2. 根据视频命中的 region/filter 统计，给 owner 选一个 dominant label。
3. 只有在 dominant label 占比足够高时，才保留这个 owner 样本。

这样做的优点是规模可做得很大，缺点是标签噪声一定存在，所以实验时必须始终记住：

1. 离线准确率衡量的是“对 pseudo-label 的拟合能力”，不是最终线上质量。
2. 如果采样窗口太窄，模型可能学到的是窗口偏差，而不是 owner 的长期主题特征。

从当前阶段开始，这套 pseudo-label 构造只保留两个用途：

1. 快速验证某种字段设计是否完全无效。
2. 给 profile-based 检索特征提供最粗粒度的离线 sanity check。

它不再是主训练范式。

## 3. 新主线：从粗分类切换到 Owner Profile

新的主线不是“先给 owner 打粗标签再分类”，而是下面这条 profile-first 路线：

1. 从 `videos` 流式聚合出 owner 级中间画像。
2. 画像里保存可增量合并的稀疏主题特征，而不是只保存一个粗标签。
3. 中间画像优先写入 Mongo 预计算集合，作为训练和 ES 索引的统一上游。
4. 训练阶段从 Mongo owner profile 读取，而不是每次都直接扫原始 `videos`。
5. 检索阶段把 Mongo 画像压缩成 ES 需要的轻量字段，在线只做召回与小规模重排。

当前已经补了一个新的原型模块：`models.owners.profile`。它支持：

1. 流式扫描 `videos` 构建 owner profile
2. 输出 `topic_terms` 和 `feature_weights`
3. 计算 `recent_7d_videos` / `recent_30d_videos`
4. 批量写入 Mongo 预计算集合
5. 增量 merge 两份 owner profile 文档

这里需要特别强调：当前 `feature_weights` 已经不是早期那种直接累计关键词 token 的做法，而是基于 hashed subword bucket 的稀疏特征草图。这样做的原因是 owner 语料规模大、噪声高、同义表达和新词很多，直接维护开放词表的 token 权重既不稳，也不利于快速增量合并。

这条路线更符合生产要求，因为它天然支持：

1. 训练和索引解耦
2. 增量 owner 画像快速更新
3. 全量重训与在线替换分离

## 4. 训练流程

目前采用的是分层递进的低成本实验流程：

1. 先用 bounded sample 做快速对比，验证有没有继续投入的价值。
2. 再比较多种轻量 baseline，而不是一开始上重模型。
3. 在确认某个 baseline 明显更优后，再围绕它做小范围超参数调优。
4. 最后把实验放大到更接近真实服务窗口的数据集上复验。

这套流程的原因很直接：当前数据规模接近 9 亿视频、6000 万用户，如果一开始不做预算约束，很容易把实验变成基础设施压测，而不是模型验证。

## 5. 已尝试的模型

目前主要做过四类轻量 baseline：

1. `centroid`
2. `naive_bayes`
3. `naive_bayes_weighted`
4. `linear`

其中最关键的一步是把 `naive_bayes` 扩展成 `naive_bayes_weighted`，对结构化字段做差异化加权。直觉是：

1. `owner_name` 往往浓缩了 owner 的主身份或人设信号。
2. `top_tags` 是当前最稳定的主题信号。
3. `sample_titles` 有帮助，但噪声比 `top_tags` 更高。
4. `desc_samples` 能补充长尾信息，但强度不能给太高。

## 6. 关键实验结果

### 6.1 小窗口、300 owner 级别快速实验

实验设置：

1. `-m 300`
2. `--max-scanned-videos 200000`
3. 时间窗：`2026-02-01 00:00:00` 到 `2026-03-07 00:00:00`
4. `--min-videos 5`
5. `--sample-per-owner 10`

对比结果：

1. `centroid = 0.5179`
2. `naive_bayes = 0.5536`
3. `naive_bayes_weighted = 0.6071`
4. `linear = 0.4643`

结论：

1. `naive_bayes_weighted` 明显优于未加权 `naive_bayes`。
2. 结构化字段加权是值得保留的方向。

在同一预算和切分上继续做 tuning 后，最佳结果提升到：

1. `accuracy = 0.7143`
2. `alpha = 1.5`
3. `owner_name = 3.0`
4. `top_tags = 3.0`
5. `sample_titles = 1.0`
6. `desc_samples = 0.5`

### 6.2 更大真实窗口、5000 owner 级别复验

实验设置：

1. `-m 5000`
2. `--max-scanned-videos 2000000`
3. 时间窗：`2025-12-15 00:00:00` 到 `2026-03-07 16:00:00`
4. `--min-videos 5`
5. `--sample-per-owner 10`

这次一共从约 `186,789` 条视频构造出 `5,000` 个 owner 样本。

对比结果：

1. `centroid = 0.5793`
2. `naive_bayes = 0.6034`
3. `naive_bayes_weighted = 0.6185`
4. `linear = 0.5301`

这说明两件事：

1. `naive_bayes_weighted` 仍然是当前最稳的 cheap baseline。
2. 但小样本里看到的巨大优势，在大窗口里会明显收缩。

继续在这组 5000-owner 样本上调参，最终得到：

1. `accuracy = 0.6697`
2. `alpha = 1.0`
3. `owner_name = 3.0`
4. `top_tags = 3.0`
5. `sample_titles = 1.0`
6. `desc_samples = 0.5`

这个结果很关键，因为它说明之前在小窗口里调出来的最优结构，放到大窗口后基本仍成立，但最优 `alpha` 会发生漂移，而且整体可达到的上限没有小样本那么乐观。

## 7. 训练与调参时踩过的坑

### 6.1 不能默认全扫

这是最基础也最重要的一条。owner-domain 实验必须带预算约束，至少给以下之一：

1. `--max-owners`
2. `--max-scanned-videos`
3. 时间窗口 `-s/-e`

否则实验很容易退化成长时间无反馈的全库扫描，既不利于迭代，也很难定位瓶颈。

### 6.2 小样本最优配置不能直接外推

300-owner 实验里调出来的最佳结果是 `0.7143`，很容易让人误以为模型已经足够稳定。但一旦把窗口扩大到 5000 owner，未调参结果会回落到 `0.6185`，调优后也只有 `0.6697`。这说明：

1. 小样本更容易被窗口分布“奖励”。
2. 大窗口能更真实地暴露标签噪声和主题交叉问题。

### 7.3 `top_tags` 很强，但也会带来错配

`top_tags` 目前仍然是最有效的字段之一，但它也会把一些泛娱乐、泛二创、泛生活 owner 往错误领域拉。过去在粗标签体系里，这个问题会集中表现为：

1. `daily_life`
2. `other_game`
3. `douga_anime`

这三个类之间的混淆最明显。说明当前 pseudo-label 体系和字段本身都还有噪声，也进一步证明不应继续把这套粗标签当主路线。

### 7.4 `desc_samples` 价值存在，但权重不能高

目前多轮实验里，`desc_samples` 的最佳权重一直比较低，通常是 `0.5`。经验上它更像补充信息，而不是主信号。给太高时，容易把模型拉向泛描述词和情绪词。

## 8. 为什么要给 tuning 加进度日志

5000-owner 的 tuning 一共有 `243` 个候选配置。哪怕单个候选不重，整轮跑下来也足够长。如果没有进度日志，用户很难判断：

1. 当前是在正常训练
2. 还是卡在数据构造/参数搜索某一步

因此现在 `tune_naive_bayes_weighted` 已经会：

1. 在开始时打印总候选数
2. 每发现更优配置时立刻打印新 best
3. 按 `--tune-log-every` 周期输出当前进度和 best accuracy

默认每 `10` 个候选打一次日志，足够判断进展，又不会把终端刷爆。

## 9. 面向生产的训练与更新设计

目标要求是：

1. 全量数据 `1-3` 天内训完
2. 每日增量 `1` 小时内训完
3. 每小时增量 `10` 分钟内训完

要满足这个目标，训练链路必须拆成两层：

1. 重计算层：从 `videos` 到 `owner profile snapshots`
2. 轻训练层：从 `owner profile snapshots` 到 检索特征 / 向量 / 排序参数

具体建议如下：

1. 全量阶段按 `owner.mid % N` 分片并行构建 owner profile，中间结果写 Mongo，而不是直接边扫边训。
2. 每日增量只重算受影响 owner 的 profile delta，然后和全量 profile merge。
3. 每小时增量只更新最近窗口 owner 的稀疏主题权重、活跃度统计和必要向量，不做全量回放。
4. ES 只消费已冻结的 snapshot 版本。模型替换时切换 profile/version 指针，而不是在线重算。

当前建议的中间集合至少包括：

1. `owner_profile_snapshots_full`
2. `owner_profile_snapshots_daily`
3. `owner_profile_snapshots_hourly`
4. `owner_profile_training_samples`

训练性能之所以能提升，根因不是“把分类器再优化一点”，而是把最贵的原始视频聚合从训练阶段拿掉。

### 9.1 当前新的大样本预计算实验

在新的 `models.owners.profile` 原型上，我已经做了一次更接近生产形态的大样本实验：

```bash
cd /home/asimov/repos/bili-search-algo
PYTHONPATH=. python -m models.owners.profile \
	-m 20000 \
	--max-scanned-videos 2000000 \
	-s "2025-12-15 00:00:00" \
	-e "2026-03-07 16:00:00" \
	--mongo-output-collection owner_profile_snapshots_dev_poc1 \
	--mongo-batch-size 1000 \
	--log-every 200000
```

实测结果：

1. `20,000` 个 owner profile
2. `204,583` 条视频
3. `276.18s`
4. `740.77 videos/s`
5. `72.42 profiles/s`

### 9.2 最新 50k owner profile 真实链路验证

在替换成 hashed subword sparse feature 之后，又做了一轮更大的真实 DEV 预计算实验：

```bash
cd /home/asimov/repos/bili-search-algo
PYTHONPATH=. python -m models.owners.profile \
	-m 50000 \
	--max-scanned-videos 2000000 \
	-s "2025-12-15 00:00:00" \
	-e "2026-03-07 16:00:00" \
	--mongo-output-collection owner_profile_snapshots_dev_poc2_50k1 \
	--mongo-batch-size 1000 \
	--feature-buckets 4096 \
	--log-every 200000
```

实测结果：

1. `50,000` 个 owner profile
2. `456,502` 条视频
3. `1390.86s`
4. `328.22 videos/s`
5. `35.95 profiles/s`
6. Mongo collection: `owner_profile_snapshots_dev_poc2_50k1`

随后把这批 snapshot 写入新的 DEV owner 索引：

1. ES index: `bili_owners_dev_poc2_50k1_v2`
2. 写入量：`50,000` docs
3. 用时：`29.06s`
4. 写入吞吐：`1720.34 docs/s`

这轮真实链路至少验证了三件事：

1. `Mongo videos -> Mongo owner_profile_snapshots -> ES owners v2 -> OwnerSearcher` 这条链已经可以在真实 DEV 环境中跑通。
2. 新的 profile 字段设计（`topic_phrases` / `profile_text` / `feature_weights`）至少在工程集成上是闭环的。
3. 预计算和 ES 写入的职责已经拆开，后续可以分别优化训练与索引。

## 10. 当前真实查询观察

针对新索引 `bili_owners_dev_poc2_50k1_v2` 做了几组真实查询：

1. 查询 `影视飓风` 时，top1 能稳定命中 owner `影视飓风` 本人。
2. 查询 `王者荣耀` 时，结果里能召回 `幸运的盆子`、`指法芬芳张大仙` 这类明显相关 owner。
3. 查询 `黑神话悟空` 时，能召回部分相关游戏 owner，但 top1 仍被强 `王者荣耀` 画像 owner 占据，说明广义游戏语义仍会互相污染。

同时也暴露了一个必须继续修的点：

1. 用长尾短句 `当你半年不上线的账号打一把王者排位的时候` 做 domain 查询时，top1 反而命中了 `影视飓风`，说明当前 domain query 仍然过度依赖 analyzer 切词后的通用片段匹配，`topic_phrases/profile_text` 还缺少更强的 phrase 级约束。

这条观察很重要，因为它说明：

1. 新的 profile-first 架构已经通了。
2. 但 online retrieval 的 query 设计还不够“短句语义友好”。
3. 下一步优化重点应该放在短语匹配、字段分层召回和更稳的 sparse/dense 融合，而不是再回去做 coarse label 分类。

### 9.3 最新 100k owner 并行分片预计算实验

为了继续放大真实实验规模，同时验证预计算层的并行化可行性，新增了 owner shard 参数：

1. `--owner-shard-count`
2. `--owner-shard-id`
3. `--mongo-read-batch-size`

本轮采用 `4` 个 shard 并行写入同一个 Mongo snapshot collection：

```bash
cd /home/asimov/repos/bili-search-algo
PYTHONPATH=. python -m models.owners.profile \
	-m 25000 \
	-s "2025-12-15 00:00:00" \
	-e "2026-03-07 16:00:00" \
	--mongo-output-collection owner_profile_snapshots_dev_poc3_100k4 \
	--mongo-batch-size 1000 \
	--mongo-read-batch-size 10000 \
	--feature-buckets 4096 \
	--log-every 100000 \
	--owner-shard-count 4 \
	--owner-shard-id <0..3>
```

4 个 shard 的完成结果分别为：

1. shard0: `25,000` profiles, `222,014` videos, `1623.31s`
2. shard1: `25,000` profiles, `215,424` videos, `1630.07s`
3. shard2: `25,000` profiles, `214,860` videos, `1635.02s`
4. shard3: `25,000` profiles, `223,982` videos, `1573.32s`

合并后得到：

1. `100,000` 个 owner profile
2. 总扫描 `876,280` 条视频
3. wall-clock 约 `1635.02s`
4. 聚合后的有效吞吐约 `535.94 videos/s`
5. 聚合后的 owner 生成速率约 `61.16 profiles/s`
6. Mongo collection: `owner_profile_snapshots_dev_poc3_100k4`

随后写入 DEV ES v2：

1. ES index: `bili_owners_dev_poc3_100k4_v2`
2. 写入量：`100,000` docs
3. 用时：`57.4s`
4. 写入吞吐：`1742.16 docs/s`

这轮结果说明：

1. 分片并行是有效的，100k 真实全链路已经跑通。
2. 但当前按 `owner.mid % N` 的 shard 方式仍有明显效率损失，实际 wall-clock 提速还没有接近理想的 `4x`。
3. 当前瓶颈仍在 `videos -> owner profile snapshot` 阶段，而不是 `snapshot -> ES` 阶段。

## 10. 最新真实检索评估

### 10.1 长尾短句误召回修复

针对 `当你半年不上线的账号打一把王者排位的时候` 这类长尾短句，已经把 online domain query 改成：

1. 长 query 强制至少命中一个 phrase/AND 严格子句
2. 再用较宽松的 `top_tags/topic_phrases/sample_titles/profile_text` 做补充分数
3. 避免在 `sort_by=influence` 时，被宽泛 token 匹配的高影响力 owner 直接顶上来

在 `bili_owners_dev_poc2_50k1_v2` 和 `bili_owners_dev_poc3_100k4_v2` 上，这个长尾短句都已经从原来的明显误召回，收敛成单一命中结果。

### 10.2 100k query panel 观察

在新的 `bili_owners_dev_poc3_100k4_v2` 上做了几组真实查询：

1. 用 `OwnerSearcher.search("影视飓风")` 走 relevance 路由时，top1 仍然是 owner `影视飓风` 本人。
2. 用 domain-only + `sort_by=influence` 路由查 `影视飓风` 时，top1 会被更高影响力但只是在语料里碰到“影视”相关 token 的账号顶掉，说明 name-like query 和 domain-like query 仍需要更强的路由区分。
3. `黑神话悟空` 在 relevance 路由上，已经能把强相关的黑神话实况/混剪 owner 顶到前面，明显好于只看 `influence` 的 domain-only 排序。
4. `王者荣耀` 这类超泛主题词，在 domain-only + influence 排序下仍然会混入与主题弱相关但高影响力的账号，说明超泛 query 还需要更强的领域约束或二阶段重排。

因此，这一阶段可以明确下来的结论是：

1. 长尾短句误召回问题已经在 query 结构上被修正。
2. 100k 全链路已经验证完成。
3. 下一步在线优化重点，应从“短句误召回”转到“name/domain 路由分流”和“超泛 query 的领域约束”。

### 10.3 query routing 最新进展

这一轮又继续把 query routing 往前推了一步，不再只依赖 merge 之后的 `query_type` 判断，而是在查询入口就做 route probe：

1. phrase-like query 继续走严格 domain route
2. 非 relevance 排序下，如果 query 存在 `name.keyword` 精确命中，则直接切到 strict name route
3. strict name route 只保留 `name.keyword` + `name.words(and)`，避免被 `top_tags/topic_phrases/profile_text` 的宽匹配污染

真实 DEV 验证结果：

1. 在 `bili_owners_dev_poc3_100k4_v2` 上，`search_by_domain("影视飓风", sort_by="influence")` 已经收敛成单一命中 owner `影视飓风`。
2. `search("影视飓风", sort_by="influence")` 也已经走 strict name route，top1 同样是 owner 本人。
3. 长尾短句 `当你半年不上线的账号打一把王者排位的时候` 在新路由下仍保持单一命中，没有回退。

这说明 name/domain/phrase 三类请求已经不再完全共用同一条非 relevance 路由，至少把最明显的“name-like query 被 domain-only 排序污染”的问题修掉了。

### 10.4 planned-range 分片优化结果

在 `% shard` 之外，又补了一个新的 range 分片路线：

1. 先对当前时间窗里的唯一 `owner.mid` 做 `bucketAuto`，得到近似均衡的 owner range plan
2. 每个 shard 使用显式 `owner_mid_min / owner_mid_max`
3. 对 range shard 强制使用 `owner.mid_1` Mongo index hint

对应命令链路已经补到 `models.owners.profile`：

1. `--plan-owner-shards`
2. `--owner-shard-mode range`
3. `--owner-mid-min`
4. `--owner-mid-max`
5. `--mongo-hint owner.mid_1`

新的 100k planned-range 实验结果：

1. `100,000` owner profile
2. 总扫描 `921,308` 视频
3. wall-clock `1488.37s`
4. 聚合吞吐 `619.00 videos/s`
5. owner 生成速率 `67.19 profiles/s`
6. Mongo collection: `owner_profile_snapshots_dev_poc9_100k4_rangeplan`

相对上一轮 `% shard` 的 `535.94 videos/s` / `61.16 profiles/s`：

1. video throughput 提升约 `15.5%`
2. profile throughput 提升约 `9.9%`
3. wall-clock 从 `1635.02s` 降到 `1488.37s`

随后写入 ES：

1. ES index: `bili_owners_dev_poc9_100k4_rangeplan_v2`
2. `100,000` docs
3. `53.39s`
4. `1873.09 docs/s`

但这里必须记一条重要 caveat：

1. 当 `range` 分片和 `--max-owners 25000` 一起使用时，每个 shard 实际上会优先取本 range 内按 `owner.mid` 排序靠前的 owner，样本组成会和 `% shard` 路线不一样。
2. 这意味着 planned-range 这一轮更适合用来评估“预计算吞吐”，不适合直接和 `% shard` 的 100k 检索质量做一一对应对比。
3. 例如在 `bili_owners_dev_poc9_100k4_rangeplan_v2` 上，`影视飓风` 本人就不在这个 100k 样本里，因此不能拿这组索引来评判 `影视飓风` 的 name route 是否失效。

所以当前比较稳妥的结论是：

1. query routing 的线上行为，用 `bili_owners_dev_poc3_100k4_v2` 这类更均匀的 sample 来看。
2. range-planned 方案的主要价值，是证明 owner-mid range + index hint 的预计算性能更好。
3. 后续如果要用 range 路线做公平检索评估，要么取消 `max_owners` 截断，要么再设计一层 range 内采样策略。

真实结果：

1. `20,000` 个 owner profile
2. `204,583` 条视频
3. `276.18s`
4. 单进程吞吐约 `740.77 videos/s`
5. 单进程画像生成速率约 `72.42 owners/s`

这个实验很重要，因为它第一次把“直接从 videos 流式构建 owner profile 并写 Mongo 中间层”的链路跑通了。

### 9.2 生产时效估算

按这次单进程约 `741 videos/s` 的实测吞吐，粗略估算：

1. 全量 `9e8` 视频，单进程约需 `14` 天
2. `8` 个并行分片 worker，约需 `1.75` 天
3. `12` 个并行分片 worker，约需 `1.17` 天

这说明全量 `1-3` 天的目标不是靠“更强分类器”达成的，而是靠：

1. owner 分片并行
2. Mongo 中间层持久化
3. 训练阶段只读取 owner profile，不再重复扫视频

对于增量：

1. 如果日增大约在数百万视频量级，那么 `3-4` 个并行 worker`就能压进 `1` 小时量级
2. 如果小时增是几十万视频量级，单 worker 就有机会压进 `10` 分钟左右

前提是：

1. 增量链路只处理受影响 owner
2. profile merge 是增量合并，而不是全量重建
3. 向量编码只对受影响 owner 做局部更新

## 10. 当前阶段的经验结论

如果目标是在 owner search 里快速获得可用的主题表征，并且满足生产级全量/增量时效，那么当前最稳妥的路线是：

1. 粗标签分类器退居二线，只保留为低成本 sanity check。
2. 主路线切换到 `owner profile` 预计算、增量 merge、ES 轻索引。
3. `naive_bayes_weighted` 仍可保留为 profile 特征验证 baseline，但不再承担主语义建模职责。
4. 默认采用 `owner_name=3.0, top_tags=3.0, sample_titles=1.0, desc_samples=0.5` 这一组权重起步。
5. `alpha` 不要写死，小窗口和大窗口的最优值不完全相同，需要随数据预算重跑。
6. 每次扩大数据窗口时，都要同时看 `accuracy`、`leaderboard` 和 `error_summary`，不能只盯一个最终分数。

## 11. 下一步建议

后续如果继续迭代 owners 搜索模型，优先级建议如下：

1. 先把 `models.owners.profile` 真正用于大规模 owner 画像预计算，并把产物落到 Mongo。
2. 再从 Mongo profile 产物生成 ES v2 索引文档，而不是直接从 `videos` 重新聚合。
3. 设计 profile-based domain retrieval 和小规模重排，不再依赖 `daily_life / other_game / douga_anime` 这类粗标签桶。
4. 继续扩充更贴近线上查询分布的 owner 样本，而不是一味扩大总量。
5. 保持实验预算和进度日志，避免再次出现“任务是否卡住不可判断”的问题。
