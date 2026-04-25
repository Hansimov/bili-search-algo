# 语义资产生成流程

`models.semantics` 负责从真实视频文档中生成一份可版本化、可增量更新、可被 `es-tok` 查询期直接加载的 compact semantic bundle。当前实现的核心是“构建期只写 doc-term segment，合并期先做 DF 过滤，再生成关系”。

## 设计边界

- `bili-search-algo` 负责离线抽取文档词项、合并节点和关系，并写出 compact TSV。
- `es-tok` 只负责加载 bundle 并在 `related_tokens_by_tokens(mode=semantic)` 中消费。
- `bili-search` 继续通过 relation client 调用 `mode=semantic`，不直接读取离线产物。
- 构建期不再产出 `edges.tsv` 或 `edges.seg.*.tsv`；共现关系只在合并期、经过 broad/rare term 过滤后生成。

## 产物结构

默认输出目录为 `data/semantics/<version>/`：

```text
group_00/processed.sqlite
group_00/segments/docs.seg.<ts>.tsv
...
merged/nodes.tsv
merged/rewrite.tsv
merged/synonym.tsv
merged/near_synonym.tsv
merged/doc_cooccurrence.tsv
merged/negative_samples.tsv   # 默认为空；显式开启训练负样本时写入
merged/meta.json
```

`docs.seg.*.tsv` 每行是一篇文档，格式为：

```text
doc_key<TAB>content_hash<TAB>term<TAB>roles<TAB>score...
```

merged 下的四个关系文件使用同一格式：

```text
source<TAB>target<TAB>weight<TAB>target<TAB>weight...
```

这样 Java 侧可以按行加载，不需要解析嵌套 JSON。TSV 中的 token 内部空格统一编码为 `▂`，Java 侧加载时会解码回普通空格。

## 常用命令

从 JSONL 调试：

```bash
python -m models.semantics --version v1 --num-groups 10 --vocab-limit 800000 build \
  --input-jsonl debugs/live_case_reports/semantic_docs_merged_real.jsonl \
  --workers 10 --group-chunk-size 20000
```

从 MongoDB 增量构建：

```bash
SEMANTICS_MONGO_URI="mongodb://USER:PASSWORD@HOST:PORT" \
python -m models.semantics --version v1 --num-groups 10 --vocab-limit 800000 build \
  --filter '{}' --limit 12000000 --workers 10 --log-every 500000
```

合并产物：

```bash
python -m models.semantics --version v1 --num-groups 10 merge \
  --min-df 8 --min-cooc 5 --top-k 24 --max-df-ratio 0.06 --min-score 0.3
```

查询期 bundle 默认不生成训练负样本，以减少 merge 时间和产物体积；如果需要为后续训练任务导出负样本，可显式追加：

```bash
--negative-samples-per-doc 4
```

检查状态：

```bash
python -m models.semantics --version v1 status
python -m models.semantics --version v1 inspect 采访
```

吞吐 benchmark：

```bash
python -m debugs.semantics.bench --docs 100000 --workers 10
```

该脚本使用合成词表和合成文档，只测 `models.semantics` 的分组、增量、抽取和 segment 写出吞吐，不包含 Mongo 读取和完整词表 I/O。

## 增量机制

每个 group 都有独立的 `processed.sqlite`，表中记录 `doc_key -> content_hash`。`doc_key` 优先使用 `bvid`，缺失时回退到 `aid/id/_id`；`content_hash` 覆盖 `title/desc/tags/rtags/owner.name/tid`。再次构建时，未变化文档会被跳过。

segment 文件是追加写入的；如果同一个 `doc_key` 的内容发生变化，旧行不会原地删除。merge 阶段会先构建 `current_docs` 索引，只把每个 `doc_key` 的最新 segment 行计入 DF 和共现统计。首次全量构建时 `segment_rows_seen == current_docs`，meta 中的 `current_doc_filter_enabled` 会是 `false`，此时不会做逐行 current-doc 查询，避免拖慢大规模合并。

## 合并策略

- 预扫描 `docs.seg.*.tsv`，记录每个 `doc_key` 的最新 `content_hash`，保证增量更新不会重复计入旧版本文档。
- 第一遍统计扫描用 SQLite 汇总 term DF 和字段角色 DF。
- `min_df` 过滤过稀有词，`max_df_ratio` 过滤过泛词。
- 第二遍只在 allowed terms 内生成 doc co-occurrence pair，并受 `max_terms_per_doc`、`max_pairs_per_doc` 和 `top_k` 限制；pair 聚合使用内存中的 int-key 结构，避免 SQLite 高频 upsert。
- `rewrite / synonym` 包含一小组确定性高置信规则，`near_synonym` 会在这些兜底规则之外，从高支持度、高 lift 的真实共现边中动态提炼候选；`doc_cooccurrence` 则保留更宽的真实文档共现关系。
- 合并输出会清理旧 `edges.tsv`，避免 Java 侧或人工排查时误读旧格式。

## 性能策略

- 文档按稳定哈希分为 10 个 group。
- 每个 group 分批提交到 worker，segment 直接落盘。
- merge 阶段只用 SQLite 保存 `current_docs` 和 `term_stats`；共现边和可选负样本使用整数 pair key 在内存中聚合，再直接写 compact TSV。
- 抽取阶段只保留有限数量的 title/tag/owner 词，控制单文档候选词数量。
- 默认只加载 `vocabs.txt` 前 80 万个词；需要完整词表时可传 `--vocab-limit 0`。
- merge 默认关闭 negative samples。它们只供训练型模型使用，不进入 es-tok 查询扩展；关闭后可以避免无用的大文件复制和排序开销。

## Live 验证结果

真实 Mongo 语料使用 10 个 group、10 个 worker、默认 80 万词表，已完成 build + merge。2026-04-25 的 1M merge 参数为：

```text
--min-df 12 --min-cooc 20 --top-k 24 --max-df-ratio 0.05
--max-terms-per-doc 8 --max-pairs-per-doc 20 --min-score 0.36
```

```text
200k merge: 199999 docs, 2034940 term rows, 2055333 unique edge pairs,
            24484 doc_cooccurrence rows, 18.92s elapsed, 578MB max RSS
1M merge:   999993 docs, 9955276 term rows, 8604590 unique edge pairs,
            21633 doc_cooccurrence rows, 93.36s elapsed, 2041MB max RSS
```

此前 1M merge 的 SQLite edge upsert 路径在 4 分钟后仍未完成；当前实现可以稳定完成 1M 真实样本合并。

插件集成也已做 live reload：`es_tok 1.0.0` 在 dev ES 9.2.4 上加载成功，cluster health 为 green。`semantic_docs_merged_real.jsonl` probe 在收口 composable 关系后结果为 `28 docs -> 67 terms -> 57 non-empty`；`康夫 ui` live 请求返回 `comfyui`，验证了 TSV 空格掩码和 Java 解码链路。
