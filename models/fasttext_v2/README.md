# fasttext_v2

`fasttext_v2` 直接使用 `models.semantics` 生成的 compact TSV 作为训练数据源，避免继续依赖旧的 parquet、旧 SentencePiece 和旧 token frequency 文件。

## 设计目标

- 训练语料来自 `data/semantics/<version>/group_*/segments/docs.seg.*.tsv`。
- 词表和频次来自 `data/semantics/<version>/merged/nodes.tsv`。
- 默认用 `nodes.tsv` 过滤低质量标题碎片、超高频泛化 tag、活动 tag 和标点噪声。
- 默认把 `synonym.tsv` / `near_synonym.tsv` 中的高置信关系写成训练句，让模型更偏向 rewrite/expansion/correction，而不是只学习视频共现。
- 标签词比标题切词权重更高，训练句子中会重复标签词，并插入 `__tag__ / __title__ / __video__` 轻量上下文标记。
- 输出语料是普通空格分隔文本，便于用 gensim FastText 或原生 fastText CLI 训练。
- 不建议把 fastText v2 当全词表 ANN 使用；它更适合给语义图、纠错生成器、TEI/LSH 召回出来的小候选集做 rerank/filter。

## 常用命令

只生成训练语料：

```sh
python -m models.fasttext_v2 \
  --version-root data/semantics/v1 \
  --output-root data/fasttext_v2 \
  --name semantic_v1 \
  --prepare-only
```

小样本快速训练：

```sh
python -m models.fasttext_v2 \
  --version-root data/semantics/v1 \
  --output-root data/fasttext_v2 \
  --name semantic_v1_smoke \
  --max-docs 200000 \
  --epochs 1 \
  --vector-size 128 \
  --workers 8
```

只训练高可信关系句，适合先验证 rewrite/expansion 候选 rerank：

```sh
python -m models.fasttext_v2 \
  --version-root data/semantics/v1 \
  --output-root data/fasttext_v2 \
  --name semantic_v1_relations \
  --relations-only \
  --epochs 10 \
  --min-count 1
```

固定候选打分探测：

```sh
PYTHONPATH=. python debugs/semantics/probe_fasttext_v2_candidates.py \
  data/fasttext_v2/semantic_v1_relations.model
```

## 后续接入方向

- query rewrite：先由语义图/规则/embedding 产生小候选集，再用 centered fastText v2 分数做弱 rerank。
- correction：结合字符 ngram 向量召回相似型号/拼写，但必须再用 ES 文档频次和 embedding/LSH 过滤。
- denoising：把查询词和候选标题/tag 的 centered fastText v2 相似度作为轻量特征，减少每次都调用远程 embedding 的成本。
