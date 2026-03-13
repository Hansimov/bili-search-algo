# Vocabs 工作流使用说明

本文档整理当前 `bili-search-algo` 中完整的词表构建流程，覆盖：

- `models.word.eng` 的中英文词表提取
- `models.sentencepiece.workflow` 的训练、merge、convert 统一入口
- 全量训练、断点排查、常用参数和最常用命令

当前最新的视频语料前缀统一使用 `sp_908m`。

性能说明：

- `workflow all` 现在会让 `models.word.eng` 和 `models.sentencepiece.train` 并行启动，而不是先跑完 `words` 再开始训练。
- `workflow words` 内部也会默认并行启动英文词表和中文词表提取。
- 如需回退到严格串行流程做对比或排障，可在 `all` 子命令上增加 `--serial-steps`。

## 1. 总体流程

完整流程通常分为 4 步：

1. 提取英文和中文词表记录
2. 训练各个 region 的 SentencePiece 模型
3. 将各 region 模型 merge 成统一词表
4. 将 merge 后的 SentencePiece 词表与 `models.word.eng` 的词表一起转换并合并成最终可用的 txt

统一入口是：

```sh
cd ~/repos/bili-search-algo
python -m models.sentencepiece.workflow -h
```

主要子命令：

```sh
python -m models.sentencepiece.workflow words -h
python -m models.sentencepiece.workflow train -h
python -m models.sentencepiece.workflow merge -h
python -m models.sentencepiece.workflow convert -h
python -m models.sentencepiece.workflow all -h
```

## 2. 训练目标设计

当前训练目标已经以单个 region 为基本单位，不再以 `1/2/3/4/r/w` 作为主要并发单元。

视频 region 列表：

- `cine_movie`
- `douga_anime`
- `tech_sports`
- `music_dance`
- `fashion_ent`
- `know_info`
- `daily_life`
- `other_life`
- `mobile_game`
- `other_game`
- `recent`

额外目标：

- `zhwiki`
- `test`

兼容旧习惯，仍然支持别名：

- `all` 或 `videos`: 所有 11 个视频 region
- `wiki` 或 `w`: `zhwiki`
- `x`: `test`
- `1/2/3/4/r`: 旧分组别名，仅为兼容旧命令保留

推荐直接使用 region 名称，或者直接使用 `all`。

## 3. 最常用命令

### 3.1 仅提取中英文词表

```sh
python -m models.sentencepiece.workflow words
```

默认会并行提取英文和中文词表。如果只想取一类：

```sh
python -m models.sentencepiece.workflow words --skip-chinese
python -m models.sentencepiece.workflow words --skip-english
```

等价于：

```sh
python -m models.word.eng -ec -en -mf 6
python -m models.word.eng -ec -zh -mf 6
```

### 3.2 全量训练 11 个视频 region

```sh
python -m models.sentencepiece.workflow train all -j 11 -p sp_908m -av -fd -e
```

### 3.3 全量训练 11 个视频 region + zhwiki

推荐在高并发时适当降低每个训练进程的线程数，避免总线程数过高：

```sh
python -m models.sentencepiece.workflow train all -iw -j 12 -nt 4 -p sp_908m -av -fd -e
```

### 3.4 训练单个 region

```sh
python -m models.sentencepiece.workflow train cine_movie -p sp_908m -av -fd -e
python -m models.sentencepiece.workflow train recent -p sp_908m -av -fd -e
python -m models.sentencepiece.workflow train zhwiki -j 1 -p sp_908m -fd -e
```

### 3.5 Dry-run 查看将要启动的全量命令

```sh
python -m models.sentencepiece.workflow train all -iw -j 12 -nt 4 -p sp_908m -av -fd -e -dr
```

### 3.6 只做 merge

```sh
python -m models.sentencepiece.workflow merge -i sp_908m -o sp_merged -mvs 1000000 -es
```

### 3.7 只做 convert

```sh
python -m models.sentencepiece.workflow convert -o sp_merged -cs -cr -cm -dc 908000000
```

### 3.8 一条命令跑完整流程

```sh
python -m models.sentencepiece.workflow all all -iw -j 12 -nt 4 -p sp_908m -av -fd -e -i sp_908m -o sp_merged -mvs 1000000 -es -cs -cr -cm -dc 908000000
```

说明：

- 第一个 `all` 是子命令
- 第二个 `all` 是训练目标，表示所有视频 region
- `-iw` 会额外包含 `zhwiki`
- 在这个命令里，`words` 和 `train` 会并发运行

## 4. `train.sh` 兼容入口

旧的 shell 入口仍然可用，但现在只是 Python CLI 的兼容包装：

```sh
./models/sentencepiece/train.sh all -iw -j 12 -nt 4 -p sp_908m -av -fd -e
./models/sentencepiece/train.sh cine_movie -p sp_908m -av -fd -e
```

不再推荐继续按 `1/2/3/4/r` 分组并发；更推荐直接用单 region 粒度。

## 5. 常用参数说明

### 5.1 通用训练参数

- `targets`: 训练目标，可写 `all`、单个 region、`zhwiki`、`test`
- `-p`, `--prefix-base`: 视频 region 模型前缀，默认 `sp_908m`
- `-iw`, `--include-wiki`: 在视频 region 之外追加训练 `zhwiki`
- `-j`, `--parallel`: 同时运行多少个训练进程
- `-nt`, `--num-threads`: 每个训练进程内部使用的线程数
- `-av`, `--auto-vocab-size`: 根据样本量自动估算 `vocab_size`
- `-vs`, `--vocab-size`: 手动指定词表大小；与 `-av` 二选一
- `-fd`, `--force-delete`: 覆盖已有模型，不再交互确认
- `-e`, `--edit-model`: 训练后执行模型清洗逻辑，额外写出 `.vocabx`
- `-cc`, `--character-coverage`: 默认 `0.9995`
- `-sf`, `--shrinking-factor`: 默认 `0.75`
- `-ml`, `--max-sentencepiece-length`: 默认 `16`
- `-bs`, `--batch-size`: 数据拉取 batch 大小
- `-mb`, `--max-batch`: 仅跑前几个 batch，适合做小批量测试
- `-dr`, `--dry-run`: 只打印命令，不真正执行
- `--serial-steps`: 在 `workflow all` 中关闭 `words` 和 `train` 的并发，强制串行执行

### 5.1.1 词表提取参数

- `--word-max-count`: 只处理前 N 条 doc，适合小批量性能测试
- `--word-parallel`: `workflow words` 内部并行数，默认 `2`
- `--word-log-dir`: `workflow words` 的日志目录

### 5.2 merge 参数

- `-i`, `--input-prefix`: 输入模型前缀，默认 `sp_908m`
- `-o`, `--output-prefix`: 输出模型前缀，默认 `sp_merged`
- `-mvs`, `--merge-vocab-size`: merge 后保留的最大词表大小
- `-tr`, `--trunc-ratio`: 每个 region 模型前多少比例的词参与 merge
- `-mc`, `--max-cjk-char-len`: 最大 CJK 长度限制
- `-avs`, `--min-ascii-video-support`: ASCII token 最少需要多少视频模型支持
- `-ass`, `--min-ascii-source-support`: ASCII token 最少需要多少总来源支持
- `-al`, `--min-ascii-len`: ASCII token 的最小长度阈值
- `-es`, `--export-stats`: 额外导出 `.stats.jsonl` 方便分析 merge 结果

### 5.3 convert 参数

- `-cs`, `--convert-sentencepiece`: 将 merge 后的 `.vocab` 转成 `.txt`
- `-cr`, `--convert-record`: 将 `models.word.eng` 的中英文 csv 转成 `.txt`
- `-cm`, `--convert-merge`: 将上面多个 `.txt` 合并成最终词表
- `-dc`, `--doc-count`: 词表记录的 doc count 前缀，当前使用 `908000000`
- `-n`, `--min-doc-freq`: `models.word.eng` 记录最小 doc freq，默认 `20`
- `-l`, `--max-char-len`: 词表记录最大字符长度，默认 `32`
- `-sp`, `--save-path`: 最终合并 txt 输出路径

## 6. 典型操作顺序

### 6.1 正式全量构建

```sh
python -m models.sentencepiece.workflow all all -iw -j 12 -nt 4 -p sp_908m -av -fd -e -i sp_908m -o sp_merged -mvs 1000000 -es -cs -cr -cm -dc 908000000
```

### 6.2 先训练，后 merge / convert

```sh
python -m models.sentencepiece.workflow train all -iw -j 12 -nt 4 -p sp_908m -av -fd -e
python -m models.sentencepiece.workflow merge -i sp_908m -o sp_merged -mvs 1000000 -es
python -m models.sentencepiece.workflow convert -o sp_merged -cs -cr -cm -dc 908000000
```

### 6.3 小批量调参

```sh
python -m models.sentencepiece.workflow train test recent -j 2 -bs 200 -mb 1 -vs 2000 -fd -e
```

### 6.4 小批量性能对比

串行基线：

```sh
time python -m models.sentencepiece.workflow all --steps words,train --serial-steps --word-max-count 50000 test -j 1 -bs 200 -mb 1 -cc 0.98 -vs 1200 -fd -e
```

并行版本：

```sh
time python -m models.sentencepiece.workflow all --steps words,train --word-max-count 50000 test -j 1 -bs 200 -mb 1 -cc 0.98 -vs 1200 -fd -e
```

这两条命令的区别只有 `--serial-steps`，适合直接比较 `words + train` 的阶段并行收益。

如果需要检查词表噪声，可使用：

```sh
python debugs/sentencepiece_eval.py models/sentencepiece/checkpoints/sp_merged.vocab
python debugs/sentencepiece_eval.py models/sentencepiece/checkpoints/sp_dbg_new_test.vocabx
```

## 7. 监控与产物位置

训练编排器会为每个训练目标写出：

- 状态文件目录：`models/sentencepiece/checkpoints/status/`
- 日志目录：`models/sentencepiece/checkpoints/logs/`
- 词表提取日志目录：`models/sentencepiece/checkpoints/logs/words/`

模型产物默认位于：

- `models/sentencepiece/checkpoints/*.model`
- `models/sentencepiece/checkpoints/*.vocab`
- `models/sentencepiece/checkpoints/*.vocabx`

merge 和 convert 产物通常位于：

- `models/sentencepiece/checkpoints/sp_merged.model`
- `models/sentencepiece/checkpoints/sp_merged.vocab`
- `models/sentencepiece/checkpoints/sp_merged.txt`

## 8. 额外说明

- 如果修改了 `/home/asimov/repos/sentencepiece` 的 C++ 源码，必须先执行 `python/rebuild.sh` 重建 Python bindings。
- merge 已经不再依赖不同模型间不可比的绝对 `score`，而是更偏向跨 region 支持度和相对排名。
- `zhwiki` 仍然作为补充来源参与 merge，但不再让 wiki-only 模板英文 token 主导最终排序。
