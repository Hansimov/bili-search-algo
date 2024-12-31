# bili-search-algo
Algorithms and models for Bilibili Search Engine (blbl.top).

## models.sentencepiece.train

Train sentencepiece model from video texts.

```bash
python -m models.sentencepiece.train -m sp_507m_400k_0.9995_0.9 -ec -vs 400000 -cc 0.9995 -sf 0.9 -e
```

Test:

```bash
python -m models.sentencepiece.train -m sp_507m_400k_0.9995_0.9 -t
```

## models.sentencepiece.merge

Merge sentencepiece models which are trained on different datasets.

```bash
python -m models.sentencepiece.merge
```

## datasets.videos.cache

Tokenize video texts from database, and save to parquets. Used by `datasets.videos.freq` and `models.fasttext.train`.

```bash
python -m datasets.videos.cache -ec -dn video_texts_tid_all -fw 200 -bw 100 -bs 10000
```

## datasets.videos.freq

Count video terms freqs from database or parquets, and save to csv and pickle. Used by `models.fasttext.train`.

Specify region tid:

```bash
python -m datasets.videos.freq -o video_texts_freq_tid_17_nt -dn "video_texts_tid_17" -tid 17 -nt
```

All regions:

```bash
python -m datasets.videos.freq -o video_texts_freq_tid_all_nt -dn "video_texts_tid_all" -nt
```