from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from models.fasttext_v2.corpus import (
    FastTextV2CorpusConfig,
    iter_semantic_segment_sentences,
    iter_relation_sentences,
    load_nodes_vocab_map,
    load_nodes_vocab,
    sentence_from_doc_terms,
    write_training_corpus,
)
from models.fasttext_v2.scoring import FastTextV2CandidateScorer
from models.semantics.storage import encode_normalized_term


def test_sentence_from_doc_terms_weights_tags_and_masks_spaces():
    sentence = sentence_from_doc_terms(
        (
            ("机器 学习", 2, 1.0),
            ("教程", 1, 0.72),
        ),
        FastTextV2CorpusConfig(tag_repeat=2, title_repeat=1),
    )

    assert sentence == [
        "__video__",
        "__tag__",
        "机器▂学习",
        "机器▂学习",
        "__title__",
        "教程",
    ]


def test_iter_semantic_segment_sentences_and_write_corpus():
    with TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        segment_dir = root / "group_00" / "segments"
        segment_dir.mkdir(parents=True)
        segment_dir.joinpath("docs.seg.1.tsv").write_text(
            "\n".join(
                [
                    "\t".join(
                        [
                            "bvid:1",
                            "hash1",
                            encode_normalized_term("显卡"),
                            "2",
                            "1",
                            encode_normalized_term("GPU"),
                            "1",
                            "0.72",
                        ]
                    ),
                    "\t".join(
                        [
                            "bvid:2",
                            "hash2",
                            encode_normalized_term("孤例"),
                            "1",
                            "0.72",
                        ]
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        config = FastTextV2CorpusConfig(min_terms_per_doc=2)
        sentences = list(iter_semantic_segment_sentences(root, config))
        assert sentences == [["__video__", "__tag__", "显卡", "显卡", "__title__", "GPU"]]

        output_path = root / "train.corpus.txt"
        meta = write_training_corpus(root, output_path, config)
        assert meta["docs"] == 1
        assert output_path.read_text(encoding="utf-8").strip() == (
            "__video__ __tag__ 显卡 显卡 __title__ GPU"
        )
        assert output_path.with_suffix(".txt.meta.json").exists()


def test_load_nodes_vocab_filters_min_df_and_keeps_compact_tokens():
    with TemporaryDirectory() as tmp_dir:
        merged = Path(tmp_dir) / "merged"
        merged.mkdir()
        merged.joinpath("nodes.tsv").write_text(
            "\n".join(
                [
                    "机器▂学习\ttag\t10\t2\t8\t0",
                    "孤例\tvocab\t1\t1\t0\t0",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        entries = load_nodes_vocab(merged, min_df=2)

        assert [entry.token for entry in entries] == ["机器▂学习"]
        assert entries[0].tag_ratio == 0.8


def test_sentence_from_doc_terms_uses_nodes_vocab_filter():
    with TemporaryDirectory() as tmp_dir:
        merged = Path(tmp_dir) / "merged"
        merged.mkdir()
        merged.joinpath("nodes.tsv").write_text(
            "\n".join(
                [
                    "显卡\ttag\t12\t1\t12\t0",
                    "价格\tvocab\t20\t12\t8\t0",
                    "标题碎片\tvocab\t20\t20\t0\t0",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        vocab = load_nodes_vocab_map(merged)

        sentence = sentence_from_doc_terms(
            (
                ("显卡", 2, 1.0),
                ("价格", 1, 0.9),
                ("标题碎片", 1, 0.8),
            ),
            FastTextV2CorpusConfig(vocab=vocab, min_terms_per_doc=2),
        )

        assert sentence == [
            "__video__",
            "__tag__",
            "显卡",
            "显卡",
            "__title__",
            "价格",
        ]


def test_iter_relation_sentences_keeps_filtered_high_confidence_edges():
    with TemporaryDirectory() as tmp_dir:
        merged = Path(tmp_dir) / "merged"
        merged.mkdir()
        merged.joinpath("nodes.tsv").write_text(
            "\n".join(
                [
                    "gpu\ttag\t10\t0\t10\t0",
                    "显卡\ttag\t10\t0\t10\t0",
                    "标题碎片\tvocab\t10\t10\t0\t0",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        merged.joinpath("synonym.tsv").write_text(
            "gpu\t显卡\t0.82\t标题碎片\t0.95\n",
            encoding="utf-8",
        )

        vocab = load_nodes_vocab_map(merged)
        sentences = list(
            iter_relation_sentences(
                merged,
                FastTextV2CorpusConfig(
                    vocab=vocab,
                    min_relation_weight=0.8,
                    relation_repeat=1,
                ),
            )
        )

        assert sentences == [["__synonym__", "gpu", "显卡"]]


class _FakeKeyedVectors:
    def __init__(self):
        self.vectors = np.asarray(
            [
                [2.0, 1.0],
                [1.8, 1.0],
                [-2.0, -1.0],
            ],
            dtype=np.float32,
        )
        self.data = {
            "gpu": self.vectors[0],
            "显卡": self.vectors[1],
            "洗地机": self.vectors[2],
        }

    def __getitem__(self, token):
        return self.data[token]


def test_candidate_scorer_ranks_fixed_candidates_with_centered_vectors():
    scorer = FastTextV2CandidateScorer(_FakeKeyedVectors())

    ranked = scorer.rank("gpu", ["洗地机", "显卡"])

    assert [item.term for item in ranked] == ["显卡", "洗地机"]
