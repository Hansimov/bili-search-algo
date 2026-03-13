import unittest

from pathlib import Path

from models.sentencepiece.merge import (
    ModelSource,
    PieceAggregate,
    SentencePieceModelMerger,
)


class SentencePieceMergeTests(unittest.TestCase):
    def setUp(self):
        self.merger = SentencePieceModelMerger(
            model_paths=[],
            output_path=Path("/tmp/sp_merged_test.model"),
        )
        self.video_source_a = ModelSource(
            name="sp_908m_cine_movie",
            path=Path("/tmp/a.model"),
            vocab_size=1000,
            trunc_count=900,
            is_wiki=False,
        )
        self.video_source_b = ModelSource(
            name="sp_908m_music_dance",
            path=Path("/tmp/b.model"),
            vocab_size=1000,
            trunc_count=900,
            is_wiki=False,
        )
        self.wiki_source = ModelSource(
            name="sp_wiki_8m_400k",
            path=Path("/tmp/wiki.model"),
            vocab_size=1000,
            trunc_count=900,
            is_wiki=True,
        )
        self.merger.sources = [
            self.video_source_a,
            self.video_source_b,
            self.wiki_source,
        ]

    def test_drop_wiki_only_ascii_piece(self):
        aggregate = PieceAggregate("archive-url")
        aggregate.add(self.wiki_source, rank_score=0.99, normalized_score=0.95)
        self.assertTrue(self.merger.should_drop_aggregate(aggregate))

    def test_keep_supported_video_ascii_piece(self):
        aggregate = PieceAggregate("youtube")
        aggregate.add(self.video_source_a, rank_score=0.99, normalized_score=0.95)
        aggregate.add(self.video_source_b, rank_score=0.98, normalized_score=0.92)
        self.assertFalse(self.merger.should_drop_aggregate(aggregate))

    def test_cjk_piece_scores_higher_than_ascii_for_same_support(self):
        cjk_aggregate = PieceAggregate("王者荣耀")
        cjk_aggregate.add(self.video_source_a, rank_score=0.9, normalized_score=0.8)
        cjk_aggregate.add(self.video_source_b, rank_score=0.88, normalized_score=0.78)

        ascii_aggregate = PieceAggregate("youtube")
        ascii_aggregate.add(self.video_source_a, rank_score=0.9, normalized_score=0.8)
        ascii_aggregate.add(self.video_source_b, rank_score=0.88, normalized_score=0.78)

        self.assertGreater(
            self.merger.calc_piece_score(cjk_aggregate),
            self.merger.calc_piece_score(ascii_aggregate),
        )


if __name__ == "__main__":
    unittest.main()
