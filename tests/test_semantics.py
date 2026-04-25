import json
import unittest

from pathlib import Path
from tempfile import TemporaryDirectory

from models.semantics.pipeline import SemanticPipeline
from models.semantics.embedding_filter import filter_mapping_by_similarity, hash_similarity
from models.semantics.storage import (
    _can_promote_near_synonym,
    _promote_embedding_semantic_bridges,
    merge_groups,
)


class SemanticsPipelineTests(unittest.TestCase):
    def test_near_synonym_promotion_requires_surface_variant(self):
        self.assertTrue(_can_promote_near_synonym("comfyui", "comfyu"))
        self.assertTrue(_can_promote_near_synonym("055大", "055大驱"))
        self.assertFalse(_can_promote_near_synonym("00后", "90后"))
        self.assertFalse(_can_promote_near_synonym("0分", "10分钟"))
        self.assertFalse(_can_promote_near_synonym("15分钟", "5分钟"))
        self.assertFalse(_can_promote_near_synonym("教程", "入门"))

    def test_build_merge_and_incremental_skip(self):
        docs = [
            {
                "bvid": "BV1",
                "title": "原神 新手 教程",
                "tags": ["原神", "教程"],
                "owner": {"name": "作者甲"},
            },
            {
                "bvid": "BV2",
                "title": "原神 入门 讲解",
                "tags": ["原神", "讲解"],
                "owner": {"name": "作者乙"},
            },
            {
                "bvid": "BV3",
                "title": "黑神话 评测",
                "tags": ["黑神话", "评测"],
                "owner": {"name": "作者丙"},
            },
        ]
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            vocab_path = root / "vocabs.txt"
            vocab_path.write_text(
                "原神\n教程\n入门\n讲解\n黑神话\n评测\n", encoding="utf-8"
            )

            pipeline = SemanticPipeline(
                output_root=root / "semantics",
                version="test",
                num_groups=2,
                workers=1,
                group_chunk_size=2,
                vocab_path=vocab_path,
            )
            first_stats = pipeline.submit_iter(iter(docs), log_every=0)
            self.assertEqual(first_stats["processed"], 3)
            self.assertEqual(first_stats["doc_rows"], 3)

            second_stats = pipeline.submit_iter(iter(docs), log_every=0)
            self.assertEqual(second_stats["processed"], 0)
            self.assertEqual(second_stats["skipped"], 3)

            merge_stats = merge_groups(
                root / "semantics" / "test",
                min_df=1,
                min_cooc=1,
                top_k=8,
                max_df_ratio=1.0,
                min_score=0.0,
                negative_samples_per_doc=4,
            )
            merged_dir = Path(merge_stats["output_dir"])
            self.assertTrue((merged_dir / "rewrite.tsv").exists())
            self.assertTrue((merged_dir / "doc_cooccurrence.tsv").exists())
            self.assertFalse((merged_dir / "edges.tsv").exists())
            self.assertIn(
                "原神",
                (merged_dir / "doc_cooccurrence.tsv").read_text(encoding="utf-8"),
            )
            self.assertTrue(
                (merged_dir / "negative_samples.tsv").read_text(encoding="utf-8")
            )

    def test_space_mask_and_broad_rare_filtering(self):
        docs = [
            {"bvid": "BV1", "title": "机器 学习 教程", "tags": ["机器 学习", "泛词"]},
            {"bvid": "BV2", "title": "机器 学习 入门", "tags": ["机器 学习", "泛词"]},
            {"bvid": "BV3", "title": "深度 学习 教程", "tags": ["深度 学习", "泛词"]},
            {
                "bvid": "BV4",
                "title": "深度 学习 入门",
                "tags": ["深度 学习", "泛词", "孤例"],
            },
        ]
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            vocab_path = root / "vocabs.txt"
            vocab_path.write_text(
                "机器 学习\n深度 学习\n教程\n入门\n泛词\n孤例\n", encoding="utf-8"
            )
            pipeline = SemanticPipeline(
                output_root=root / "semantics",
                version="filter",
                num_groups=2,
                workers=1,
                group_chunk_size=2,
                vocab_path=vocab_path,
            )
            pipeline.submit_iter(iter(docs), log_every=0)
            merged = merge_groups(
                root / "semantics" / "filter",
                min_df=2,
                min_cooc=1,
                max_df_ratio=0.75,
                min_score=0.0,
            )
            merged_dir = Path(merged["output_dir"])
            nodes_text = (merged_dir / "nodes.tsv").read_text(encoding="utf-8")
            self.assertIn("机器▂学习", nodes_text)
            self.assertIn("深度▂学习", nodes_text)
            self.assertNotIn("泛词", nodes_text)
            self.assertNotIn("孤例", nodes_text)

    def test_merge_uses_latest_incremental_doc_version(self):
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            vocab_path = root / "vocabs.txt"
            vocab_path.write_text("原神\n教程\n黑神话\n评测\n", encoding="utf-8")
            pipeline = SemanticPipeline(
                output_root=root / "semantics",
                version="incremental",
                num_groups=1,
                workers=1,
                group_chunk_size=1,
                vocab_path=vocab_path,
            )
            pipeline.submit_iter(
                iter([{"bvid": "BV1", "title": "原神 教程", "tags": ["原神"]}]),
                log_every=0,
            )
            pipeline.submit_iter(
                iter([{"bvid": "BV1", "title": "黑神话 评测", "tags": ["黑神话"]}]),
                log_every=0,
            )
            merged = merge_groups(
                root / "semantics" / "incremental",
                min_df=1,
                min_cooc=1,
                max_df_ratio=1.0,
                min_score=0.0,
            )
            merged_dir = Path(merged["output_dir"])
            nodes_text = (merged_dir / "nodes.tsv").read_text(encoding="utf-8")
            self.assertIn("黑神话", nodes_text)
            self.assertIn("评测", nodes_text)
            self.assertNotIn("原神", nodes_text)
            self.assertNotIn("教程\t", nodes_text)
            self.assertEqual(merged["current_docs"], 1)
            self.assertEqual(merged["doc_count"], 1)

    def test_jsonl_input_shape_is_plain_dicts(self):
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "docs.jsonl"
            path.write_text(
                json.dumps(
                    {"bvid": "BV1", "title": "教程", "tags": ["教程"]},
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(path.read_text(encoding="utf-8").count("\n"), 1)

    def test_embedding_filter_removes_low_similarity_cooccurrence(self):
        mapping = {
            "显卡": {"gpu": 0.9, "洗地机": 0.8, "硬件": 0.7},
            "价格": {"价格曲线": 0.9},
        }

        def similarity(source: str, targets: list[str]) -> list[float]:
            scores = {
                ("显卡", "gpu"): 0.72,
                ("显卡", "洗地机"): 0.31,
                ("显卡", "硬件"): 0.60,
                ("价格", "价格曲线"): 0.75,
            }
            return [scores.get((source, target), 0.0) for target in targets]

        filtered, stats = filter_mapping_by_similarity(
            mapping,
            similarity,
            min_score=0.52,
            cjk_min_score=0.72,
            mixed_script_min_score=0.62,
        )

        self.assertEqual(set(filtered["显卡"]), {"gpu"})
        self.assertEqual(set(filtered["价格"]), {"价格曲线"})
        self.assertEqual(stats.targets_seen, 4)
        self.assertEqual(stats.targets_removed, 2)

    def test_embedding_bridge_promotion_adds_only_verified_semantic_pairs(self):
        synonyms = {}
        cooccurrence = {
            "显卡": {"gpu": 0.92, "硬件": 0.88, "h20": 0.95},
            "价格": {"价格曲线": 0.9},
        }

        def similarity(source: str, targets: list[str]) -> list[float]:
            scores = {
                ("显卡", "gpu"): 0.73,
                ("显卡", "硬件"): 0.61,
                ("价格", "价格曲线"): 0.76,
            }
            return [scores.get((source, target), 0.0) for target in targets]

        stats = _promote_embedding_semantic_bridges(
            synonyms,
            cooccurrence,
            similarity,
            min_weight=0.72,
            min_score=0.52,
            cjk_min_score=0.72,
            mixed_script_min_score=0.62,
            max_sources=0,
            max_targets_per_source=8,
        )

        self.assertEqual(set(synonyms["显卡"]), {"gpu"})
        self.assertEqual(set(synonyms["价格"]), {"价格曲线"})
        self.assertNotIn("h20", synonyms["显卡"])
        self.assertEqual(stats["targets_promoted"], 2)

    def test_hash_similarity_uses_normalized_hamming_similarity(self):
        self.assertEqual(hash_similarity("ff", "ff"), 1.0)
        self.assertEqual(hash_similarity("00", "ff"), 0.0)
        self.assertEqual(hash_similarity("f0", "ff"), 0.5)


if __name__ == "__main__":
    unittest.main()
