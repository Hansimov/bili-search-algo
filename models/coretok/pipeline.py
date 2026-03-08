import json

from dataclasses import dataclass
from pathlib import Path

from sedb import MongoOperator

from configs.envs import MONGO_ENVS
from models.coretok.core import (
    CoreCorpusStats,
    CoreImpEvaluator,
    CoreTagTokenizer,
    CoreTexTokenizer,
    CoreTokenLexicon,
)


VIDEO_TEXT_PROJECTION = {
    "_id": 0,
    "tags": 1,
    "title": 1,
    "desc": 1,
}


class MongoVideoTextStream:
    def __init__(self, db_name: str = "bili", collection: str = "videos"):
        self.mongo = MongoOperator(
            configs=MONGO_ENVS,
            connect_cls=self.__class__,
            verbose_args=False,
        )
        self.collection = self.mongo.client[db_name][collection]

    def iter_tag_texts(self, query: dict | None = None, limit: int | None = None):
        cursor = self.collection.find(
            query or {"tags": {"$exists": True, "$ne": ""}},
            VIDEO_TEXT_PROJECTION,
        )
        if limit:
            cursor = cursor.limit(limit)
        for doc in cursor:
            tags = (doc.get("tags") or "").split(",")
            for tag in tags:
                normalized = tag.strip()
                if normalized:
                    yield normalized

    def iter_title_desc_texts(
        self,
        query: dict | None = None,
        limit: int | None = None,
    ):
        cursor = self.collection.find(
            query
            or {
                "$or": [
                    {"title": {"$exists": True, "$ne": ""}},
                    {"desc": {"$exists": True, "$ne": ""}},
                ]
            },
            VIDEO_TEXT_PROJECTION,
        )
        if limit:
            cursor = cursor.limit(limit)
        for doc in cursor:
            title = (doc.get("title") or "").strip()
            desc = (doc.get("desc") or "").strip()
            if title:
                yield title
            if desc:
                yield desc


@dataclass
class CoreTokTrainingPipeline:
    tag_tokenizer: CoreTagTokenizer | None = None
    text_tokenizer: CoreTexTokenizer | None = None
    importance: CoreImpEvaluator | None = None
    tag_corpus_stats: CoreCorpusStats | None = None
    text_corpus_stats: CoreCorpusStats | None = None

    def train_stage1(
        self,
        tags: list[str],
        epochs: int = 3,
    ) -> list[list[int]]:
        self.tag_corpus_stats = self.tag_corpus_stats or CoreCorpusStats().fit(
            tags,
            for_stage1=True,
        )
        self.tag_tokenizer = self.tag_tokenizer or CoreTagTokenizer(
            corpus_stats=self.tag_corpus_stats
        )
        self.tag_tokenizer.corpus_stats = self.tag_corpus_stats
        return self.tag_tokenizer.fit(tags, epochs=epochs)

    def train_stage2(
        self,
        texts: list[str],
        epochs: int = 1,
    ) -> list[list[int]]:
        self.text_corpus_stats = self.text_corpus_stats or CoreCorpusStats().fit(
            texts,
            for_stage1=False,
        )
        seed_lexicon = None
        if self.tag_tokenizer is not None:
            seed_lexicon = self.tag_tokenizer.lexicon
        self.text_tokenizer = self.text_tokenizer or CoreTexTokenizer(
            lexicon=seed_lexicon,
            corpus_stats=self.text_corpus_stats,
        )
        self.text_tokenizer.corpus_stats = self.text_corpus_stats
        return self.text_tokenizer.fit(texts, epochs=epochs)

    def train_importance(
        self,
        tag_token_sequences: list[list[int]],
        text_token_sequences: list[list[int]],
    ) -> CoreImpEvaluator:
        self.importance = CoreImpEvaluator().fit(
            tag_token_sequences=tag_token_sequences,
            text_token_sequences=text_token_sequences,
        )
        return self.importance

    def build_bundle(self, bundle_version: str = "coretok-v1") -> dict:
        if self.text_tokenizer is not None:
            lexicon = self.text_tokenizer.lexicon
        elif self.tag_tokenizer is not None:
            lexicon = self.tag_tokenizer.lexicon
        else:
            lexicon = CoreTokenLexicon()

        return {
            "bundle_version": bundle_version,
            "lexicon": lexicon.to_dict(),
            "tag_tokenizer": {
                "novelty_threshold": getattr(
                    self.tag_tokenizer, "novelty_threshold", 0.55
                ),
                "reuse_threshold": getattr(self.tag_tokenizer, "reuse_threshold", 0.5),
                "source": "tag",
            },
            "tag_corpus_stats": (
                self.tag_corpus_stats.to_dict()
                if self.tag_corpus_stats is not None
                else CoreCorpusStats().to_dict()
            ),
            "text_tokenizer": {
                "novelty_threshold": getattr(
                    self.text_tokenizer, "novelty_threshold", 0.82
                ),
                "reuse_threshold": getattr(self.text_tokenizer, "reuse_threshold", 0.6),
                "source": "text",
            },
            "text_corpus_stats": (
                self.text_corpus_stats.to_dict()
                if self.text_corpus_stats is not None
                else CoreCorpusStats().to_dict()
            ),
            "importance": (
                self.importance.to_dict()
                if self.importance is not None
                else CoreImpEvaluator().to_dict()
            ),
        }

    def save_bundle(
        self,
        bundle_path: str | Path,
        bundle_version: str = "coretok-v1",
    ) -> Path:
        path = Path(bundle_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                self.build_bundle(bundle_version=bundle_version),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return path

    @classmethod
    def from_bundle_dict(cls, payload: dict | None) -> "CoreTokTrainingPipeline":
        payload = payload or {}
        lexicon = CoreTokenLexicon.from_dict(payload.get("lexicon"))
        tag_cfg = payload.get("tag_tokenizer") or {}
        text_cfg = payload.get("text_tokenizer") or {}
        tag_corpus_stats = CoreCorpusStats.from_dict(payload.get("tag_corpus_stats"))
        text_corpus_stats = CoreCorpusStats.from_dict(payload.get("text_corpus_stats"))
        pipeline = cls(
            tag_tokenizer=CoreTagTokenizer(
                lexicon=lexicon,
                corpus_stats=tag_corpus_stats,
            ),
            text_tokenizer=CoreTexTokenizer(
                lexicon=lexicon,
                corpus_stats=text_corpus_stats,
            ),
            importance=CoreImpEvaluator.from_dict(payload.get("importance")),
            tag_corpus_stats=tag_corpus_stats,
            text_corpus_stats=text_corpus_stats,
        )
        pipeline.tag_tokenizer.novelty_threshold = float(
            tag_cfg.get("novelty_threshold", pipeline.tag_tokenizer.novelty_threshold)
        )
        pipeline.tag_tokenizer.reuse_threshold = float(
            tag_cfg.get("reuse_threshold", pipeline.tag_tokenizer.reuse_threshold)
        )
        pipeline.text_tokenizer.novelty_threshold = float(
            text_cfg.get("novelty_threshold", pipeline.text_tokenizer.novelty_threshold)
        )
        pipeline.text_tokenizer.reuse_threshold = float(
            text_cfg.get("reuse_threshold", pipeline.text_tokenizer.reuse_threshold)
        )
        return pipeline

    @classmethod
    def from_bundle_path(cls, bundle_path: str | Path) -> "CoreTokTrainingPipeline":
        payload = json.loads(Path(bundle_path).read_text(encoding="utf-8"))
        return cls.from_bundle_dict(payload)
