from dataclasses import dataclass

from sedb import MongoOperator

from configs.envs import MONGO_ENVS
from models.coretok.core import CoreImpEvaluator, CoreTagTokenizer, CoreTexTokenizer


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

    def train_stage1(
        self,
        tags: list[str],
        epochs: int = 3,
    ) -> list[list[int]]:
        self.tag_tokenizer = self.tag_tokenizer or CoreTagTokenizer()
        return self.tag_tokenizer.fit(tags, epochs=epochs)

    def train_stage2(
        self,
        texts: list[str],
        epochs: int = 1,
    ) -> list[list[int]]:
        seed_lexicon = None
        if self.tag_tokenizer is not None:
            seed_lexicon = self.tag_tokenizer.lexicon
        self.text_tokenizer = self.text_tokenizer or CoreTexTokenizer(
            lexicon=seed_lexicon
        )
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
