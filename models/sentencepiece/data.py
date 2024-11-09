from sedb import MongoOperator
from tclogger import logger, logstr, TCLogbar, TCLogbarGroup, dict_get
from typing import Literal, Generator

from configs.envs import MONGO_ENVS


class VideosTagsAggregator:
    def __init__(
        self,
        videos_collect_name: str = "videos",
        tags_collect_name: str = "videos_tags",
        tags_join_name: str = "tagger",
    ):
        self.videos_collect_name = videos_collect_name
        self.tags_collect_name = tags_collect_name
        self.tags_join_name = tags_join_name
        self.init_pipeline()
        self.init_mongo()
        self.init_cursor()

    def init_pipeline(self):
        self.pipeline = [
            {
                "$lookup": {
                    "from": self.tags_collect_name,
                    "localField": "bvid",
                    "foreignField": "bvid",
                    "as": self.tags_join_name,
                }
            },
            {"$unwind": f"${self.tags_join_name}"},
            {
                "$project": {
                    "title": 1,
                    "desc": 1,
                    "owner.name": 1,
                    "tid": 1,
                    f"{self.tags_join_name}.tags": 1,
                    f"{self.tags_join_name}.region_tags": 1,
                }
            },
        ]

    def init_mongo(self):
        self.mongo = MongoOperator(
            MONGO_ENVS, connect_msg=f"from {self.__class__.__name__}", indent=2
        )
        self.videos_collect = self.mongo.db[self.videos_collect_name]
        self.videos_estimated_count = self.videos_collect.estimated_document_count()

    def init_cursor(self):
        self.cursor = self.videos_collect.aggregate(self.pipeline, allowDiskUse=True)

    def __iter__(self):
        for doc in self.cursor:
            yield doc


class SentencesDataloader:

    def __init__(
        self,
        videos_collect_name: str = "videos",
        tags_collect_name: str = "videos_tags",
        batch_size: int = 10000,
        max_batch: int = None,
        iter_val: Literal["doc", "sentence"] = "sentence",
        iter_epochs: int = None,
    ):
        self.videos_collect_name = videos_collect_name
        self.tags_collect_name = tags_collect_name
        self.batch_size = batch_size
        self.max_batch = max_batch
        self.iter_val = iter_val
        self.iter_epochs = iter_epochs
        self.iter_epoch = 0
        self.init_mongo()
        self.init_progress_bars()

    def init_progress_bars(self):
        self.epoch_bar = TCLogbar(head=logstr.note("> Epoch:"))
        self.batch_bar = TCLogbar(head=logstr.note("  * Batch:"))
        self.sample_bar = TCLogbar(head=logstr.note("  * Sample:"))
        TCLogbarGroup([self.epoch_bar, self.batch_bar, self.sample_bar])

    def init_mongo(self):
        self.mongo = MongoOperator(
            MONGO_ENVS, connect_msg=f"from {self.__class__.__name__}", indent=2
        )
        self.videos_collect = self.mongo.db[self.videos_collect_name]
        self.videos_cursor = self.mongo.get_cursor(self.videos_collect_name)
        self.store_cursor()

    def store_cursor(self):
        self.videos_cursor_unevaluated = self.videos_cursor.clone()

    def restore_cursor(self):
        self.videos_cursor = self.videos_cursor_unevaluated.clone()

    def __epoch_end__(self):
        self.epoch_bar.update(increment=1)
        self.batch_bar.reset()
        self.sample_bar.reset()

    def doc_batch(self) -> Generator[dict, None, None]:
        while True:
            videos_batch = []
            videos_count = 0
            for doc in self.videos_cursor:
                videos_batch.append(doc)
                videos_count += 1
                if videos_count >= self.batch_size:
                    videos_count = 0
                    break
            if not videos_batch:
                break
            bvids = [doc["bvid"] for doc in videos_batch]
            filter = {"bvid": {"$in": bvids}}
            projection = {"bvid": 1, "tags": 1, "region_tags": 1}
            tags_cursor = self.mongo.db[self.tags_collect_name].find(filter, projection)
            tags_dict = {doc["bvid"]: doc for doc in tags_cursor}
            res = {}
            self.sample_bar.total = len(videos_batch)
            for doc in videos_batch:
                bvid = doc["bvid"]
                tags_doc = tags_dict.get(bvid, {})
                combined_doc = {**doc, **tags_doc}
                res[bvid] = combined_doc
                self.sample_bar.update(increment=1)
            yield res

    def doc_to_sentence(self, doc: dict) -> str:
        title = doc.get("title", "")
        desc = doc.get("desc", "")
        author = doc.get("owner", {}).get("name", "")
        region_tags = doc.get("region_tags", "")
        tags = doc.get("tags", "")
        sentence = f"Title: {title}. Desc: {desc}. Author: {author}. Tags: {region_tags}, {tags}"
        return sentence

    def __iter__(self):
        self.epoch_bar.total = self.iter_epochs or 1
        self.batch_bar.total = (
            int(self.videos_collect.estimated_document_count() / self.batch_size) + 1
        )
        for batch_idx, batch in enumerate(self.doc_batch()):
            if self.max_batch is not None and batch_idx >= self.max_batch:
                break
            for bvid, doc in batch.items():
                if self.iter_val == "sentence":
                    res = self.doc_to_sentence(doc)
                else:
                    res = doc
                yield res
            self.sample_bar.reset()
            self.batch_bar.update(increment=1)
        self.__epoch_end__()


if __name__ == "__main__":
    loader = SentencesDataloader(batch_size=10000)
    for doc in loader:
        continue

    # python -m models.sentencepiece.data
