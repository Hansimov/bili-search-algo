from sedb import MongoOperator
from tclogger import logger, logstr, TCLogbar, ts_to_str
from typing import Literal

from configs.envs import MONGO_ENVS


class DataProgressBar:
    def __init__(self, total: int = None):
        self.total = total
        self.init_bar()

    def init_bar(self):
        self.bar = TCLogbar(total=self.total, flush_interval=0.25)

    def update(self, increment: int = None, head: str = None, desc: str = None):
        if increment is not None:
            self.bar.update(increment)
        if head is not None:
            self.bar.set_head(head)
        if desc is not None:
            self.bar.set_desc(desc)

    def reset(self):
        self.bar.reset()


class VideosTagsDataLoader:
    def __init__(
        self,
        collection: str = "videos_tags",
        max_count: int = None,
        iter_val: Literal["doc", "tag_list"] = "tag_list",
        iter_log: bool = False,
        iter_epochs: int = None,
    ):
        self.collection = collection
        self.init_mongo()
        self.max_count = max_count
        self.iter_val = iter_val
        self.iter_log = iter_log
        self.iter_epochs = iter_epochs
        self.iter_epoch = 0
        if iter_log:
            self.progress_bar = DataProgressBar(total=max_count)

    def init_mongo(self):
        self.mongo = MongoOperator(
            MONGO_ENVS, connect_msg=f"from {self.__class__.__name__}", indent=2
        )
        self.cursor = self.mongo.get_cursor(self.collection)
        self.store_cursor()

    def store_cursor(self):
        self.cursor_unevaluated = self.cursor.clone()

    def restore_cursor(self):
        self.cursor = self.cursor_unevaluated.clone()
        if self.iter_epochs is not None:
            self.iter_epoch += 1
        self.progress_bar.reset()

    def get_tag_list(self, doc):
        return [tag.strip() for tag in doc["tags"].split(",")]

    def __iter__(self):
        for idx, doc in enumerate(self.cursor):
            if self.max_count is not None and idx >= self.max_count:
                break
            else:
                if self.iter_val == "tag_list":
                    res = self.get_tag_list(doc)
                else:
                    res = doc

                if self.iter_log and self.progress_bar:
                    if self.iter_epochs is not None:
                        epoch_str = f"[{logstr.file(self.iter_epoch+1)}/{logstr.mesg(self.iter_epochs)}]"
                    else:
                        epoch_str = ""
                    head = f"* {epoch_str}"
                    desc = f"{doc['bvid']}"
                    self.progress_bar.update(increment=1, head=head, desc=desc)
                yield res
        self.restore_cursor()


if __name__ == "__main__":
    from tqdm import tqdm

    max_count = 10000
    loader = VideosTagsDataLoader(max_count=max_count, iter_val="doc", iter_log=True)
    logger.note(f"> Iterating over documents: [{logstr.mesg(max_count)}]")

    for doc in loader:
        if doc is None:
            break
        else:
            pass

    # python -m models.fasttext.data
