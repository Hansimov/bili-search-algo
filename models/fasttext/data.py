from sedb import MongoOperator
from tclogger import logger, logstr, ts_to_str
from tqdm import tqdm
from typing import Literal

from configs.envs import MONGO_ENVS


class DataProgressBar:
    def __init__(self, total: int = None):
        self.total = total
        self.init_bar()

    def init_bar(self):
        self.bar = tqdm(
            total=self.total,
            bar_format="  {desc}{percentage:3.0f}%|{bar:25}{r_bar}",
        )
        print()

    def update(self, desc: str = None, count: int = None):
        if desc is not None:
            self.bar.set_description(desc)
        if count is not None:
            self.bar.update(count)

    def reset(self):
        self.bar.refresh()
        print()
        self.bar.n = 0
        self.bar.total = self.total


class VideosTagsDataLoader:
    def __init__(
        self,
        collection: str = "videos_tags",
        max_count: int = None,
        iter_val: Literal["doc", "tag_list"] = "tag_list",
        iter_log: bool = False,
    ):
        self.collection = collection
        self.init_mongo()
        self.max_count = max_count
        self.iter_val = iter_val
        self.iter_log = iter_log
        if iter_log:
            self.progress_bar = DataProgressBar(total=max_count)

    def init_mongo(self):
        self.mongo = MongoOperator(
            MONGO_ENVS,
            connect_msg=f"from {self.__class__.__name__}",
        )
        self.cursor = self.mongo.get_cursor(self.collection)
        self.store_cursor()

    def store_cursor(self):
        self.cursor_unevaluated = self.cursor.clone()

    def restore_cursor(self):
        self.cursor = self.cursor_unevaluated.clone()
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
                    desc = f"{doc['bvid']}"
                    self.progress_bar.update(desc=desc, count=1)

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
