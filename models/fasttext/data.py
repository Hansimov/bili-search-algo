from sedb import MongoOperator
from tclogger import logger, logstr, ts_to_str
from typing import Literal

from configs.envs import MONGO_ENVS


class VideosTagsDataLoader:
    def __init__(
        self,
        collection: str = "videos_tags",
        max_count: int = 1000000,
        iter_val: Literal["doc", "tag_list"] = "tag_list",
    ):
        self.collection = collection
        self.init_mongo()
        self.max_count = max_count
        self.iter_val = iter_val

    def init_mongo(self):
        self.mongo = MongoOperator(
            MONGO_ENVS,
            connect_msg=f"from {self.__class__.__name__}",
        )
        self.cursor = self.mongo.get_cursor(self.collection)

    def get_tag_list(self, doc):
        return [tag.strip() for tag in doc["tags"].split(",")]

    def __iter__(self):
        for idx, doc in enumerate(self.cursor):
            if self.max_count is not None and idx >= self.max_count:
                break
            else:
                if self.iter_val == "tag_list":
                    yield self.get_tag_list(doc)
                else:
                    yield doc


if __name__ == "__main__":
    from tqdm import tqdm

    max_count = 10000
    loader = VideosTagsDataLoader(max_count=max_count, iter_val="doc")
    progress_bar = tqdm(
        loader,
        total=max_count,
        bar_format="{desc}{percentage:3.0f}%|{bar:25}{r_bar}",
    )

    logger.note(f"\n> Iterating over {max_count} documents:")

    for doc in progress_bar:
        if doc is None:
            break
        else:
            bvid = doc["bvid"]
            tag = doc["tags"].split(",")[0].strip()
            insert_at_str = ts_to_str(doc["insert_at"])
            progress_bar.set_description(
                f"[{logstr.mesg(insert_at_str)}]: {logstr.mesg(bvid)}"
            )

    # python -m models.fasttext.data
