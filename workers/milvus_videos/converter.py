import concurrent.futures
import numpy as np

from tclogger import dict_get

from models.vectors.forms import padding_zeros
from workers.milvus_videos.schema import DOCVEC_DIM, KEEP_COLUMNS

ZEROS_DOCVEC = np.zeros(DOCVEC_DIM).astype(np.float16)


class MongoDocToMilvusDocConverter:
    STAT_KEYS = ["view", "danmaku", "reply", "favorite", "coin", "share", "like"]

    def __init__(self, max_workers: int = 32):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    def stats_to_list(self, stats: dict) -> list[int]:
        return [int(stats.get(key, 0)) for key in self.STAT_KEYS]

    def text_to_embedding(self, text: str) -> list[float]:
        vec = padding_zeros(
            np.array([ord(char) for char in text][:DOCVEC_DIM]), DOCVEC_DIM
        )
        vec = (vec / np.linalg.norm(vec)).astype(np.float16)
        return vec

    def get_kept_dict(self, doc: dict) -> dict:
        return {col.replace(".", "_"): dict_get(doc, col) for col in KEEP_COLUMNS}

    def convert(self, doc: dict) -> dict:
        milvus_doc = {
            "bvid": doc["bvid"],
            **self.get_kept_dict(doc),
            "stats_arr": self.stats_to_list(doc["stat"]),
            "title_vec": self.text_to_embedding(doc["title"]),
            "title_status": 1,
            "tags_vec": ZEROS_DOCVEC,
            "title_tags_owner_vec": ZEROS_DOCVEC,
            "title_tags_owner_desc_vec": ZEROS_DOCVEC,
            "tags_status": 0,
        }
        return milvus_doc

    def convert_batch(self, docs: list[dict]) -> list[dict]:
        futures = [self.executor.submit(self.convert, doc) for doc in docs]
        return [future.result() for future in futures]

