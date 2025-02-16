from sedb import MongoOperator, MilvusOperator
from sedb import to_mongo_pipeline
from sedb import MilvusBridger
from tclogger import logger, logstr, brk, dict_get, TCLogbar
from typing import Literal, Union
from collections.abc import Generator

from configs.envs import MONGO_ENVS, MILVUS_ENVS
from workers.milvus_videos.constants import (
    MONGO_VIDEOS_COLLECTION,
    MILVUS_VIDEOS_COLLECTION,
    MONGO_VIDEOS_TAGS_COLLECTION,
    MONGO_VIDEOS_TAGS_AGG_AS_NAME,
    MONGO_VIDEOS_COLLECTION_ID,
    MONGO_VIDEOS_TAGS_COLLECTION_ID,
    MILVUS_VIDEOS_COLLECTION_ID,
    MONGO_VIDEOS_FIELDS,
    MONGO_VIDEOS_TAGS_FIELDS,
)


class MilvusVideoDocsGenerator:
    def __init__(
        self,
        max_count: int = None,
        batch_size: int = 10000,
        agg_batch_size: int = 1000,
        verbose: bool = False,
    ):
        self.max_count = max_count
        self.batch_size = batch_size
        self.agg_batch_size = agg_batch_size
        self.verbose = verbose
        self.init_mongo()
        self.init_milvus()

    def init_mongo(self):
        self.mongo = MongoOperator(
            configs=MONGO_ENVS,
            connect_msg=f"{logstr.mesg(self.__class__.__name__)} -> {logstr.mesg(brk('mongo'))}",
        )

    def init_milvus(self):
        self.milvus = MilvusOperator(
            configs=MILVUS_ENVS,
            connect_msg=f"{logstr.mesg(self.__class__.__name__)} -> {logstr.mesg(brk('milvus'))}",
        )
        self.milvus_bridger = MilvusBridger(self.milvus)

    def init_mongo_cursor(
        self,
        collection: str = MONGO_VIDEOS_COLLECTION,
        filter_index: Literal["insert_at", "pubdate"] = "insert_at",
        filter_op: Literal["gt", "lt", "gte", "lte", "range"] = "gte",
        filter_range: Union[int, str, tuple, list] = None,
        include_fields: list[str] = MONGO_VIDEOS_FIELDS,
        exclude_fields: list[str] = None,
        sort_index: Literal["insert_at", "pubdate"] = "insert_at",
        sort_order: Literal["asc", "desc"] = "asc",
        estimate_count: bool = False,
    ):
        """This must be called before using `batch_generator`."""
        self.total_count = self.mongo.get_total_count(
            collection=collection,
            filter_index=filter_index,
            filter_op=filter_op,
            filter_range=filter_range,
            estimate_count=estimate_count,
        )

        if self.total_count == 0:
            logger.warn(f"× No docs found")
            self.cursor = None
            return

        if self.max_count:
            self.total_count = min(self.total_count, self.max_count)

        self.cursor = self.mongo.get_cursor(
            collection=collection,
            filter_index=filter_index,
            filter_op=filter_op,
            filter_range=filter_range,
            include_fields=include_fields,
            exclude_fields=exclude_fields,
            sort_index=sort_index,
            sort_order=sort_order,
        )

    def init_progress_bar(self):
        self.sample_bar = TCLogbar(desc=logstr.note("  * Sample:"))
        self.sample_bar.set_total(self.total_count)
        self.sample_bar.update(flush=True)

    def create_mongo_agg_cursor(
        self,
        bvids: list[Union[str, int]],
        local_filter_dict: dict = None,
        foreign_filter_dict: dict = None,
    ):
        if not bvids:
            return []
        pipeline_params = {
            "local_collection": MILVUS_VIDEOS_COLLECTION,
            "foreign_collection": MONGO_VIDEOS_TAGS_COLLECTION,
            "local_id_field": MONGO_VIDEOS_COLLECTION_ID,
            "foreign_id_field": MONGO_VIDEOS_TAGS_COLLECTION_ID,
            "local_fields": MONGO_VIDEOS_FIELDS,
            "foreign_fields": MONGO_VIDEOS_TAGS_FIELDS,
            "must_in_local_ids": bvids,
            "must_in_foreign_ids": None,
            "must_have_local_fields": None,
            "must_have_foreign_fields": ["tags"],
            "local_filter_dict": local_filter_dict,
            "foreign_filter_dict": foreign_filter_dict,
            "as_name": MONGO_VIDEOS_TAGS_AGG_AS_NAME,
        }
        pipeline = to_mongo_pipeline(**pipeline_params)
        # logger.hint(dict_to_str(pipeline), indent=2)
        agg_cursor = self.mongo.get_agg_cursor(MILVUS_VIDEOS_COLLECTION, pipeline)
        return agg_cursor

    def get_bvids(self, docs: list[dict]) -> list[str]:
        return [doc["bvid"] for doc in docs]

    def filter_bvids(self, bvids: list[str]) -> tuple[list[str], list[str]]:
        vectorized_docs = self.milvus_bridger.filter_ids(
            collection_name=MILVUS_VIDEOS_COLLECTION,
            ids=bvids,
            id_field=MILVUS_VIDEOS_COLLECTION_ID,
            expr="vectorized_status == 1",
            output_fields=[MILVUS_VIDEOS_COLLECTION_ID],
        )
        vectorized_bvids = self.get_bvids(vectorized_docs)
        non_vectorized_bvids = list(set(bvids) - set(vectorized_bvids))
        return vectorized_bvids, non_vectorized_bvids

    def batch_generator(self) -> Generator[list[dict], None, None]:
        batch = []
        for idx, doc in enumerate(self.cursor):
            if self.max_count and idx >= self.max_count:
                break
            batch.append(doc)
            if len(batch) >= self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
