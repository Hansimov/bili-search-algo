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

    def get_bvids(self, docs: list[dict]) -> list[str]:
        return [doc["bvid"] for doc in docs]

    def fetch_bvids_to_update(
        self,
        bvids: list[str],
        check_fields: list[Literal["title_status", "tags_status"]] = None,
    ) -> list[str]:
        expr_of_any_field_false = self.milvus.get_expr_of_any_field_false(
            fields=check_fields
        )
        expr_of_bvids = self.milvus.get_expr_of_list_contain("bvid", bvids)
        expr_of_bvids_to_update = f"({expr_of_bvids}) AND ({expr_of_any_field_false})"

        bvids_to_update = self.milvus.client.query(
            collection_name=self.milvus_collection,
            filter=expr_of_bvids_to_update,
            output_fields=["bvid"],
        )
        return bvids_to_update

    def batch_generator(self) -> Generator[list[dict], None, None]:
        batch = []
        for idx, doc in enumerate(self.cursor):
            if self.max_count and idx >= self.max_count:
                break
            self.sample_bar.update(increment=1)
            batch.append(doc)
            if len(batch) >= self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
