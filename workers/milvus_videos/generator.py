import concurrent.futures
import threading

from sedb import MongoOperator, MilvusOperator, to_mongo_filter
from tclogger import TCLogbar, FileLogger, logger, logstr, dict_to_str, brk
from tclogger import get_now_str, get_now_ts, ts_to_str
from typing import Literal, Union
from collections.abc import Generator

from configs.envs import MONGO_ENVS, MILVUS_ENVS


class MilvusVideoDocsGenerator:
    def __init__(
        self, max_count: int = None, batch_size: int = 1000, verbose: bool = False
    ):
        self.max_count = max_count
        self.batch_size = batch_size
        self.verbose = verbose
        self.init_mongo()
        self.init_milvus()
        self.init_progress_bar()

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

    def init_progress_bar(self):
        self.sample_bar = TCLogbar(head=logstr.note("* Sample:"))

    def init_cursor(
        self,
        milvus_collection: str = "videos",
        mongo_collection: str = "videos",
        filter_index: Literal["insert_at", "pubdate"] = "insert_at",
        filter_op: Literal["gt", "lt", "gte", "lte", "range"] = "gte",
        filter_range: Union[int, str, tuple, list] = None,
        sort_index: Literal["insert_at", "pubdate"] = "insert_at",
        sort_order: Literal["asc", "desc"] = "asc",
        estimate_count: bool = False,
    ):
        """This must be called before `batch_generator`."""
        self.milvus_collection = milvus_collection
        self.total_count = self.mongo.get_total_count(
            collection=mongo_collection,
            filter_index=filter_index,
            filter_op=filter_op,
            filter_range=filter_range,
            estimate_count=estimate_count,
        )

        if self.total_count == 0:
            logger.warn(f"× No docs found")
            self.cursor = None
            return

        self.total_count = self.max_count or self.total_count
        self.sample_bar.set_total(self.total_count)

        self.cursor = self.mongo.get_cursor(
            collection=mongo_collection,
            filter_index=filter_index,
            filter_op=filter_op,
            filter_range=filter_range,
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
