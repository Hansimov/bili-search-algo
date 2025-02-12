import argparse
import concurrent.futures
import sys
import threading

from sedb import MilvusOperator
from tclogger import TCLogbar, FileLogger, logger, logstr, dict_to_str, brk
from tclogger import get_now_str, get_now_ts, ts_to_str
from typing import Literal, Union
from workers.milvus_videos.converter import MongoDocToMilvusDocConverter
from workers.milvus_videos.generator import MilvusVideoDocsGenerator

from configs.envs import MILVUS_ENVS


class MilvusVideoSubmitter:
    def __init__(self, collection: str = None, verbose: bool = False):
        self.collection = collection
        self.verbose = verbose
        self.init_milvus()

    def init_milvus(self):
        self.milvus = MilvusOperator(
            configs=MILVUS_ENVS,
            connect_msg=f"{logstr.mesg(self.__class__.__name__)} -> {logstr.mesg(brk('milvus'))}",
        )

    def submit(
        self,
        docs: Union[dict, list[dict]],
        collection: str = None,
        op_type: Literal["insert", "upsert"] = "upsert",
    ):
        collection = collection or self.collection
        if not collection:
            logger.err(f"× Empty collection!")
            return
        try:
            if op_type == "insert":
                self.milvus.client.insert(collection_name=collection, data=docs)
            elif op_type == "upsert":
                self.milvus.client.upsert(collection_name=collection, data=docs)
            else:
                logger.err(f"× Invalid op_type: [{op_type}]")
        except Exception as e:
            logger.err(f"× Submit error: {e}")


class MilvusVideoIndexer:
    def __init__(self, collection: str, verbose: bool = False):
        self.collection = collection
        self.verbose = verbose
        self.init_event()
        self.init_converter_submitter()

    def init_event(self):
        self.stop_event = threading.Event()
        self.complete_event = threading.Event()

    def init_converter_submitter(self):
        self.converter = MongoDocToMilvusDocConverter()
        self.submitter = MilvusVideoSubmitter(verbose=self.verbose)

    def index_milvus_docs(
        self,
        generator: MilvusVideoDocsGenerator,
        op_type: Literal["insert", "upsert"] = "upsert",
        dry_run: bool = False,
        quick_break: bool = False,
    ):
        logger.note(f"> Indexing docs:")
        if dry_run:
            logger.success(f"✓ Dry run done at: [{get_now_str()}]")
            return

        end_ts = get_now_ts()

        for batch_idx, docs_batch in enumerate(generator.batch_generator()):
            if self.stop_event.is_set():
                logger.warn(f"\n× Break indexer by stop event")
                break
            # if quick_break and doc["insert_at"] > end_ts:
            #     logger.warn(f"\n× Break indexer at start time: [{ts_to_str(end_ts)}]")
            #     break

            milvus_docs = self.converter.convert_batch(docs_batch)
            self.submitter.submit(
                milvus_docs, collection=self.collection, op_type=op_type
            )

        if not self.stop_event.is_set():
            logger.success(f"\n> Index completed at: [{get_now_str()}]")
            self.complete_event.set()


class MilvusVideoIndexerArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("-mg", "--mongo-collection", type=str, default="videos")
        self.add_argument("-ml", "--milvus-collection", type=str, default="videos")
        self.add_argument(
            "-f",
            "--filter-index",
            type=str,
            choices=["pubdate", "insert_at"],
            default="insert_at",
        )
        self.add_argument("-s", "--start-date", type=str, default=None)
        self.add_argument("-e", "--end-date", type=str, default=None)
        self.add_argument("-d", "--dry-run", action="store_true", default=False)
        self.add_argument("-c", "--max-count", type=int, default=None)

    def parse_args(self):
        self.args, self.unknown_args = self.parse_known_args(sys.argv[1:])
        return self.args


def main(args: argparse.Namespace):
    indexer = MilvusVideoIndexer(args.milvus_collection)
    generator = MilvusVideoDocsGenerator(max_count=args.max_count)
    generator.init_cursor(
        milvus_collection=args.milvus_collection,
        mongo_collection=args.mongo_collection,
        filter_index=args.filter_index,
        filter_op="range",
        filter_range=[args.start_date, args.end_date],
        estimate_count=False,
    )
    indexer.index_milvus_docs(
        generator,
        op_type="upsert",
        dry_run=args.dry_run,
        quick_break=True,
    )


if __name__ == "__main__":
    arg_parser = MilvusVideoIndexerArgParser()
    args = arg_parser.parse_args()
    main(args)

    # python -m workers.milvus_videos.indexer -f pubdate -s "2025-02-10" -e "2025-02-12" -c 50000
