import argparse
import sys
import threading

from tclogger import logger
from tclogger import get_now_str
from typing import Literal, Union

from workers.milvus_videos.constants import MONGO_VIDEOS_COLLECTION
from workers.milvus_videos.constants import MILVUS_VIDEOS_COLLECTION
from workers.milvus_videos.converter import MongoDocToMilvusDocConverter
from workers.milvus_videos.generator import MilvusVideoDocsGenerator
from workers.milvus_videos.submitter import MilvusVideoSubmitter


class MilvusVideoIndexer:
    def __init__(
        self,
        collection: str,
        runner_mode: Literal["local", "remote"] = "local",
        verbose: bool = False,
    ):
        self.collection = collection
        self.runner_mode = runner_mode
        self.verbose = verbose
        self.init_event()
        self.init_converter_submitter()

    def init_event(self):
        self.stop_event = threading.Event()
        self.complete_event = threading.Event()

    def init_converter_submitter(self):
        self.converter = MongoDocToMilvusDocConverter(runner_mode=self.runner_mode)
        self.submitter = MilvusVideoSubmitter(verbose=self.verbose)

    def index_milvus_docs(
        self,
        generator: MilvusVideoDocsGenerator,
        op_type: Literal["insert", "upsert"] = "upsert",
        dry_run: bool = False,
    ):
        logger.note(f"> Indexing docs:")
        if dry_run:
            logger.success(f"✓ Dry run done at: [{get_now_str()}]")
            return

        for batch_idx, doc_batch in enumerate(generator.agg_batch_generator()):
            if self.stop_event.is_set():
                logger.warn(f"\n× Break indexer by stop event")
                break
            milvus_docs = self.converter.convert_batch(doc_batch)
            self.submitter.submit(
                milvus_docs, collection=self.collection, op_type=op_type
            )

        if not self.stop_event.is_set():
            logger.success(f"> Index completed at: [{get_now_str()}]")
            self.complete_event.set()


class MilvusVideoIndexerArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument(
            "-mg", "--mongo-collection", type=str, default=MONGO_VIDEOS_COLLECTION
        )
        self.add_argument(
            "-ml", "--milvus-collection", type=str, default=MILVUS_VIDEOS_COLLECTION
        )
        self.add_argument(
            "-f",
            "--filter-index",
            type=str,
            choices=["pubdate", "insert_at"],
            default="insert_at",
        )
        self.add_argument(
            "-r",
            "--runner-mode",
            type=str,
            choices=["local", "remote"],
            default="local",
        )
        self.add_argument("-s", "--start-date", type=str, default=None)
        self.add_argument("-e", "--end-date", type=str, default=None)
        self.add_argument("-d", "--dry-run", action="store_true", default=False)
        self.add_argument("-c", "--max-count", type=int, default=None)

    def parse_args(self):
        self.args, self.unknown_args = self.parse_known_args(sys.argv[1:])
        return self.args


def main(args: argparse.Namespace):
    indexer = MilvusVideoIndexer(args.milvus_collection, runner_mode=args.runner_mode)
    generator = MilvusVideoDocsGenerator(max_count=args.max_count)
    generator.init_mongo_cursor(
        collection=args.mongo_collection,
        filter_index=args.filter_index,
        filter_op="range",
        filter_range=[args.start_date, args.end_date or get_now_str()],
        estimate_count=False,
    )
    indexer.index_milvus_docs(generator, op_type="upsert", dry_run=args.dry_run)


if __name__ == "__main__":
    arg_parser = MilvusVideoIndexerArgParser()
    args = arg_parser.parse_args()
    main(args)

    # python -m workers.milvus_videos.indexer -f pubdate -r remote -s "2025-02-16" -e "2025-02-16 01:00:00" -c 5000
    # python -m cProfile -o milvus_videos_indexer.prof -m workers.milvus_videos.indexer -f pubdate -s "2025-02-10" -e "2025-02-12" -c 5000
    # snakeviz milvus_videos_indexer.prof -H 0.0.0.0 -p 10888
