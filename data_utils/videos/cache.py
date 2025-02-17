import argparse
import sys

from tclogger import Runtimer, logger, logstr, dict_to_str, brk

from configs.envs import SP_MERGED_MODEL_PATH
from data_utils.args import DATA_LOADER_ARG_PARSER
from data_utils.videos.data import SentencesDataloader
from data_utils.videos.parquet import VideoTextsParquetWriter
from models.sentencepiece.filter import REGION_MONGO_FILTERS
from models.sentencepiece.filter import construct_mongo_filter_from_args
from models.sentencepiece.tokenizer_parallel import ParallelSentenceFullTokenizer


class VideoTextsTokenCacher:
    def __init__(
        self,
        data_loader: SentencesDataloader,
        tokenizer: ParallelSentenceFullTokenizer,
        parquet_writer: VideoTextsParquetWriter,
        max_count_batch: int = None,
    ):
        self.data_loader = data_loader
        self.tokenizer = tokenizer
        self.parquet_writer = parquet_writer
        self.max_count_batch = max_count_batch

    def write(self):
        self.parquet_writer.clear_dataset()
        self.data_loader.__epoch_start__()
        for batch_idx, doc_batch in enumerate(
            self.data_loader.doc_batch_generator(doc_type="doc")
        ):
            if self.max_count_batch and batch_idx >= self.max_count_batch:
                break
            sentence_batch = [
                self.data_loader.doc_converter.convert(doc) for doc in doc_batch
            ]
            tokenize_results = self.tokenizer.tokenize_list(sentence_batch)
            tokens_batch = [result["tokens"] for result in tokenize_results]
            rows_batch = [
                {
                    "bvid": doc["bvid"],
                    "ptid": doc["ptid"],
                    "tid": doc["tid"],
                    "sentence": sentence,
                    "tokens": tokens,
                }
                for doc, sentence, tokens in zip(
                    doc_batch, sentence_batch, tokens_batch
                )
            ]
            self.parquet_writer.append_buffer(rows_batch)
        self.parquet_writer.flush_buffer()
        self.tokenizer.terminate()


class TokenCacherArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("-o", "--output-prefix", type=str, default="video_texts_")
        self.add_argument("-mcb", "--max-count-batch", type=int, default=None)
        self.add_argument("-fd", "--force-delete", action="store_true")
        self.add_argument("-fl", "--for-loop", action="store_true")

    def parse_args(self):
        self.args, self.unknown_args = self.parse_known_args(sys.argv[1:])
        return self.args


def main(args: argparse.Namespace):
    logger.note("> Initiating data loader ...")
    mongo_filter = construct_mongo_filter_from_args(args)
    data_loader_params = {
        "dbname": args.dbname,
        "collect_name": args.collect_name,
        "data_fields": [
            *["title", "owner.name", "rtags", "tags"],
            *["owner.name", "title", "rtags", "tags"],
            "desc",
        ],
        "mongo_filter": mongo_filter,
        "max_batch": args.max_batch,
        "batch_size": args.batch_size,
        "estimate_count": args.estimate_count,
    }
    logger.mesg(dict_to_str(data_loader_params), indent=2)
    data_loader = SentencesDataloader(
        **data_loader_params,
        show_at_init=False,
        task_type="freq",
        show_epoch_bar=False,
        verbose=True,
    )

    tokenizer = ParallelSentenceFullTokenizer(
        SP_MERGED_MODEL_PATH,
        # drop_non_word=True, # This param is not needed as doc_coverter in data_loader already does this
        drop_whitespace=True,
        workers_num=16,
        batch_size=args.batch_size * 2,
    )

    logger.note("> Initiating parquet operator ...")
    parquet_writer_params = {
        "dataset_root": args.dataset_root,
        "dataset_name": args.dataset_name,
        "parquet_prefix": args.parquet_prefix,
        "col_types": {
            "bvid": str,
            "ptid": int,
            "tid": int,
            "sentence": str,
            "tokens": list[str],
        },
        "dataset_max_rows": int(args.dataset_max_w_rows * 1e4),
        "file_max_rows": int(args.file_max_w_rows * 1e4),
        "buffer_max_rows": int(args.buffer_max_w_rows * 1e4),
        "force_delete": args.force_delete,
        "verbose": True,
    }
    logger.mesg(dict_to_str(parquet_writer_params), indent=2)
    parquet_writer = VideoTextsParquetWriter(**parquet_writer_params)

    cacher = VideoTextsTokenCacher(
        data_loader=data_loader,
        tokenizer=tokenizer,
        parquet_writer=parquet_writer,
        max_count_batch=args.max_count_batch,
    )

    logger.note("> Caching ...")
    cacher.write()


if __name__ == "__main__":
    arg_parser = DATA_LOADER_ARG_PARSER
    arg_parser.add_parser_class(TokenCacherArgParser)
    args = arg_parser.parse_args()

    if not args.dataset_root:
        args.dataset_root = "parquets"

    if args.for_loop:
        regions = list(REGION_MONGO_FILTERS.keys())
        for idx, region in enumerate(regions):
            args.filter_group = region
            args.dataset_name = f"{args.output_prefix}{region}"
            region_str = logstr.file(brk(region))
            idx_str = f"{logstr.mesg(str(idx+1))}/{logstr.file(str(len(regions)))}"
            with Runtimer():
                logger.note(f"> [{idx_str}] Caching region: {region_str}")
                main(args)
                logger.success(f"✓ [{idx_str}] Cached region: {region_str}")
    elif args.filter_group:
        region = args.filter_group
        args.dataset_name = f"{args.output_prefix}{region}"
        with Runtimer():
            logger.note(f"> Caching region: {region}")
            main(args)
            logger.success(f"✓ Cached region: {region}")
    elif args.dataset_name:
        with Runtimer():
            main(args)
    else:
        raise ValueError("× Must provide output dataset_name")

    # python -m data_utils.videos.cache -ec -mcb 22 -bw 5 -fw 10
    # python -m data_utils.videos.cache -dn video_texts_tid_17
    # python -m data_utils.videos.cache -dn video_texts_tid_201

    # python -m data_utils.videos.cache -dn video_texts_tid_all -ec -fw 200 -bw 100 -bs 10000

    # WARNING: Run multiple region-tasks could make mongodb out-of-memory

    # python -m data_utils.videos.cache -fd -fl
    # python -m data_utils.videos.cache -fd -fg douga_anime
