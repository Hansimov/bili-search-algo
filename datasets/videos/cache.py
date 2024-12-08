import argparse
import sys


from pathlib import Path
from tclogger import logger, logstr, dict_to_str, brk

from datasets.videos.data import CommonDataLoaderArgParser
from datasets.videos.data import SentencesDataloader, SentencesDataLoaderArgParser
from datasets.videos.parquet import VideoTextsParquetWriter
from datasets.videos.parquet import ParquetOperatorArgParser, ParquetWriterArgParser
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
        self.add_argument(
            "-o", "--output-prefix", type=str, default="video_texts_tokens"
        )
        self.add_argument("-td", "--tid", type=int, default=None)
        self.add_argument("-pd", "--ptid", type=int, default=None)
        self.add_argument("-mcb", "--max-count-batch", type=int, default=None)

    def parse_args(self):
        self.args, self.unknown_args = self.parse_known_args(sys.argv[1:])
        return self.args


if __name__ == "__main__":
    common_data_loader_parser = CommonDataLoaderArgParser(add_help=False)
    sentences_data_loader_parser = SentencesDataLoaderArgParser(add_help=False)
    parquet_operator_parser = ParquetOperatorArgParser(add_help=False)
    parquet_writer_parser = ParquetWriterArgParser(add_help=False)
    token_cacher_parser = TokenCacherArgParser(add_help=False)

    merged_parser = argparse.ArgumentParser(
        parents=[
            common_data_loader_parser,
            sentences_data_loader_parser,
            parquet_operator_parser,
            parquet_writer_parser,
            token_cacher_parser,
        ]
    )
    args, unknown_args = merged_parser.parse_known_args(sys.argv[1:])

    if args.tid:
        mongo_filter = {"tid": args.tid}
    elif args.ptid:
        mongo_filter = {"ptid": args.ptid}
    else:
        mongo_filter = {}

    logger.note("> Initiating data loader ...")
    data_loader_params = {
        "dbname": args.dbname,
        "collect_name": args.collect_name,
        "data_fields": args.data_fields.split(",") if args.data_fields else None,
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
        Path("sp_400k_merged.model"),
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
    }
    logger.mesg(dict_to_str(parquet_writer_params), indent=2)
    parquet_writer = VideoTextsParquetWriter(**parquet_writer_params, verbose=True)

    cacher = VideoTextsTokenCacher(
        data_loader=data_loader,
        tokenizer=tokenizer,
        parquet_writer=parquet_writer,
        max_count_batch=args.max_count_batch,
    )

    logger.note("> Caching ...")
    cacher.write()

    # python -m datasets.videos.cache -ec -mcb 22 -bw 5 -fw 10
    # python -m datasets.videos.cache -dn video_texts_tid_17
    # python -m datasets.videos.cache -dn video_texts_tid_201
    # python -m datasets.videos.cache -ec -dn video_texts_tid_all -fw 200 -bw 100 -bs 10000
