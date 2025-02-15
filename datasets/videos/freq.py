import argparse
import json
import multiprocessing as mp
import numpy as np
import pandas as pd
import pickle
import sys

from collections import defaultdict
from tclogger import logger, logstr, dict_to_str, brk
from typing import Union, Literal

from configs.envs import SP_MERGED_MODEL_PATH, TOKEN_FREQS_ROOT
from datasets.videos.data import SentencesDataloader
from datasets.videos.data import ParquetRowsDataLoader
from datasets.videos.parquet import VideoTextsParquetReader
from datasets.args import DATA_LOADER_ARG_PARSER
from models.sentencepiece.tokenizer_parallel import ParallelSentenceFullTokenizer


def read_token_freq_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(
        path,
        na_filter=False,
        dtype={"token": str, "doc_freq": int, "term_freq": int, "pos": str},
    )


class VideoTextsTokenFreqCounter:
    def __init__(
        self,
        data_loader: Union[ParquetRowsDataLoader, SentencesDataloader],
        data_source: Literal["mongo", "parquet"] = "parquet",
        tokenizer: ParallelSentenceFullTokenizer = None,
        max_count_batch: int = None,
    ):
        self.data_loader = data_loader
        self.data_source = data_source
        self.tokenizer = tokenizer
        self.max_count_batch = max_count_batch
        self.term_freqs = defaultdict(int)
        self.doc_freqs = defaultdict(int)

    def count_tokens_freq(self, tokens: list[str]):
        for token in tokens:
            self.term_freqs[token] += 1
        for token in set(tokens):
            self.doc_freqs[token] += 1

    def sort_freqs(self):
        self.term_freqs = dict(
            sorted(self.term_freqs.items(), key=lambda x: x[1], reverse=True)
        )
        self.doc_freqs = dict(
            sorted(self.doc_freqs.items(), key=lambda x: x[1], reverse=True)
        )

    def count(self):
        self.data_loader.__epoch_start__()
        if self.data_source == "mongo":
            for batch_idx, sentence_batch in enumerate(
                self.data_loader.doc_batch_generator(doc_type="sentence")
            ):
                if self.max_count_batch and batch_idx >= self.max_count_batch:
                    break
                tokenize_results = self.tokenizer.tokenize_list(sentence_batch)
                for result in tokenize_results:
                    self.count_tokens_freq(result["tokens"])
                batch_bar_desc = logstr.mesg(f"tokens: {brk(len(self.term_freqs))}")
                self.data_loader.batch_bar.update(desc=batch_bar_desc)
            print()
            logger.hint(dict_to_str(result), indent=2)
            self.tokenizer.terminate()
        elif self.data_source == "parquet":
            for batch_idx, tokens_batch in enumerate(self.data_loader.batch_generator):
                if self.max_count_batch and batch_idx >= self.max_count_batch:
                    break
                self.data_loader.sample_bar.total = len(tokens_batch)
                for tokens in tokens_batch:
                    self.count_tokens_freq(tokens)
                    self.data_loader.sample_bar.update(increment=1)
                self.data_loader.sample_bar.reset()
                batch_bar_desc = logstr.mesg(f"tokens: {brk(len(self.term_freqs))}")
                self.data_loader.batch_bar.update(increment=1, desc=batch_bar_desc)
            print()
            logger.hint(tokens, indent=2)
        else:
            raise ValueError(f"Unknown data source: {self.data_source}")

        self.sort_freqs()

    def calc_percentiles(self) -> dict:
        percentiles = [
            *[0, 0.2, 0.4, 0.5, 0.6, 0.75],
            *[0.8, 0.85, 0.9, 0.95, 0.99, 0.999, 1],
        ]
        percentiles_np = np.array(percentiles) * 100

        def freq_dict_to_percentiles(d: dict) -> tuple[list[int], dict[float, int]]:
            percentile_list = (
                np.percentile(list(d.values()), percentiles_np).astype(int).tolist()
            )
            percentile_dict = {p: f for p, f in zip(percentiles, percentile_list)}
            return percentile_list, percentile_dict

        self.term_freq_percentiles, self.term_freq_percentiles_dict = (
            freq_dict_to_percentiles(self.term_freqs)
        )
        self.doc_freq_percentiles, self.doc_freq_percentiles_dict = (
            freq_dict_to_percentiles(self.doc_freqs)
        )
        self.percentiles = percentiles
        self.percentile_info = {
            "percentiles": self.percentiles,
            "doc_freq": self.doc_freq_percentiles,
            "term_freq": self.term_freq_percentiles,
        }
        self.percentile_info_list = [
            {"percentile": percentile, "doc_freq": doc_freq, "term_freq": term_freq}
            for percentile, doc_freq, term_freq in zip(
                self.percentiles, self.doc_freq_percentiles, self.term_freq_percentiles
            )
        ]
        return self.percentile_info

    def dump(
        self,
        output_prefix: str,
        percent_threshold: float = 0.85,
        count_threshold: int = 20,
        no_threshold: bool = False,
    ):
        percentiles = sorted(self.percentiles)
        for i in range(len(percentiles) - 1):
            if (
                percentiles[i] >= percent_threshold
                and percentiles[i + 1] <= percent_threshold
            ):
                percent_threshold = percentiles[i]
                break

        doc_freq_threshold = self.doc_freq_percentiles_dict[percent_threshold]
        max_threshold = max(count_threshold, doc_freq_threshold)

        logger.mesg(f"  * percent_threshold  : {logstr.file(brk(percent_threshold))}")
        logger.mesg(f"  * doc_freq_threshold : {logstr.file(brk(doc_freq_threshold))}")
        logger.mesg(f"  * count_threshold    : {logstr.file(brk(count_threshold))}")
        logger.mesg(f"  * no_threshold       : {logstr.file(brk(no_threshold))}")

        # dump freqs info to .pickle
        freq_pickle_path = TOKEN_FREQS_ROOT / f"{output_prefix}.pickle"
        freq_info = {
            "doc_freqs": self.doc_freqs,
            "term_freqs": self.term_freqs,
        }
        with open(freq_pickle_path, "wb") as wf:
            pickle.dump(freq_info, wf)
        logger.file(f"  * {str(freq_pickle_path.resolve())}")

        # dump freqs info to .csv
        df = pd.DataFrame()
        freq_list = [
            {
                "token": token,
                "doc_freq": self.doc_freqs[token],
                "term_freq": self.term_freqs[token],
            }
            for token in self.term_freqs
            if (no_threshold) or (self.doc_freqs[token] >= max_threshold)
        ]
        df = pd.DataFrame(freq_list)
        df = df.sort_values(by=["doc_freq", "term_freq"], ascending=False)
        freq_csv_path = TOKEN_FREQS_ROOT / f"{output_prefix}.csv"
        df.to_csv(freq_csv_path, index=False)
        logger.file(f"  * {str(freq_csv_path.resolve())}")
        logger.mesg(f"  * vocab_size: {logstr.file(brk(len(df)))}")

        # dump percentiles info to .jsonv
        percentile_info_path = TOKEN_FREQS_ROOT / f"{output_prefix}.jsonv"
        with open(percentile_info_path, "w") as wf:
            json.dump(self.percentile_info_list, wf, indent=4)
        logger.file(f"  * {str(percentile_info_path.resolve())}")


def freq_worker_process(input_queue: mp.Queue, output_queue: mp.Queue):
    while True:
        task = input_queue.get()
        if task is None:
            break

        batch_idx, tokens_batch = task
        term_freqs = defaultdict(int)
        doc_freqs = defaultdict(int)

        for tokens in tokens_batch:
            for token in tokens:
                term_freqs[token] += 1
                doc_freqs[token] = 1

        output_queue.put((batch_idx, (term_freqs, doc_freqs)))


class ParallelVideoTextsTokenFreqCounter(VideoTextsTokenFreqCounter):
    """
    Experiments indicate that this parallel implementation
    is slower than the original one,
    as the bottleneck is dict update,
    and the merge part in parallel version still cannot avoid it.
    """

    def __init__(
        self,
        data_loader: Union[ParquetRowsDataLoader, SentencesDataloader],
        data_source: Literal["mongo", "parquet"] = "parquet",
        tokenizer: ParallelSentenceFullTokenizer = None,
        max_count_batch: int = None,
    ):
        self.data_loader = data_loader
        self.data_source = data_source
        self.tokenizer = tokenizer
        self.max_count_batch = max_count_batch
        self.term_freqs = defaultdict(int)
        self.doc_freqs = defaultdict(int)

    def create_workers(self):
        self.workers_num = mp.cpu_count() // 2
        self.workers = []
        self.input_queue = mp.Queue()
        self.output_queue = mp.Queue()
        for _ in range(self.workers_num):
            p = mp.Process(
                target=freq_worker_process,
                args=(self.input_queue, self.output_queue),
            )
            p.start()
            self.workers.append(p)

    def terminate_workers(self):
        for _ in self.workers:
            self.input_queue.put(None)
        for p in self.workers:
            p.join()
        self.workers = []

    def submit_queue(self):
        batch_results = []
        for _ in range(self.tasks_count):
            task_id, result = self.output_queue.get()
            batch_results.append(result)
            self.tasks_count -= 1
        self.merge_batch_count_results(batch_results)
        batch_bar_desc = logstr.mesg(f"tokens: {brk(len(self.term_freqs))}")
        self.data_loader.batch_bar.update(
            increment=len(batch_results), desc=batch_bar_desc
        )

    def merge_batch_count_results(self, batch_results):
        for term_freq, doc_freq in batch_results:
            for token, count in term_freq.items():
                self.term_freqs[token] += count
            for token, count in doc_freq.items():
                self.doc_freqs[token] = 1

    def count(self):
        self.data_loader.__epoch_start__()
        if self.data_source == "parquet":
            self.create_workers()
            self.tasks_count = 0
            for batch_idx, tokens_batch in enumerate(self.data_loader.batch_generator):
                if self.max_count_batch and batch_idx >= self.max_count_batch:
                    break
                self.input_queue.put((batch_idx, tokens_batch))
                self.tasks_count += 1
                if self.tasks_count >= 5:
                    self.submit_queue()
            if self.tasks_count > 0:
                self.submit_queue()
            print()
        else:
            raise ValueError(f"Unknown data source: {self.data_source}")

        self.sort_freqs()


class FreqCounterArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument(
            "-ds",
            "--data-source",
            type=str,
            choices=["mongo", "parquet"],
            default="parquet",
        )
        self.add_argument("-o", "--output-prefix", type=str, default="video_texts_freq")
        self.add_argument("-mcb", "--max-count-batch", type=int, default=None)
        self.add_argument("-nt", "--no-threshold", action="store_true")

    def parse_args(self):
        self.args, self.unknown_args = self.parse_known_args(sys.argv[1:])
        return self.args


if __name__ == "__main__":
    arg_parser = DATA_LOADER_ARG_PARSER
    arg_parser.add_parser_class(FreqCounterArgParser)
    args = arg_parser.parse_args()

    logger.note("> Initiating data loader ...")
    if args.data_source == "mongo":
        if args.tid:
            mongo_filter = {"tid": args.tid}
        else:
            mongo_filter = {}
        data_params = {
            "dbname": args.dbname,
            "collect_name": args.collect_name,
            "data_fields": args.data_fields.split(",") if args.data_fields else None,
            "mongo_filter": mongo_filter,
            "max_batch": args.max_batch,
            "batch_size": args.batch_size,
            "estimate_count": args.estimate_count,
        }
        logger.mesg(dict_to_str(data_params), indent=2)
        data_loader = SentencesDataloader(
            **data_params,
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
    elif args.data_source == "parquet":
        parquet_params = {
            "dataset_root": args.dataset_root,
            "dataset_name": args.dataset_name,
            "parquet_prefix": args.parquet_prefix,
        }
        parquet_reader = VideoTextsParquetReader(**parquet_params)
        logger.mesg(dict_to_str(parquet_params), indent=2)
        data_params = {
            "column": "tokens",
            "max_batch": args.max_batch,
            "batch_size": args.batch_size,
            "max_rows": args.max_rows,
            "show_at_init": False,
            "show_epoch_bar": False,
            "verbose": True,
        }
        data_loader = ParquetRowsDataLoader(
            **data_params, parquet_reader=parquet_reader
        )
        logger.mesg(dict_to_str(data_params), indent=2)
        tokenizer = None
    else:
        raise ValueError(f"Unknown data source: {args.data_source}")

    counter = VideoTextsTokenFreqCounter(
        data_loader=data_loader,
        tokenizer=tokenizer,
        max_count_batch=args.max_count_batch,
    )

    logger.note("> Counting tokens frequency ...")
    counter.count()

    logger.note("> Calculating percentiles ...")
    percentile_res = counter.calc_percentiles()
    logger.success(dict_to_str(percentile_res), indent=2)

    logger.note("> Dumping ...")
    counter.dump(args.output_prefix, no_threshold=args.no_threshold)

    # python -m datasets.videos.freq -dr "parquets" -o video_texts_freq_all
    # python -m datasets.videos.freq -dr "parquets" -dn "video_texts_other_game" -o video_texts_other_game_nt -nt
