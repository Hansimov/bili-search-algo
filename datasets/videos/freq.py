import argparse
import json
import numpy as np
import pandas as pd
import pickle
import sys

from collections import defaultdict
from pathlib import Path
from tclogger import logger, logstr, dict_to_str, brk
from typing import Union, Literal

from datasets.videos.data import SentencesDataloader
from datasets.videos.data import ParquetRowsDataLoader
from datasets.videos.parquet import VideoTextsParquetReader
from datasets.args import DATA_LOADER_ARG_PARSER
from models.sentencepiece.tokenizer_parallel import ParallelSentenceFullTokenizer


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
        recorded_doc_tokens = set()
        for token in tokens:
            self.term_freqs[token] += 1
            if token not in recorded_doc_tokens:
                self.doc_freqs[token] += 1
                recorded_doc_tokens.add(token)

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
        freq_pickle_path = Path(output_prefix).with_suffix(".pickle")
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
        freq_csv_path = Path(output_prefix).with_suffix(".csv")
        df.to_csv(freq_csv_path, index=False)
        logger.file(f"  * {str(freq_csv_path.resolve())}")
        logger.mesg(f"  * vocab_size: {logstr.file(brk(len(df)))}")

        # dump percentiles info to .jsonv
        percentile_info_path = Path(output_prefix).with_suffix(".jsonv")
        with open(percentile_info_path, "w") as wf:
            json.dump(self.percentile_info_list, wf, indent=4)
        logger.file(f"  * {str(percentile_info_path.resolve())}")


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
        self.add_argument(
            "-tid", "--tid", type=int, default=None, help="tid filter for mongo"
        )

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
            Path("sp_400k_merged.model"),
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
            "max_tables": args.max_tables,
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

    # python -m datasets.videos.freq
    # python -m datasets.videos.freq -o video_texts_freq_all -ec -mcb 1
    # python -m datasets.videos.freq -o video_texts_freq_all -dn "video_texts_tid_all"
    # python -m datasets.videos.freq -o video_texts_freq_tid_17 -dn "video_texts_tid_17" -tid 17
    # python -m datasets.videos.freq -o video_texts_freq_tid_17_nt -dn "video_texts_tid_17" -tid 17 -nt
