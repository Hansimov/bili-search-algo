import argparse
import json
import numpy as np
import pandas as pd
import sys


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
        self.term_freqs: dict[str, int] = {}
        self.doc_freqs: dict[str, int] = {}

    def count_tokens_freq(self, tokens: list[str]):
        recorded_doc_tokens = set()
        for token in tokens:
            self.term_freqs[token] = self.term_freqs.get(token, 0) + 1
            if token not in recorded_doc_tokens:
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1
                recorded_doc_tokens.add(token)

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
        output_path: Union[str, Path],
        percent_threshold: float = 0.85,
        count_threshold: int = 20,
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
        min_threshold = min(count_threshold, doc_freq_threshold)

        logger.mesg(f"  * percent_threshold  : {logstr.file(brk(percent_threshold))}")
        logger.mesg(f"  * doc_freq_threshold : {logstr.file(brk(doc_freq_threshold))}")
        logger.mesg(f"  * count_threshold    : {logstr.file(brk(count_threshold))}")

        df = pd.DataFrame()
        freq_list = [
            {
                "token": token,
                "doc_freq": self.doc_freqs[token],
                "term_freq": self.term_freqs[token],
            }
            for token in self.term_freqs
            if self.doc_freqs[token] >= min_threshold
        ]
        df = pd.DataFrame(freq_list)
        df = df.sort_values(by=["doc_freq", "term_freq"], ascending=False)
        df.to_csv(output_path, index=False)
        logger.file(f"  * {str(output_path.resolve())}")

        percentile_info_path = Path(output_path).with_suffix(".jsonv")
        with open(percentile_info_path, "w") as wf:
            json.dump(self.percentile_info_list, wf, indent=4)
        logger.file(f"  * {str(percentile_info_path.resolve())}")


class FreqCounterArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("-o", "--output-prefix", type=str, default="video_texts_freq")
        self.add_argument("-mcb", "--max-count-batch", type=int, default=None)
        self.add_argument(
            "-ds",
            "--data-source",
            type=str,
            choices=["mongo", "parquet"],
            default="parquet",
        )

    def parse_args(self):
        self.args, self.unknown_args = self.parse_known_args(sys.argv[1:])
        return self.args


if __name__ == "__main__":
    data_loader_parser = DataLoaderArgParser(add_help=False)
    freq_counter_parser = FreqCounterArgParser(add_help=False)
    merged_parser = argparse.ArgumentParser(
        parents=[data_loader_parser, freq_counter_parser]
    )
    args, unknown_args = merged_parser.parse_known_args(sys.argv[1:])

    # mongo_filter = {"tid": 231}
    mongo_filter = {}

    logger.note("> Initiating data loader ...")
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

    logger.note("> Dumping to csv ...")
    csv_path = Path(f"{args.output_prefix}.csv")
    counter.dump(csv_path)
    logger.file(f"  * {str(csv_path.resolve())}")

    # python -m datasets.videos.freq
    # python -m datasets.videos.freq -o video_texts_freq_all -ec
    # python -m datasets.videos.freq -o video_texts_freq_all -ec -mcb 1
