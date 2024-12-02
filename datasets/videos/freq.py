import argparse
import numpy as np
import pandas as pd
import sys


from pathlib import Path
from tclogger import logger, logstr, dict_to_str, brk
from typing import Union

from datasets.videos.data import SentencesDataloader, DataLoaderArgParser
from models.sentencepiece.tokenizer_parallel import ParallelSentenceFullTokenizer


class VideoTextsTokenFreqCounter:
    def __init__(
        self,
        data_loader: SentencesDataloader,
        tokenizer: ParallelSentenceFullTokenizer,
        max_count_batch: int = None,
    ):
        self.data_loader = data_loader
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
        res = {
            "percentiles": self.percentiles,
            "doc_freq": self.doc_freq_percentiles,
            "term_freq": self.term_freq_percentiles,
        }
        return res

    def dump(self, output_path: Union[str, Path], percentile_threshold: float = 0.85):
        percentiles = sorted(self.percentiles)
        for i in range(len(percentiles) - 1):
            if (
                percentiles[i] >= percentile_threshold
                and percentiles[i + 1] <= percentile_threshold
            ):
                percentile_threshold = percentiles[i]
                break
        logger.mesg(f"  * percentile_threshold: {brk(percentile_threshold)}")

        doc_freq_threshold = self.doc_freq_percentiles_dict[percentile_threshold]
        df = pd.DataFrame()
        freq_list = [
            {
                "token": token,
                "doc_freq": self.doc_freqs[token],
                "term_freq": self.term_freqs[token],
            }
            for token in self.term_freqs
            if self.doc_freqs[token] >= doc_freq_threshold
        ]
        df = pd.DataFrame(freq_list)
        df = df.sort_values(by=["doc_freq", "term_freq"], ascending=False)
        df.to_csv(output_path, index=False)


if __name__ == "__main__":
    data_loader_parser = DataLoaderArgParser(add_help=False)
    merged_parser = argparse.ArgumentParser(parents=[data_loader_parser])
    args, unknown_args = merged_parser.parse_known_args(sys.argv[1:])

    mongo_filter = {"tid": 17}
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
        **data_params, show_at_init=False, show_epoch_bar=False, verbose=True
    )
    tokenizer = ParallelSentenceFullTokenizer(
        Path("sp_400k_merged.model"),
        # drop_non_word=True, # This param is not needed as doc_coverter in data_loader already does this
        drop_whitespace=True,
        workers_num=16,
        batch_size=args.batch_size * 2,
    )

    counter = VideoTextsTokenFreqCounter(data_loader=data_loader, tokenizer=tokenizer)

    logger.note("> Counting tokens frequency ...")
    counter.count()

    logger.note("> Calculating percentiles ...")
    percentile_res = counter.calc_percentiles()
    logger.success(dict_to_str(percentile_res), indent=2)

    logger.note("> Dumping to csv ...")
    csv_path = Path("video_texts_freq.csv")
    counter.dump("video_texts_freq.csv")
    logger.file(f"  * {str(csv_path.resolve())}")

    # python -m datasets.videos.freq
