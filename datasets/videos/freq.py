import argparse
import sys

from pathlib import Path
from tclogger import logger, logstr, dict_to_str, brk

from datasets.videos.data import SentencesDataloader, DataLoaderArgParser
from models.sentencepiece.tokenizer_parallel import ParallelSentenceFullTokenizer


class VideoTextsTokenFrequencyCounter:
    def __init__(
        self, data_loader: SentencesDataloader, tokenizer: ParallelSentenceFullTokenizer
    ):
        self.data_loader = data_loader
        self.tokenizer = tokenizer

    def count(self):
        self.data_loader.__epoch_start__()
        for batch_idx, sentence_batch in enumerate(
            self.data_loader.doc_batch_generator(doc_type="sentence")
        ):
            tokenize_results: list[dict[str]] = self.tokenizer.tokenize_list(
                sentence_batch
            )
            logger.success(dict_to_str(tokenize_results[-2]), indent=2)
            break
        self.tokenizer.terminate()


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
    counter = VideoTextsTokenFrequencyCounter(
        data_loader=data_loader, tokenizer=tokenizer
    )
    logger.note("> Counting tokens frequency ...")
    counter.count()

    # python -m datasets.videos.freq
