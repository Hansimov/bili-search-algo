import argparse
import sys
import timeit

from pathlib import Path
from tclogger import logger

# from models.sentencepiece.tokenizer import SentenceFullTokenizer
from models.sentencepiece.tokenizerC import SentenceFullTokenizer
from models.sentencepiece.test import TEST_SENTENCES


class SentenceTokenzierBenchmarker:
    def __init__(
        self, model_prefix: str = "sp_400k_merged", drop_non_word: bool = True
    ):
        self.model_prefix = model_prefix
        self.drop_non_word = drop_non_word
        self.load_model()

    def load_model(self):
        logger.note("> Loading tokenzier ...")
        self.model_path = str(Path(f"{self.model_prefix}.model").resolve())
        logger.file(f"  * {self.model_path}")
        self.tokenizer = SentenceFullTokenizer(
            self.model_path, drop_non_word=self.drop_non_word, verbose=True
        )

    def test(self):
        logger.note("> Testing ...")
        for sentence in TEST_SENTENCES[:]:
            tokens = self.tokenizer.tokenize(sentence)
            pretty_tokens = self.tokenizer.stringify(tokens)
            logger.mesg(f"  * {pretty_tokens}")

    def benchmark(self):
        logger.note("> Benchmarking ...")
        sum_time = 0
        epochs, iterations = 3, 10000
        test_sentence = TEST_SENTENCES[-1]
        test_sentence_len = len(test_sentence)
        for i in range(epochs):
            res = timeit.timeit(
                lambda: self.tokenizer.tokenize(test_sentence), number=iterations
            )
            iter_per_sec = iterations / res
            chars_per_sec = int(test_sentence_len * iter_per_sec)
            logger.file(
                f"  * {res:.2f}s: {iter_per_sec:.1f} iter/s; {chars_per_sec} char/s"
            )
            sum_time += res

        avg_time = sum_time / epochs
        avg_iter_per_sec = iterations / avg_time
        avt_chars_per_sec = int(test_sentence_len * avg_iter_per_sec)
        logger.success(
            f"{avg_time:.2f}s: {avg_iter_per_sec:.1f} iter/s; {avt_chars_per_sec} char/s"
        )


class ArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("-t", "--test", action="store_true")

    def parse_args(self):
        self.args, self.unknown_args = self.parse_known_args(sys.argv[1:])
        return self.args


if __name__ == "__main__":
    args = ArgParser().parse_args()
    benchmarker = SentenceTokenzierBenchmarker(drop_non_word=True)

    if args.test:
        benchmarker.test()
    else:
        benchmarker.benchmark()

    # python -m models.sentencepiece.benchmark
    # python -m models.sentencepiece.benchmark -t

    # python -m cProfile -o sentencepiece_benchmark.prof -m models.sentencepiece.benchmark
    # snakeviz sentencepiece_benchmark.prof -H 0.0.0.0 -p 10888
