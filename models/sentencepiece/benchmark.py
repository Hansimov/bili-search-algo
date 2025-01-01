import argparse
import sys
import timeit

from collections.abc import Generator
from pathlib import Path
from tclogger import logger, dict_to_str

from configs.envs import SP_MERGED_MODEL_PREFIX

# from models.sentencepiece.tokenizerC import SentenceFullTokenizer
from models.sentencepiece.tokenizer import SentenceFullTokenizer
from models.sentencepiece.tokenizer_parallel import ParallelSentenceFullTokenizer
from models.sentencepiece.test import TEST_SENTENCES


class SentenceTokenzierBenchmarker:
    def __init__(
        self, model_prefix: str = SP_MERGED_MODEL_PREFIX, drop_non_word: bool = True
    ):
        self.model_prefix = model_prefix
        self.drop_non_word = drop_non_word
        self.init_tokenizer_params()

    def init_tokenizer_params(self):
        logger.note("> Init tokenzier params ...")
        self.model_path = str(Path(f"{self.model_prefix}.model").resolve())
        self.tokenizer_params = {
            "model_path": self.model_path,
            "drop_non_word": self.drop_non_word,
            "verbose": True,
        }
        logger.mesg(dict_to_str(self.tokenizer_params), indent=2)

    def test(self):
        self.tokenizer = SentenceFullTokenizer(**self.tokenizer_params)
        logger.note("> Testing ...")
        for sentence in TEST_SENTENCES[:]:
            tokens = self.tokenizer.tokenize(sentence)
            pretty_tokens = self.tokenizer.stringify(tokens)
            logger.mesg(f"  * {pretty_tokens}")

    def log_epoch_stats(
        self, epoch_time: float, epochs: int, iterations: int, test_sentence_len: int
    ):
        iter_per_sec = iterations / epoch_time
        chars_per_sec = int(test_sentence_len * iter_per_sec)
        logger.file(
            f"  * {epoch_time:.2f}s: {iter_per_sec:.1f} iter/s; {chars_per_sec} char/s"
        )

    def log_total_stats(
        self, total_time: float, epochs: int, iterations: int, test_sentence_len: int
    ):
        avg_time = total_time / epochs
        avg_iter_per_sec = iterations / avg_time
        avt_chars_per_sec = int(test_sentence_len * avg_iter_per_sec)
        logger.success(
            f"{avg_time:.2f}s: {avg_iter_per_sec:.1f} iter/s; {avt_chars_per_sec} char/s"
        )

    def benchmark(self, epochs: int = 5, iterations: int = 100000):
        logger.note("> Benchmarking ...")
        self.tokenizer = SentenceFullTokenizer(**self.tokenizer_params)
        test_sentence = TEST_SENTENCES[-1]
        test_sentence_len = len(test_sentence)
        total_time = 0
        func = self.tokenizer.tokenize
        for i in range(epochs):
            epoch_time = timeit.timeit(lambda: func(test_sentence), number=iterations)
            self.log_epoch_stats(epoch_time, epochs, iterations, test_sentence_len)
            total_time += epoch_time
        self.log_total_stats(total_time, epochs, iterations, test_sentence_len)

    def benchmark_parallel(
        self,
        epochs: int = 5,
        iterations: int = 100000,
        workers_num=16,
        batch_size=100000,
    ):
        self.parallel_tokenizer = ParallelSentenceFullTokenizer(
            **self.tokenizer_params, workers_num=workers_num, batch_size=batch_size
        )
        logger.note("> Benchmarking parallel ...")
        test_sentence = TEST_SENTENCES[-1]
        test_sentence_len = len(test_sentence)

        def test_sentence_generator() -> Generator[str, None, None]:
            for i in range(iterations):
                yield test_sentence

        total_time = 0

        def iterate_test_sentence_generator():
            sentence_generator = test_sentence_generator()
            for results in self.parallel_tokenizer.tokenize_iter(sentence_generator):
                pass

        for i in range(epochs):
            epoch_time = timeit.timeit(
                lambda: iterate_test_sentence_generator(), number=1
            )
            self.log_epoch_stats(epoch_time, epochs, iterations, test_sentence_len)
            total_time += epoch_time
        self.parallel_tokenizer.terminate()
        self.log_total_stats(total_time, epochs, iterations, test_sentence_len)


class ArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("-t", "--test", action="store_true")
        self.add_argument("-p", "--parallel", action="store_true")

    def parse_args(self):
        self.args, self.unknown_args = self.parse_known_args(sys.argv[1:])
        return self.args


if __name__ == "__main__":
    args = ArgParser().parse_args()
    benchmarker = SentenceTokenzierBenchmarker(drop_non_word=True)

    if args.test:
        benchmarker.test()
    elif args.parallel:
        benchmarker.benchmark_parallel()
    else:
        benchmarker.benchmark()

    # python -m models.sentencepiece.benchmark
    # python -m models.sentencepiece.benchmark -t
    # python -m models.sentencepiece.benchmark -p

    # python -m cProfile -o sentencepiece_benchmark.prof -m models.sentencepiece.benchmark
    # snakeviz sentencepiece_benchmark.prof -H 0.0.0.0 -p 10888
