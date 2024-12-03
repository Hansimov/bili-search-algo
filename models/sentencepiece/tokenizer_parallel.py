import argparse
import multiprocessing as mp
import sys

from collections.abc import Iterable, Generator
from pathlib import Path
from tclogger import logger, logstr, brk
from typing import Union

from models.sentencepiece.tokenizer import SentenceFullTokenizer


class TokenizerWorker:
    def __init__(
        self,
        model_path: Union[Path, str],
        drop_non_word: bool = False,
        drop_whitespace: bool = False,
        verbose: bool = False,
    ):
        self.tokenizer = SentenceFullTokenizer(
            model_path=model_path,
            drop_non_word=drop_non_word,
            drop_whitespace=drop_whitespace,
            verbose=verbose,
        )

    def tokenize(self, sentence: str) -> list[str]:
        return self.tokenizer.tokenize(sentence)

    def stringify(self, tokens: list[str]) -> str:
        return self.tokenizer.stringify(tokens)


def worker_process(worker_params: dict, input_queue: mp.Queue, output_queue: mp.Queue):
    worker = TokenizerWorker(**worker_params)
    while True:
        task = input_queue.get()
        if task is None:
            break
        task_id, sentence = task
        tokens = worker.tokenize(sentence)
        tokens_str = worker.stringify(tokens)
        result = {
            "sentence": sentence,
            "tokens": tokens,
            "string": tokens_str,
        }
        output_queue.put((task_id, result))


class ParallelSentenceFullTokenizer:
    def __init__(
        self,
        model_path: Union[Path, str],
        drop_non_word: bool = False,
        drop_whitespace: bool = False,
        verbose: bool = False,
        workers_num: int = None,
        batch_size: int = 1000,
    ):
        self.model_path = str(model_path)
        self.drop_non_word = drop_non_word
        self.drop_whitespace = drop_whitespace
        self.verbose = verbose
        self.workers_num = workers_num or mp.cpu_count() // 2
        self.batch_size = batch_size
        self.create_workers()

    def create_workers(self):
        self.input_queue = mp.Queue()
        self.output_queue = mp.Queue()
        self.tasks_count = 0
        self.workers = []
        self.worker_params = {
            "model_path": self.model_path,
            "drop_non_word": self.drop_non_word,
            "drop_whitespace": self.drop_whitespace,
            "verbose": self.verbose,
        }
        for _ in range(self.workers_num):
            p = mp.Process(
                target=worker_process,
                args=(self.worker_params, self.input_queue, self.output_queue),
            )
            p.start()
            self.workers.append(p)

    def format_results(
        self, results: list[tuple[int, dict[str, str]]], sort: bool = True
    ) -> list[dict[str, str]]:
        if sort:
            return [result for _, result in sorted(results, key=lambda x: x[0])]
        else:
            return [result for _, result in results]

    def tokenize_list(
        self, sentences: list[str], sort: bool = True
    ) -> list[dict[str, str]]:
        for idx, sentence in enumerate(sentences):
            self.input_queue.put((idx, sentence))

        results = []
        for _ in range(len(sentences)):
            task_id, result = self.output_queue.get()
            results.append((task_id, result))

        return self.format_results(results, sort)

    def tokenize_iter(
        self,
        sentences_iter: Union[list[str], Iterable[str]],
        quick_return: bool = False,
    ) -> Generator[list[dict[str, str]], None, None]:
        self.tasks_count = 0
        for idx, sentence in enumerate(sentences_iter):
            self.input_queue.put((idx, sentence))
            self.tasks_count += 1
            if quick_return or self.tasks_count >= self.batch_size:
                yield self.submit_queue()
        if self.tasks_count > 0:
            yield self.submit_queue()

    def submit_queue(self) -> list[dict[str, str]]:
        results = []
        for _ in range(self.tasks_count):
            task_id, result = self.output_queue.get()
            results.append((task_id, result))
            self.tasks_count -= 1
        return self.format_results(results)

    def terminate(self):
        for _ in self.workers:
            self.input_queue.put(None)
        for p in self.workers:
            p.join()


def test_parallel(tokenizer: ParallelSentenceFullTokenizer):
    logger.note(f"> tokenize_list: {logstr.mesg(brk(tokenizer.workers_num))}")
    sentences = TEST_SENTENCES
    results = tokenizer.tokenize_list(sentences)
    for result in results:
        logger.mesg(f"  * {result['string']}")
    tokenizer.terminate()


def test_parallel_iter(tokenizer: ParallelSentenceFullTokenizer):
    logger.note(f"> tokenize_iter: {logstr.mesg(brk(tokenizer.workers_num))}")
    sentences = TEST_SENTENCES
    for results in tokenizer.tokenize_iter(sentences):
        for result in results:
            logger.mesg(f"  * {result['string']}")
    tokenizer.terminate()


class ArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("-i", "--iter", action="store_true")

    def parse_args(self):
        self.args, self.unknown_args = self.parse_known_args(sys.argv[1:])
        return self.args


if __name__ == "__main__":
    from models.sentencepiece.test import TEST_SENTENCES

    args = ArgParser().parse_args()
    model_prefix = "sp_400k_merged"
    model_path = str(Path(__file__).parents[2] / f"{model_prefix}.model")
    tokenizer = ParallelSentenceFullTokenizer(
        model_path, drop_non_word=True, verbose=False, workers_num=8
    )

    if args.iter:
        test_parallel_iter(tokenizer)
    else:
        test_parallel(tokenizer)

    # python -m models.sentencepiece.tokenizer_parallel
    # python -m models.sentencepiece.tokenizer_parallel -i
