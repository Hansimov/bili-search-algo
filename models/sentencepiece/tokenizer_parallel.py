import multiprocessing as mp

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
        verbose: bool = False,
    ):
        self.tokenizer = SentenceFullTokenizer(
            model_path=model_path, drop_non_word=drop_non_word, verbose=verbose
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
        verbose: bool = False,
        workers_num: int = None,
        batch_size: int = 1000,
    ):
        self.model_path = str(model_path)
        self.drop_non_word = drop_non_word
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
        self, results: list[tuple[int, dict[str, str]]]
    ) -> list[dict[str, str]]:
        sorted_results = sorted(results, key=lambda x: x[0])
        return [result for _, result in sorted_results]

    def tokenize_list(self, sentences: list[str]) -> list[dict[str, str]]:
        for idx, sentence in enumerate(sentences):
            self.input_queue.put((idx, sentence))

        results = []
        for _ in range(len(sentences)):
            task_id, result = self.output_queue.get()
            results.append((task_id, result))

        return self.format_results(results)

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


if __name__ == "__main__":
    from models.sentencepiece.test import TEST_SENTENCES

    model_prefix = "sp_400k_merged"
    model_path = str(Path(__file__).parents[2] / f"{model_prefix}.model")
    tokenizer = ParallelSentenceFullTokenizer(
        model_path, drop_non_word=True, verbose=False, workers_num=8
    )

    logger.note(
        f"> Parallel Tokenizing with workers: {logstr.mesg(brk(tokenizer.workers_num))}"
    )
    sentences = TEST_SENTENCES
    results = tokenizer.tokenize(sentences)

    for sentence, result in zip(sentences, results):
        logger.mesg(f"  * {result['string']}")

    tokenizer.terminate()

    # python -m models.sentencepiece.tokenizer_parallel
