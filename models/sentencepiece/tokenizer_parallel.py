import multiprocessing as mp

from tclogger import logger, logstr, brk
from typing import List, Union
from pathlib import Path

from models.sentencepiece.tokenizer import SentenceFullTokenizer


class Worker:
    def __init__(
        self,
        model_path: Union[Path, str],
        drop_non_word: bool = False,
        verbose: bool = False,
    ):
        self.tokenizer = SentenceFullTokenizer(
            model_path=model_path, drop_non_word=drop_non_word, verbose=verbose
        )

    def tokenize(self, sentence: str) -> List[str]:
        return self.tokenizer.tokenize(sentence)

    def stringify(self, tokens: List[str]) -> str:
        return self.tokenizer.stringify(tokens)


def worker_process(
    model_path: str,
    input_queue: mp.Queue,
    output_queue: mp.Queue,
    verbose: bool = False,
    drop_non_word: bool = False,
):
    worker = Worker(model_path, drop_non_word=drop_non_word, verbose=verbose)
    while True:
        task = input_queue.get()
        if task is None:  # Sentinel to terminate the worker
            break
        task_id, sentence = task
        tokens = worker.tokenize(sentence)
        tokens_str = worker.stringify(tokens)
        result = {}
        result["tokens"] = tokens
        result["str"] = tokens_str

        output_queue.put((task_id, result))


class ParallelSentenceFullTokenizer:
    def __init__(
        self,
        model_path: Union[Path, str],
        drop_non_word: bool = False,
        verbose: bool = False,
        workers_num: int = None,
    ):
        self.model_path = str(model_path)
        self.drop_non_word = drop_non_word
        self.verbose = verbose
        self.workers_num = workers_num or mp.cpu_count() // 2
        self.input_queue = mp.Queue()
        self.output_queue = mp.Queue()
        self.workers = []

        for _ in range(self.workers_num):
            p = mp.Process(
                target=worker_process,
                args=(
                    self.model_path,
                    self.input_queue,
                    self.output_queue,
                    self.drop_non_word,
                    self.verbose,
                ),
            )
            p.start()
            self.workers.append(p)

    def tokenize(self, sentences: List[str]) -> List[List[str]]:
        """Tokenize a list of sentences using multiple workers."""
        # Submit tasks
        task_ids = {}
        for idx, sentence in enumerate(sentences):
            self.input_queue.put((idx, sentence))
            task_ids[idx] = sentence

        # Collect results
        results = {}
        for _ in range(len(sentences)):
            task_id, result = self.output_queue.get()
            results[task_id] = result

        # Sort results by original order
        return [results[idx] for idx in sorted(results.keys())]

    def close(self):
        """Terminate worker processes."""
        for _ in self.workers:
            self.input_queue.put(None)  # Send sentinel to terminate workers
        for p in self.workers:
            p.join()


# Example usage
if __name__ == "__main__":

    from models.sentencepiece.test import TEST_SENTENCES

    model_path = str(Path(__file__).parents[2] / "sp_400k_merged.model")
    tokenizer = ParallelSentenceFullTokenizer(
        model_path, drop_non_word=True, verbose=False
    )

    logger.note(
        f"> Parallel Tokenizing with workers: {logstr.mesg(brk(tokenizer.workers_num))}"
    )
    sentences = TEST_SENTENCES
    results = tokenizer.tokenize(sentences)

    for sentence, result in zip(sentences, results):
        logger.mesg(f"  * {result['str']}")

    tokenizer.close()

    # python -m models.sentencepiece.tokenizer_parallel
