import argparse
import sys

from gensim.models import FastText, KeyedVectors
from pathlib import Path

from tclogger import logger, logstr, dict_to_str, brk, Runtimer
from typing import Union

from configs.envs import FASTTEXT_CKPT_ROOT
from models.sentencepiece.tokenizer import SentenceFullTokenizer


class FasttextModelRunner:
    def __init__(
        self,
        model_prefix: Union[str, Path] = "fasttext",
        restrict_vocab: int = 150000,
        verbose: bool = False,
    ):
        self.model_prefix = model_prefix
        self.restrict_vocab = restrict_vocab
        self.verbose = verbose

    def list_models(self):
        model_files = list(FASTTEXT_CKPT_ROOT.glob(f"*.model"))
        model_files.sort()
        logger.file(f"> List local models:")
        model_info_dict = {}
        for model_file in model_files:
            model_prefix = model_file.stem
            vocab_file = (
                FASTTEXT_CKPT_ROOT / f"{model_prefix}.model.wv.vectors_vocab.npy"
            )
            model_size_in_mb = round(model_file.stat().st_size / 1024 / 1024)
            try:
                vocab_size_in_mb = round(vocab_file.stat().st_size / 1024 / 1024)
            except Exception as e:
                vocab_size_in_mb = None
            model_info_dict[model_prefix] = (model_size_in_mb, vocab_size_in_mb)
        logger.file(dict_to_str(model_info_dict))

    def load_model(self):
        self.model_path = FASTTEXT_CKPT_ROOT / f"{self.model_prefix}.model"
        if self.verbose:
            logger.note(f"> Loading model:")
            logger.mesg(f"  * {self.model_path}")
            logger.mesg(f"  * restrict_vocab: {logstr.file(brk(self.restrict_vocab))}")
        self.model = FastText.load(str(self.model_path))

    def preprocess_words(self, words: Union[str, list[str]]):
        if isinstance(words, list):
            return [w.lower() for w in words]
        else:
            return words.lower()

    def wv_func(self, func: str, *args, **kwargs):
        if self.verbose:
            logger.note(f"> model.wv.{func}:")
            logger.mesg(f"  * args  : {args}")
            logger.mesg(f"  * kwargs: {kwargs}")
        return getattr(self.model.wv, func)(*args, **kwargs)

    def most_similar(
        self,
        positive: list = None,
        negative: list = None,
        topn: int = 10,
        restrict_vocab: int = None,
    ):
        results = self.model.wv.most_similar(
            positive=positive,
            negative=negative,
            topn=topn,
            restrict_vocab=restrict_vocab or self.restrict_vocab,
        )
        return results

    def test(self, test_words: list):
        for word in test_words:
            word = self.preprocess_words(word)
            logger.mesg(f"  * [{logstr.file(word)}]:")
            results = self.most_similar(positive=word, topn=10)[:6]
            for result in results:
                res_word, res_score = result
                logger.success(f"    * {res_score:>.4f}: {res_word}")
        logger.file(f"* {self.model_prefix}")

    def test_func(self, test_words: list):
        for word in test_words:
            word = self.preprocess_words(word)
            logger.mesg(f"  * [{logstr.file(word)}]:")
            results = self.wv_func("most_similar", positive=word, topn=10)[:6]
            for result in results:
                res_word, res_score = result
                logger.success(f"    * {res_score:>.4f}: {res_word}")
        logger.file(f"* {self.model_prefix}")


class FasttextModelRunnerArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("-m", "--model-prefix", type=str, default="fasttext")
        self.add_argument("-r", "--restrict-vocab", type=int, default=150000)
        self.add_argument("-l", "--list-models", action="store_true")

    def parse_args(self):
        self.args, self.unknown_args = self.parse_known_args(sys.argv[1:])
        return self.args


if __name__ == "__main__":
    from models.fasttext.test import TEST_KEYWORDS

    parser = FasttextModelRunnerArgParser()
    args = parser.parse_args()

    timer = Runtimer()
    timer.__enter__()

    runner = FasttextModelRunner(
        model_prefix=args.model_prefix, restrict_vocab=args.restrict_vocab, verbose=True
    )

    if args.list_models:
        runner.list_models()
    else:
        runner.load_model()
        # runner.test(TEST_KEYWORDS)
        runner.test_func(TEST_KEYWORDS)

    timer.__exit__(None, None, None)

    # python -m models.fasttext.run -l
    # python -m models.fasttext.run -m fasttext_tid_all_mv_60w -r 150000
    # python -m models.fasttext.run -m fasttext_tid_all_mv_30w -r 150000
