import argparse
import sentencepiece as spm
import sys


from collections.abc import Iterable
from tclogger import logger, logstr, Runtimer, dict_to_str
from typing import Literal

from models.sentencepiece.data import SentencesDataloader
from models.sentencepiece.test import TEST_SENTENCES


class SentencePieceModelTrainer:
    """
    Training Options:
    * https://github.com/google/sentencepiece/blob/master/doc/options.md
    """

    def __init__(
        self,
        add_dummy_prefix: bool = False,
        character_coverage: float = 0.9999,
        input_sentence_size: int = 1000000,
        minloglevel: int = 2,
        model_prefix="sentencepiece",
        model_type: Literal["unigram", "bpe", "char", "word"] = "unigram",
        num_threads: int = 16,
        split_by_unicode_script: bool = False,
        shrinking_factor: float = 0.75,
        treat_whitespace_as_suffix: bool = False,
        user_defined_symbols="▁",
        vocab_size: int = 32000,
    ):
        self.train_params = {
            "add_dummy_prefix": add_dummy_prefix,
            "character_coverage": character_coverage,
            "input_sentence_size": input_sentence_size,
            "minloglevel": minloglevel,
            "model_type": model_type,
            "model_prefix": model_prefix,
            "num_threads": num_threads,
            "split_by_unicode_script": split_by_unicode_script,
            "shrinking_factor": shrinking_factor,
            "treat_whitespace_as_suffix": treat_whitespace_as_suffix,
            "user_defined_symbols": user_defined_symbols,
            "vocab_size": vocab_size,
        }
        self.model_file = f"{model_prefix}.model"

    def load_data(
        self,
        data_loader: Iterable[str] = None,
        max_batch: int = 100,
        batch_size: int = 10000,
    ):
        logger.note("> Loading data ...")
        if data_loader:
            self.data_loader = data_loader
        else:
            self.data_loader = SentencesDataloader(
                max_batch=max_batch,
                batch_size=batch_size,
                show_at_init=False,
                verbose=True,
            )

    def train(self):
        logger.note("> Training ...")
        logger.mesg(dict_to_str(self.train_params), indent=2)
        spm.SentencePieceTrainer.Train(
            sentence_iterator=iter(self.data_loader),
            **self.train_params,
        )

    def test(self, test_sentences: list[str]):
        logger.note("> Testing ...")
        sp = spm.SentencePieceProcessor(model_file=self.model_file)
        for sentence in test_sentences:
            tokens = sp.EncodeAsPieces(sentence.lower())
            tokens_str = f"{logstr.note('_')}".join(tokens)
            logger.mesg(f"  * {tokens_str}")


class ArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("-mb", "--max-batch", type=int, default=20000)
        self.add_argument("-mp", "--model-prefix", type=str, default="sentencepiece")
        self.add_argument("-mt", "--model-type", type=str, default="unigram")
        self.add_argument("-t", "--test-only", action="store_true")
        self.add_argument("-vs", "--vocab-size", type=int, default=32000)
        self.args, self.unknown_args = self.parse_known_args(sys.argv[1:])


if __name__ == "__main__":
    args = ArgParser().args
    trainer = SentencePieceModelTrainer(
        model_prefix=args.model_prefix,
        model_type=args.model_type,
        vocab_size=args.vocab_size,
    )
    if not args.test_only:
        trainer.load_data(max_batch=args.max_batch, batch_size=10000)
        with Runtimer() as timer:
            trainer.train()
    trainer.test(TEST_SENTENCES)

    # python -m models.sentencepiece.train
    # python -m models.sentencepiece.train -mp sp_380m_500k -t

    # python -m models.sentencepiece.train -mp sp_380m_500k -mb 38000 -vs 500000
    # python -m models.sentencepiece.train -mp sp_400m_500k_bpe -mb 40000 -vs 500000 -mt bpe
