import sentencepiece as spm
import io

from collections.abc import Iterable
from tclogger import logger, logstr, Runtimer
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
        vocab_size: int = 32000,
        character_coverage: float = 0.9995,
        model_type: Literal["unigram", "bpe", "char", "word"] = "unigram",
        model_prefix="sentencepiece",
        add_dummy_prefix: bool = False,
        minloglevel: int = 2,
    ):
        self.vocab_size = vocab_size
        self.character_coverage = character_coverage
        self.model_type = model_type
        self.model_prefix = model_prefix
        self.add_dummy_prefix = add_dummy_prefix
        self.minloglevel = minloglevel

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
        spm.SentencePieceTrainer.Train(
            sentence_iterator=iter(self.data_loader),
            model_prefix=self.model_prefix,
            vocab_size=self.vocab_size,
            model_type=self.model_type,
            character_coverage=self.character_coverage,
            add_dummy_prefix=self.add_dummy_prefix,
            minloglevel=self.minloglevel,
        )

    def test(self, test_sentences: list[str]):
        logger.note("> Testing ...")
        sp = spm.SentencePieceProcessor(model_file=f"{self.model_prefix}.model")
        for sentence in test_sentences:
            logger.note(f"> {sentence}")
            tokens = sp.EncodeAsPieces(sentence)
            logger.success(f"  * {tokens}")


if __name__ == "__main__":
    trainer = SentencePieceModelTrainer(vocab_size=64000)
    trainer.load_data(max_batch=100, batch_size=10000)
    with Runtimer() as timer:
        trainer.train()
    trainer.test(TEST_SENTENCES)

    # python -m models.sentencepiece.train
