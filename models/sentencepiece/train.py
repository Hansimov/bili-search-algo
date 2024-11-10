import sentencepiece as spm
import io

from collections.abc import Iterable
from tclogger import logger, logstr, Runtimer

from models.sentencepiece.data import SentencesDataloader
from models.sentencepiece.test import TEST_SENTENCES


class SentencePieceModelTrainer:
    def __init__(self, vocab_size=32000, model_prefix="sentencepiece"):
        self.vocab_size = vocab_size
        self.model_prefix = model_prefix

    def load_data(self, data_loader: Iterable[str] = None, max_batch: int = None):
        logger.note("> Loading data ...")
        if data_loader:
            self.data_loader = data_loader
        else:
            self.data_loader = SentencesDataloader(
                max_batch=max_batch, show_at_init=False, verbose=True
            )

    def train(self):
        logger.note("> Training ...")
        spm.SentencePieceTrainer.Train(
            sentence_iterator=iter(self.data_loader),
            model_prefix=self.model_prefix,
            vocab_size=self.vocab_size,
            model_type="bpe",
            character_coverage=0.9995,
        )

    def test(self, test_sentences: list[str]):
        logger.note("> Testing ...")
        sp = spm.SentencePieceProcessor(model_file=f"{self.model_prefix}.model")
        for sentence in test_sentences:
            logger.note(f"> {sentence}")
            tokens = sp.EncodeAsPieces(sentence)
            logger.success(f"  * {tokens}")


if __name__ == "__main__":
    trainer = SentencePieceModelTrainer(vocab_size=32000)
    trainer.load_data(max_batch=100)
    with Runtimer() as timer:
        trainer.train()
    trainer.test(TEST_SENTENCES)

    # python -m models.sentencepiece.train
