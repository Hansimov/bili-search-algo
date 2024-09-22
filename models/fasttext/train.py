from gensim.models import FastText
from gensim.test.utils import common_texts
from pathlib import Path
from tclogger import logger, logstr

from models.fasttext.data import DataLoader


class FasttextModelTrainer:
    def __init__(self):
        self.model = None
        self.is_model_trained = False

    def load_data(self):
        logger.note("> Loading data:")
        logger.mesg(common_texts)
        # self.data_loader = DataLoader()
        # method: self.data_loader.next()
        logger.success(f"  ✓ data loaded")

    def load_model(self):
        logger.note("> Loading model:")
        if self.model:
            logger.file("  * model already in memory")
            self.is_model_trained = True
            return

        self.model_path = Path(__file__).parent / "fasttext.model"
        if self.model_path.exists():
            self.model = FastText.load(str(self.model_path))
            self.is_model_trained = True
            logger.file(f"  * load from local: [{self.model_path}]")
        else:
            self.model = FastText(vector_size=10, min_count=1)
            self.is_model_trained = False
            logger.success("  ✓ new model created")

    def build_vocab(self, skip_trained: bool = True):
        logger.note("> Building vocab:")
        if self.is_model_trained and skip_trained:
            logger.file("  * model already trained, skip build_vocab()")
        else:
            self.model.build_vocab(corpus_iterable=common_texts)
            logger.success("  ✓ vocab built")

    def train(self, epochs: int = 5, batch_size: int = 1000, skip_trained: bool = True):
        logger.note("> Training model:")
        if self.is_model_trained and skip_trained:
            logger.file("  * model already trained, skip train()")
        else:
            self.model.train(
                corpus_iterable=common_texts,
                total_examples=len(common_texts),
                epochs=epochs,
            )
            logger.success("  ✓ model trained")
            logger.note("> Saving model:")
            self.model.save(str(self.model_path))
            logger.success(f"  * [{self.model_path}]")

    def test(self):
        logger.note("> Testing model:")
        logger.mesg(self.model.wv.most_similar("computer")[0])
        logger.mesg(self.model.wv.most_similar("human")[0])
        logger.mesg(self.model.wv.most_similar("language")[0])

    def run(self):
        self.load_data()
        self.load_model()
        self.build_vocab()
        self.train()
        self.test()


if __name__ == "__main__":
    trainer = FasttextModelTrainer()
    trainer.run()

    # python -m models.fasttext.train
