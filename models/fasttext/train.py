from copy import deepcopy
from gensim.models import FastText, KeyedVectors
from pathlib import Path
from tclogger import logger, logstr
from typing import Union

from models.fasttext.data import VideosTagsDataLoader


class FasttextModelTrainer:
    def __init__(self):
        pass

    def load_data(self):
        logger.note("> Loading data:")
        self.data_loader: VideosTagsDataLoader = VideosTagsDataLoader()
        # self.data_loader = DemoDataLoader()
        logger.file(f"  * from class {self.data_loader.__class__.__name__}")
        logger.success(f"  ✓ data loaded")

    def load_model(self, model_path: Union[str, Path], use_local: bool = True):
        self.model_path = Path(model_path)
        self.model = None
        self.is_model_trained = False

        logger.note("> Loading model:")
        if self.model:
            logger.file("  * model already in memory")
            self.is_model_trained = True
            return

        if self.model_path.exists() and use_local:
            self.model = FastText.load(str(self.model_path))
            self.is_model_trained = True
            logger.file(f"  * load from local: [{self.model_path}]")
        else:
            self.model = FastText(vector_size=100, window=3, min_count=1)
            self.is_model_trained = False
            logger.success("  ✓ new model created")

    def build_vocab(self, skip_trained: bool = True):
        logger.note("> Building vocab:")
        if self.is_model_trained and skip_trained:
            logger.file("  * model already trained, skip build_vocab()")
        else:
            self.model.build_vocab(corpus_iterable=self.data_loader)
            self.model_total_words = self.model.corpus_total_words
            logger.success("  ✓ vocab built")

    def train(
        self, epochs: int = 5, skip_trained: bool = True, overwrite: bool = False
    ):
        logger.note("> Training model:")
        if self.is_model_trained and skip_trained:
            logger.file("  * model already trained, skip train()")
        else:
            self.model.train(
                corpus_iterable=self.data_loader,
                total_examples=self.data_loader.max_count,
                # total_words=self.model_total_words,
                epochs=epochs,
            )
            logger.success("  ✓ model trained")

            if self.model_path.exists() and not overwrite:
                logger.file(f"> Skip saving existed model")
                logger.file(f"  * [{self.model_path}]")
            else:
                logger.note("> Saving model:")
                if not self.model_path.parent.exists():
                    self.model_path.parent.mkdir(parents=True, exist_ok=True)
                self.model.save(str(self.model_path))
                logger.success(f"  * [{self.model_path}]")

    def test(self, test_words: list[str] = None):
        logger.note("> Testing model:")
        if not test_words:
            logger.warn("  × No test words provided")
        else:
            for word in test_words:
                similar_words = self.model.wv.most_similar(word)[:6]
                logger.mesg(f"  * [{logstr.file(word)}]:")
                for similar_word in similar_words:
                    logger.success(f"    * {similar_word}")

    def run(
        self,
        model_path: Union[str, Path],
        epochs: int = 5,
        use_local: bool = True,
        skip_trained: bool = True,
        overwrite: bool = True,
    ):
        self.load_data()
        self.load_model(model_path, use_local=use_local)
        self.build_vocab(skip_trained=skip_trained)
        self.train(epochs=epochs, skip_trained=skip_trained, overwrite=overwrite)


if __name__ == "__main__":
    trainer = FasttextModelTrainer()
    model_path = Path(__file__).parent / "fasttext.model"
    test_words = ["香港", "搞笑", "萌宠", "GTA"]

    # trainer.run(
    #     model_path,
    #     epochs=5,
    #     use_local=False,
    #     skip_trained=False,
    #     overwrite=True,
    # )
    trainer.run(
        model_path,
        epochs=5,
        use_local=True,
        skip_trained=True,
        overwrite=False,
    )
    trainer.test(test_words)

    # python -m models.fasttext.train
