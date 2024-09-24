import argparse
import sys

from copy import deepcopy
from gensim.models import FastText, KeyedVectors
from pathlib import Path
from tclogger import logger, logstr, dict_to_str
from typing import Union

from models.fasttext.data import VideosTagsDataLoader


class FasttextModelTrainer:
    def __init__(self):
        pass

    def load_model(self, model_path: Union[str, Path], use_local_model: bool = True):
        self.model_path = Path(model_path)
        self.model = None
        self.is_model_trained = False

        logger.note("> Loading model:")
        if self.model:
            logger.file("  * model already in memory")
            self.is_model_trained = True
            return

        if self.model_path.exists() and use_local_model:
            logger.file(f"  * load from local: [{self.model_path}]")
            self.model = FastText.load(str(self.model_path))
            self.is_model_trained = True
        else:
            self.model = FastText(vector_size=100, window=5, min_count=1)
            self.is_model_trained = False
            logger.success("  ✓ new model created")

    def load_data(self, max_count: int = None):
        logger.note(f"> Loading data: [{logstr.mesg(max_count)}]")
        if self.is_model_trained:
            logger.file("  * model already trained, skip load_data()")
        else:
            self.data_loader_params = {
                "collection": "videos_tags",
                "max_count": max_count,
                "iter_val": "tag_list",
                "iter_log": True,
            }
            self.data_loader = VideosTagsDataLoader(
                **self.data_loader_params,
            )
            logger.success("  ✓ data loader created")
            logger.mesg(dict_to_str(self.data_loader_params), indent=4)

    def build_vocab(self, skip_trained: bool = True):
        logger.note("> Building vocab:")
        if self.is_model_trained and skip_trained:
            logger.file("  * model already trained, skip build_vocab()")
        else:
            self.model.build_vocab(corpus_iterable=self.data_loader)
            # self.model_total_words = self.model.corpus_total_words
            logger.success(f"  ✓ vocab built")

    def train(
        self, epochs: int = 5, skip_trained: bool = True, overwrite: bool = False
    ):
        logger.note("> Training model:")
        if self.is_model_trained and skip_trained:
            logger.file("  * model already trained, skip train()")
        else:
            self.data_loader.iter_epochs = epochs
            self.model.train(
                corpus_iterable=self.data_loader,
                total_examples=self.data_loader.max_count,
                # total_words=self.model_total_words,
                epochs=epochs,
            )
            logger.success(f"  ✓ model trained")

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
                results = self.model.wv.most_similar(word)[:6]
                logger.mesg(f"  * [{logstr.file(word)}]:")
                for result in results:
                    res_word, res_score = result
                    logger.success(f"    * {res_score:>.4f}: {res_word}")

    def run(
        self,
        model_path: Union[str, Path],
        max_count: int = 10000,
        epochs: int = 5,
        use_local_model: bool = True,
        skip_trained: bool = True,
        overwrite: bool = True,
    ):
        self.load_model(model_path, use_local_model=use_local_model)
        self.load_data(max_count=max_count)
        self.build_vocab(skip_trained=skip_trained)
        self.train(epochs=epochs, skip_trained=skip_trained, overwrite=overwrite)


class ArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("-t", "--test-only", action="store_true")
        self.args, self.unknown_args = self.parse_known_args(sys.argv[1:])


if __name__ == "__main__":
    args = ArgParser().args

    trainer = FasttextModelTrainer()
    model_path = Path(__file__).parent / "fasttext.model"

    trainer_params = {
        "model_path": model_path,
        "max_count": 10000000,
        "epochs": 10,
    }

    if args.test_only:
        extra_params = {
            "use_local_model": True,
            "skip_trained": True,
            "overwrite": False,
        }
    else:
        extra_params = {
            "use_local_model": False,
            "skip_trained": False,
            "overwrite": True,
        }

    trainer.run(
        **trainer_params,
        **extra_params,
    )

    test_words = ["上海", "搞笑", "萌宠", "GTA", "高数"]
    trainer.test(test_words)

    # python -m models.fasttext.train
    # python -m models.fasttext.train -t
