import argparse
import pandas as pd
import sys

from gensim.models import FastText, KeyedVectors
from pathlib import Path
from tclogger import logger, logstr, dict_to_str, brk
from typing import Union

from datasets.videos.data import CommonDataLoaderArgParser
from datasets.videos.data import SentencesDataloader, SentencesDataLoaderArgParser
from datasets.videos.data import ParquetRowsDataLoader, ParquetRowsDataLoaderArgParser
from datasets.videos.parquet import VideoTextsParquetReader, ParquetOperatorArgParser
from models.fasttext.test import TEST_KEYWORDS

from models.sentencepiece.tokenizer import SentenceFullTokenizer
from models.sentencepiece.tokenizer_parallel import ParallelSentenceFullTokenizer


class FasttextModelVocabLoader:
    def __init__(
        self, vocab_path: Union[str, Path], max_count: int = None, verbose: bool = False
    ):
        self.vocab_path = Path(vocab_path)
        self.max_count = max_count
        self.verbose = verbose
        self.vocab_dict = {}

    def load(self) -> dict[str, int]:
        if self.verbose:
            logger.note("> Loading vocab from file:")
            logger.file(f"  * {self.vocab_path}")
        df = pd.read_csv(self.vocab_path)
        # df = df.sort_values(by="doc_freq", ascending=False)
        if self.max_count:
            df = df.head(self.max_count)
        df = df.sort_values(by="term_freq", ascending=False)
        self.vocab_dict = dict(zip(df["token"], df["term_freq"]))
        if self.verbose:
            tokens_count_str = brk(logstr.file(df.shape[0]))
            logger.mesg(f"  * tokens count: {tokens_count_str}")
            min_term_freq_str = brk(logstr.file(df.tail(1)["term_freq"].values[0]))
            logger.mesg(f"  * min term_freq: {min_term_freq_str}")
        return self.vocab_dict


class FasttextModelDataLoader(SentencesDataloader):
    def __iter__(self):
        self.__epoch_start__()
        for batch_idx, batch in enumerate(
            self.doc_batch_generator(doc_type="sentence")
        ):
            if self.max_batch is not None and batch_idx >= self.max_batch:
                break
            tokenize_results: list[dict[str, str]] = self.tokenizer.tokenize_list(
                batch, sort=False
            )
            for result in tokenize_results:
                yield result["tokens"]
        self.__epoch_end__()


class FasttextModelParquetDataLoader(ParquetRowsDataLoader):
    def __iter__(self):
        self.__epoch_start__()
        self.batch_bar.reset()
        self.batch_bar.update(flush=True)
        for batch_idx, tokens_batch in enumerate(self.batch_generator):
            self.batch_bar.update(increment=1, flush=True)
            if self.max_batch is not None and batch_idx >= self.max_batch:
                break
            self.sample_bar.total = len(tokens_batch)
            self.sample_bar.update(flush=True)
            for tokens in tokens_batch:
                self.sample_bar.update(1)
                yield tokens
            self.sample_bar.reset()
        self.__epoch_end__()


class FasttextModelTrainer:
    def __init__(
        self,
        model_prefix: Union[str, Path] = "fasttext",
        epochs: int = 5,
        hs: int = 0,
        min_alpha: float = 0.0001,
        min_count: int = 5,
        min_n: int = 3,
        max_n: int = 6,
        shrink_windows: bool = False,
        sg: int = 0,
        vector_size: int = 128,
        window: int = 5,
        workers: int = 8,
        keep_exist_model: bool = True,
        skip_trained: bool = True,
        use_local_model: bool = True,
        vocab_dict: dict[str, int] = None,
    ):
        self.model_prefix = model_prefix
        self.keep_exist_model = keep_exist_model
        self.skip_trained = skip_trained
        self.use_local_model = use_local_model
        self.vocab_dict = vocab_dict

        self.model_path = (
            Path(__file__).parent / "checkpoints" / f"{model_prefix}.model"
        )
        self.model = None
        self.is_model_trained = False

        self.train_params = {
            "epochs": epochs,
            "hs": hs,
            "min_alpha": min_alpha,
            "min_count": min_count,
            "min_n": min_n,
            "max_n": max_n,
            "shrink_windows": shrink_windows,
            "sg": sg,
            "vector_size": vector_size,
            "window": window,
            "workers": workers,
        }

        logger.note("> Loading model:")
        if self.model:
            logger.file("  * model already in memory")
            self.is_model_trained = True
            return

        if self.model_path.exists() and self.use_local_model:
            logger.mesg(f"  * load from local:")
            logger.file(f"    * {brk(self.model_path)}")
            self.model = FastText.load(str(self.model_path))
            self.is_model_trained = True
        else:
            self.model = FastText(**self.train_params)
            self.is_model_trained = False
            logger.success(
                f"  ✓ new model created: {logstr.mesg(brk(self.model_prefix))}"
            )
            logger.mesg(dict_to_str(self.train_params), indent=4)

    def init_data_loader(self, data_loader: SentencesDataloader):
        self.data_loader = data_loader

    def build_vocab(self):
        logger.note("> Building vocab:")
        if self.is_model_trained and self.skip_trained:
            logger.file("  * model already trained, skip build_vocab()")
        elif self.vocab_dict:
            self.model.build_vocab_from_freq(self.vocab_dict)
            logger.success("  ✓ build vocab from provided freq dict")
        else:
            self.model.build_vocab(corpus_iterable=self.data_loader)
            logger.success(f"  ✓ vocab built")

    def train(self):
        logger.note("> Training model:")
        self.data_loader.epoch_bar.reset()
        if self.is_model_trained and self.skip_trained:
            logger.file("  * model already trained, skip train()")
        else:
            self.data_loader.epoch_bar.reset()
            self.data_loader.iter_epochs = self.train_params["epochs"]
            self.model.train(
                corpus_iterable=self.data_loader,
                total_examples=self.model.corpus_count,
                epochs=self.model.epochs,
                # total_words=self.self.model.corpus_total_words,
            )
            logger.success(f"  ✓ model trained")

            if self.model_path.exists() and self.keep_exist_model:
                logger.file(f"> Skip saving existed model")
                logger.file(f"  * [{self.model_path}]")
            else:
                logger.note("> Saving model:")
                if not self.model_path.parent.exists():
                    self.model_path.parent.mkdir(parents=True, exist_ok=True)
                self.model.save(str(self.model_path))
                logger.success(f"  * [{self.model_path}]")

    def test(
        self,
        test_words: list[str] = None,
        tokenizer: SentenceFullTokenizer = None,
        restrict_vocab: int = 10000,
    ):
        logger.note("> Testing model:")
        if not test_words:
            logger.warn("  × No test words provided")
        else:
            for word in test_words:
                if tokenizer and len(word) >= 3:
                    word = tokenizer.tokenize(word.lower())
                elif isinstance(word, list):
                    word = [w.lower() for w in word]
                else:
                    word = word.lower()
                results = self.model.wv.most_similar(
                    word, restrict_vocab=restrict_vocab
                )[:6]
                logger.mesg(f"  * [{logstr.file(word)}]:")
                for result in results:
                    res_word, res_score = result
                    logger.success(f"    * {res_score:>.4f}: {res_word}")


class ModelTrainerArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("-m", "--model-prefix", type=str, default="fasttext")
        self.add_argument("-ep", "--epochs", type=int, default=5)
        self.add_argument("-hs", "--hs", type=int, default=0)
        self.add_argument("-ma", "--min-alpha", type=float, default=0.0001)
        self.add_argument("-mc", "--min-count", type=int, default=20)
        self.add_argument("-in", "--min-n", type=int, default=2)
        self.add_argument("-an", "--max-n", type=int, default=8)
        self.add_argument("-sg", "--sg", type=int, default=0)
        self.add_argument("-sw", "--shrink-windows", action="store_true")
        self.add_argument("-vs", "--vector-size", type=int, default=256)
        self.add_argument("-vf", "--vocab-file", type=str, default=None)
        self.add_argument("-vm", "--vocab-max-count", type=int, default=None)
        self.add_argument("-wd", "--window", type=int, default=5)
        self.add_argument("-wk", "--workers", type=int, default=32)
        self.add_argument("-t", "--test-only", action="store_true")
        self.add_argument("-k", "--keep-exist-model", action="store_false")
        self.add_argument("-s", "--skip-trained", action="store_true")
        self.add_argument("-u", "--use-local-model", action="store_true")
        self.add_argument("-tm", "--tokenizer-model", type=str, default=None)

    def parse_args(self):
        self.args, self.unknown_args = self.parse_known_args(sys.argv[1:])
        return self.args


if __name__ == "__main__":
    data_loader_parser = DataLoaderArgParser(add_help=False)
    model_trainer_parser = ModelTrainerArgParser(add_help=False)
    merged_parser = argparse.ArgumentParser(
        parents=[data_loader_parser, model_trainer_parser]
    )
    args, unknown_args = merged_parser.parse_known_args(sys.argv[1:])

    if args.test_only:
        args.keep_exist_model = True
        args.skip_trained = True
        args.use_local_model = True
    else:
        args.keep_exist_model = False
        args.skip_trained = False
        args.use_local_model = False

    if args.vocab_file:
        vocab_loader = FasttextModelVocabLoader(
            args.vocab_file, args.vocab_max_count, verbose=True
        )
        vocab_dict = vocab_loader.load()
    else:
        vocab_dict = None

    trainer = FasttextModelTrainer(
        model_prefix=args.model_prefix,
        epochs=args.epochs,
        hs=args.hs,
        min_alpha=args.min_alpha,
        min_count=args.min_count,
        min_n=args.min_n,
        max_n=args.max_n,
        shrink_windows=args.shrink_windows,
        sg=args.sg,
        vector_size=args.vector_size,
        window=args.window,
        workers=args.workers,
        keep_exist_model=args.keep_exist_model,
        skip_trained=args.skip_trained,
        use_local_model=args.use_local_model,
        vocab_dict=vocab_dict,
    )

    tokenizer = ParallelSentenceFullTokenizer(
        Path("sp_400k_merged.model"),
        # drop_non_word=True, # This param is not needed as doc_coverter in data_loader already does this
        drop_whitespace=True,
        workers_num=16,
        batch_size=args.batch_size * 2,
    )

    if not args.test_only:
        logger.note("> Initiating data loader ...")
        mongo_filter = {"tid": 17}
        data_fields = args.data_fields.split(",") if args.data_fields else None
        data_params = {
            "dbname": args.dbname,
            "collect_name": args.collect_name,
            "data_fields": data_fields,
            "mongo_filter": mongo_filter,
            "max_batch": args.max_batch,
            "batch_size": args.batch_size,
            "estimate_count": args.estimate_count,
            "iter_val": "sentence",
            "task_type": "fasttext",
        }
        data_loader = FasttextModelDataLoader(
            **data_params, tokenizer=tokenizer, show_at_init=False, verbose=True
        )
        logger.mesg(dict_to_str(data_params), indent=2)
        trainer.init_data_loader(data_loader)
        trainer.build_vocab()
        trainer.train()
        data_loader.tokenizer.terminate()

    trainer.test(TEST_KEYWORDS, tokenizer=None, restrict_vocab=50000)

    # python -m models.fasttext.train
    # python -m models.fasttext.train -t

    # python -m models.fasttext.train -ep 3 -m fasttext_tid_17_ep_3
    # python -m models.fasttext.train -m fasttext_tid_17_ep_3_parallel -ep 3 -vf "video_texts_freq_all.csv" -vm 1200000
    # python -m models.fasttext.train -m fasttext_tid_17_ep_2_parallel -ep 2 -mc 20
