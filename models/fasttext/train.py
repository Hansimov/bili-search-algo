import argparse
import json
import pandas as pd
import sys

from gensim.models import FastText, KeyedVectors
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from pathlib import Path
from tclogger import logger, logstr, dict_to_str, brk, Runtimer
from typing import Union

from datasets.videos.data import SentencesDataloader, ParquetRowsDataLoader
from datasets.videos.parquet import VideoTextsParquetReader
from datasets.args import DATA_LOADER_ARG_PARSER
from models.fasttext.test import TEST_KEYWORDS

from models.sentencepiece.tokenizer import SentenceFullTokenizer
from models.sentencepiece.tokenizer_parallel import ParallelSentenceFullTokenizer


class FasttextModelVocabLoader:
    def __init__(
        self,
        vocab_path: Union[str, Path],
        max_vocab_count: int = None,
        verbose: bool = False,
    ):
        self.vocab_path = Path(vocab_path)
        self.max_vocab_count = max_vocab_count
        self.verbose = verbose
        self.vocab_dict = {}

    def load(self) -> dict[str, int]:
        if self.verbose:
            logger.note("> Loading vocab from file:")
            logger.file(f"  * {self.vocab_path}")
        df = pd.read_csv(self.vocab_path)
        # df = df.sort_values(by="doc_freq", ascending=False)
        if self.max_vocab_count:
            df = df.head(self.max_vocab_count)
        df = df.sort_values(by="term_freq", ascending=False)
        self.vocab_dict = dict(zip(df["token"], df["term_freq"]))
        if self.verbose:
            vocab_info = {
                "vocab_size": df.shape[0],
                "min_term_freq": df.tail(1)["term_freq"].values[0],
            }
            logger.mesg(dict_to_str(vocab_info), indent=2)
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
        sample_idx = 0
        for batch_idx, tokens_batch in enumerate(self.batch_generator):
            self.batch_bar.update(increment=1, flush=True)
            if self.max_batch is not None and batch_idx >= self.max_batch:
                break
            self.sample_bar.total = len(tokens_batch)
            self.sample_bar.update(flush=True)
            for tokens in tokens_batch:
                self.sample_bar.update(1)
                sample_idx += 1
                if hasattr(self, "model_class") and self.model_class == "doc2vec":
                    yield TaggedDocument(tokens, [sample_idx])
                else:
                    yield tokens
            self.sample_bar.reset()
        self.__epoch_end__()

    def get_count_from_local(self):
        count_file = self.parquet_reader.data_root / (
            self.parquet_reader.dataset_name + ".count.json"
        )
        if count_file.exists():
            with count_file.open("r") as f:
                count_dict = json.load(f)
            return count_dict["total_words"], count_dict["corpus_count"]
        else:
            return None, None

    def count(self) -> tuple[int, int]:
        logger.note("> Counting vocab:")
        total_words, corpus_count = self.get_count_from_local()
        if not total_words or not corpus_count:
            total_words, corpus_count = 0, 0
            for tokens in self:
                total_words += len(tokens)
                corpus_count += 1
        count_info = {
            "total_words": total_words,
            "corpus_count": corpus_count,
        }
        logger.mesg(dict_to_str(count_info), indent=2)
        return total_words, corpus_count


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
        max_final_vocab: int = None,
        shrink_windows: bool = False,
        sg: int = 0,
        vector_size: int = 128,
        window: int = 5,
        workers: int = 8,
        keep_exist_model: bool = True,
        skip_trained: bool = True,
        use_local_model: bool = True,
        use_kv: bool = False,
        vocab_loader: FasttextModelVocabLoader = None,
    ):
        self.model_prefix = model_prefix
        self.keep_exist_model = keep_exist_model
        self.skip_trained = skip_trained
        self.use_local_model = use_local_model
        self.use_kv = use_kv
        self.vocab_loader = vocab_loader

        self.model_path = (
            Path(__file__).parent / "checkpoints" / f"{model_prefix}.model"
        )
        self.kv_path = self.model_path.with_suffix(".kv")
        self.model = None
        self.is_model_trained = False

        self.train_params = {
            "epochs": epochs,
            "hs": hs,
            "min_alpha": min_alpha,
            "min_count": min_count,
            "min_n": min_n,
            "max_n": max_n,
            "max_final_vocab": max_final_vocab,
            "shrink_windows": shrink_windows,
            "sg": sg,
            "vector_size": vector_size,
            "window": window,
            "workers": workers,
        }
        self.prepare_train()
        self.init_model()

    def prepare_train(self):
        self.model_class = FastText

    def load_model(self):
        if self.use_kv:
            if not self.kv_path.exists():
                self.model = self.model_class.load(str(self.model_path))
                self.model.wv.save(str(self.kv_path))
            self.wv = KeyedVectors.load(str(self.kv_path))
        else:
            self.model = self.model_class.load(str(self.model_path))

    def init_model(self):
        logger.note("> Initializing model:")
        if self.model:
            logger.file("  * model already in memory")
            self.is_model_trained = True
            return
        if self.model_path.exists() and self.use_local_model:
            logger.mesg(f"  * load from local:")
            if self.use_kv:
                logger.file(f"    * {brk(self.kv_path)}")
            else:
                logger.file(f"    * {brk(self.model_path)}")
            self.load_model()
            self.is_model_trained = True
        else:
            self.model = self.model_class(**self.train_params)
            self.is_model_trained = False
            model_prefix_str = logstr.mesg(brk(self.model_prefix))
            logger.success(f"  ✓ new model created: {model_prefix_str}")
            logger.mesg(dict_to_str(self.train_params), indent=4)

    def init_data_loader(
        self,
        data_loader: Union[FasttextModelDataLoader, FasttextModelParquetDataLoader],
    ):
        self.data_loader = data_loader

    def build_vocab(self):
        logger.note("> Building vocab:")
        if self.is_model_trained and self.skip_trained:
            logger.file("  * model already trained, skip build_vocab()")
        elif self.vocab_loader:
            vocab_dict = self.vocab_loader.load()
            total_words, corpus_count = self.data_loader.count()
            self.model.corpus_total_words = total_words
            self.model.build_vocab_from_freq(vocab_dict, corpus_count=corpus_count)
            logger.success("  ✓ build vocab from provided freq dict")
        else:
            self.model.build_vocab(corpus_iterable=self.data_loader)
            logger.success(f"  ✓ vocab built")

    def refresh_data_loader_status(self):
        self.data_loader.iter_epochs = self.train_params["epochs"]
        self.data_loader.epoch_bar.reset()
        self.data_loader.init_generators()

    def delete_model(self):
        model_prefix_str = logstr.note(brk(self.model_prefix))
        if not self.model_path.exists():
            logger.mesg(f"  * Skip delete non-existed model: {model_prefix_str}")
            return
        logger.warn(f"  ! WARNING: You are deleting model: {model_prefix_str}")
        confirmation = input(
            logstr.mesg(
                f'  > Type "{logstr.note(self.model_prefix)}" to confirm deletion: '
            )
        )
        if confirmation != self.model_prefix:
            logger.mesg(f"  * Skip delete model: {model_prefix_str}")
        else:
            logger.warn(f"  ! Deleting model: {model_prefix_str}")
            model_files = sorted(self.model_path.parent.glob(f"{self.model_prefix}.*"))
            for model_file in model_files:
                model_file.unlink()

    def save_model(self):
        if not self.model_path.parent.exists():
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(self.model_path))
        if self.use_kv:
            self.model.wv.save(str(self.model_path.with_suffix(".kv")))

    def train(self):
        logger.note("> Training model:")
        if self.is_model_trained and self.skip_trained:
            logger.file("  * model already trained, skip train()")
        else:
            self.refresh_data_loader_status()
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
                self.save_model()
                logger.success(f"  * [{self.model_path}]")

    def test(
        self, tokenizer: SentenceFullTokenizer = None, restrict_vocab: int = 150000
    ):
        test_words = TEST_KEYWORDS
        restrict_vocab_str = logstr.mesg(brk(restrict_vocab))
        logger.note(f"> Testing model: {restrict_vocab_str}")
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
                logger.mesg(f"  * [{logstr.file(word)}]:")
                # vec = self.model.wv[word]
                # logger.success(vec)
                results = self.model.wv.most_similar(
                    word, restrict_vocab=restrict_vocab
                )[:6]
                for result in results:
                    res_word, res_score = result
                    logger.success(f"    * {res_score:>.4f}: {res_word}")


class Doc2VecModelTrainer(FasttextModelTrainer):
    def __init__(
        self,
        dm: Union[0, 1] = 1,
        dm_mean: Union[0, 1] = None,
        dm_concat: Union[0, 1] = 0,
        dm_tag_count: int = 1,
        dbow_words: Union[0, 1] = 0,
        *args,
        **kwargs,
    ):
        self.doc2vec_params = {
            "dm": dm,
            "dm_mean": dm_mean,
            "dm_concat": dm_concat,
            "dm_tag_count": dm_tag_count,
            "dbow_words": dbow_words,
        }
        super().__init__(*args, **kwargs)

    def prepare_train(self):
        self.model_class = Doc2Vec
        self.train_params.update(self.doc2vec_params)
        for param in ["sg", "min_n", "max_n"]:
            self.train_params.pop(param, None)

    def test(self, tokenizer: SentenceFullTokenizer):
        def vector_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
            return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

        logger.note(f"> Testing doc2vec model:")
        for pair in TEST_PAIRS:
            query = pair[0]
            samples = pair[1]
            if isinstance(query, list):
                query_tokens = query
            else:
                query_tokens = tokenizer.tokenize(query)
            query_vector = self.model.infer_vector(query_tokens)
            sample_vectors = []
            sample_tokens_list = []
            for sample in samples:
                sample_tokens = tokenizer.tokenize(sample)
                sample_vector = self.model.infer_vector(sample_tokens)
                sample_vectors.append(sample_vector)
                sample_tokens_list.append(sample_tokens)
            scores = [
                round(vector_similarity(query_vector, sample_vector), 4)
                for sample_vector in sample_vectors
            ]
            sample_scores = list(zip(samples, sample_tokens_list, scores))
            sample_scores.sort(key=lambda x: x[-1], reverse=True)
            logger.note(f"  * [{logstr.file(query)}]: ")
            for sample, sample_tokens, score in sample_scores:
                logger.success(f"    * {score:>.4f}: [{logstr.file(sample_tokens)}]")


class ModelTrainerArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("-m", "--model-prefix", type=str, default="fasttext")
        self.add_argument(
            "-ds",
            "--data-source",
            type=str,
            choices=["mongo", "parquet"],
            default="parquet",
        )
        self.add_argument("-ep", "--epochs", type=int, default=5)
        self.add_argument("-hs", "--hs", type=int, default=0)
        self.add_argument("-ma", "--min-alpha", type=float, default=0.0001)
        self.add_argument("-mc", "--min-count", type=int, default=20)
        self.add_argument("-mv", "--max-final-vocab", type=int, default=None)
        self.add_argument("-minn", "--min-n", type=int, default=2)
        self.add_argument("-maxn", "--max-n", type=int, default=8)
        self.add_argument("-sg", "--sg", type=int, default=0)
        self.add_argument("-sw", "--shrink-windows", action="store_true")
        self.add_argument("-vs", "--vector-size", type=int, default=256)
        self.add_argument("-vf", "--vocab-file", type=str, default=None)
        self.add_argument("-vm", "--vocab-max-count", type=int, default=None)
        self.add_argument("-wd", "--window", type=int, default=5)
        self.add_argument("-wk", "--workers", type=int, default=32)
        self.add_argument("-t", "--test-only", action="store_true")
        self.add_argument("-km", "--keep-exist-model", action="store_false")
        self.add_argument("-st", "--skip-trained", action="store_true")
        self.add_argument("-lm", "--use-local-model", action="store_true")
        self.add_argument("-kv", "--use-kv", action="store_true")
        self.add_argument("-tm", "--tokenizer-model", type=str, default=None)

    def parse_args(self):
        self.args, self.unknown_args = self.parse_known_args(sys.argv[1:])
        return self.args


if __name__ == "__main__":
    timer = Runtimer()
    timer.__enter__()
    arg_parser = DATA_LOADER_ARG_PARSER
    arg_parser.add_parser_class(ModelTrainerArgParser)
    args = arg_parser.parse_args()

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
            args.vocab_file, max_vocab_count=args.vocab_max_count, verbose=True
        )
    else:
        vocab_loader = None

    trainer = FasttextModelTrainer(
        model_prefix=args.model_prefix,
        epochs=args.epochs,
        hs=args.hs,
        min_alpha=args.min_alpha,
        min_count=args.min_count,
        min_n=args.min_n,
        max_n=args.max_n,
        max_final_vocab=args.max_final_vocab,
        shrink_windows=args.shrink_windows,
        sg=args.sg,
        vector_size=args.vector_size,
        window=args.window,
        workers=args.workers,
        keep_exist_model=args.keep_exist_model,
        skip_trained=args.skip_trained,
        use_local_model=args.use_local_model,
        use_kv=args.use_kv,
        vocab_loader=vocab_loader,
    )

    if not args.test_only:
        data_source_str = logstr.mesg(brk(args.data_source))
        logger.note(f"> Initiating data loader: {data_source_str}")
        if args.data_source == "mongo":
            tokenizer = ParallelSentenceFullTokenizer(
                Path("sp_400k_merged.model"),
                # drop_non_word=True, # This param is not needed as doc_coverter in data_loader already does this
                drop_whitespace=True,
                workers_num=16,
                batch_size=args.batch_size * 2,
            )
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
                "show_at_init": False,
                "show_epoch_bar": True,
                "verbose": True,
            }
            data_loader = FasttextModelDataLoader(**data_params, tokenizer=tokenizer)
            logger.mesg(dict_to_str(data_params), indent=2)
        elif args.data_source == "parquet":
            parquet_params = {
                "dataset_root": args.dataset_root,
                "dataset_name": args.dataset_name,
                "parquet_prefix": args.parquet_prefix,
            }
            parquet_reader = VideoTextsParquetReader(**parquet_params)
            logger.mesg(dict_to_str(parquet_params), indent=2)
            data_params = {
                "column": "tokens",
                "max_batch": args.max_batch,
                "batch_size": args.batch_size,
                "max_rows": args.max_rows,
                "max_tables": args.max_tables,
                "show_at_init": False,
                "show_epoch_bar": True,
                "verbose": True,
            }
            data_loader = FasttextModelParquetDataLoader(
                **data_params, parquet_reader=parquet_reader
            )
            logger.mesg(dict_to_str(data_params), indent=2)
        else:
            raise ValueError(f"Unknown data source: {args.data_source}")
        trainer.init_data_loader(data_loader)
        trainer.build_vocab()
        trainer.train()
        if args.data_source == "mongo":
            data_loader.tokenizer.terminate()

    trainer.test(TEST_KEYWORDS, tokenizer=None, restrict_vocab=150000)

    timer.__exit__(None, None, None)

    # python -m models.fasttext.train
    # python -m models.fasttext.train -t

    # python -m models.fasttext.train -ep 3 -m fasttext_tid_17_ep_3
    # python -m models.fasttext.train -m fasttext_tid_17_ep_3_parallel -ep 3 -vf "video_texts_freq_all.csv" -vm 1200000
    # python -m models.fasttext.train -m fasttext_tid_17_ep_2_parallel -ep 2 -mc 20

    # python -m models.fasttext.train -m fasttext_tid_17_ep_4 -ep 4 -dn "video_texts_tid_17" -mc 20 -bs 100000
    # python -m models.fasttext.train -m fasttext_tid_all -ep 1 -dn "video_texts_tid_all" -mc 20 -bs 20000

    # python -m models.fasttext.train -m fasttext_tid_all -ep 1 -dn "video_texts_tid_all" -mc 20 -bs 20000
    # python -m models.fasttext.train -m fasttext_tid_all_mc_50 -ep 1 -dn "video_texts_tid_all" -mc 50 -bs 20000
    # python -m models.fasttext.train -m fasttext_tid_all_mv_60w -ep 1 -dn "video_texts_tid_all" -mv 600000 -bs 20000
    # python -m models.fasttext.train -m fasttext_tid_all_mv_60w_vs_128 -dn "video_texts_tid_all" -bs 20000 -ep 1 -mv 600000 -vs 128
    # python -m models.fasttext.train -m fasttext_tid_all_mv_30w -dn "video_texts_tid_all" -bs 20000 -ep 1 -mv 300000
    # python -m models.fasttext.train -m fasttext_tid_all_mv_30w_vs_384 -dn "video_texts_tid_all" -bs 20000 -ep 1 -mv 300000 -vs 384

    # python -m models.fasttext.train -m fasttext_tid_all_vf_mv_30w_vs_384 -ep 1 -dn "video_texts_tid_all" -vf "video_texts_freq_all.csv" -bs 20000 -mv 300000 -vm 1000000
