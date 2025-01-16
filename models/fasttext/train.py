import argparse
import numpy as np
import pandas as pd
import sys

from collections import defaultdict
from gensim.models import FastText, KeyedVectors
from gensim.models.doc2vec import Doc2Vec
from pathlib import Path
from tclogger import Runtimer, logger, logstr, dict_to_str, brk, attrs_to_dict
from typing import Union, Literal

from configs.envs import FASTTEXT_CKPT_ROOT, SP_MERGED_MODEL_PATH, TOKEN_FREQS_ROOT
from datasets.videos.parquet import VideoTextsParquetReader
from datasets.args import DATA_LOADER_ARG_PARSER
from models.fasttext.data import FasttextDataLoader, FasttextParquetDataLoader
from models.fasttext.vocab import FasttextVocabLoader
from models.fasttext.test import TEST_KEYWORDS, TEST_PAIRS
from models.sentencepiece.tokenizer import SentenceFullTokenizer
from models.sentencepiece.tokenizer_parallel import ParallelSentenceFullTokenizer


class FasttextModelTrainer:
    def __init__(
        self,
        model_prefix: Union[str, Path] = "fasttext",
        train_stage: Literal["pre_train", "post_train", "test"] = "pre_train",
        post_train_model_prefix: Union[str, Path] = None,
        epochs: int = 5,
        bucket: int = 2000000,
        hs: int = 0,
        min_alpha: float = 0.0001,
        min_count: int = 5,
        min_n: int = 3,
        max_n: int = 6,
        max_final_vocab: int = None,
        sample: float = 1e-3,
        seed: int = 1,
        shrink_windows: bool = False,
        sg: int = 0,
        vector_size: int = 128,
        window: int = 5,
        workers: int = 8,
        skip_trained: bool = True,
        use_local_model: bool = True,
        use_kv: bool = False,
        vocab_loader: FasttextVocabLoader = None,
        vocab_load_format: Literal["csv", "pickle"] = "csv",
        sample_ratio: float = 1.0,
    ):
        self.model_prefix = model_prefix
        self.train_stage = train_stage
        self.skip_trained = skip_trained
        self.use_local_model = use_local_model
        self.use_kv = use_kv
        self.vocab_loader = vocab_loader
        self.vocab_load_format = vocab_load_format
        self.sample_ratio = sample_ratio

        self.post_train_model_prefix = post_train_model_prefix
        self.model_path = FASTTEXT_CKPT_ROOT / f"{model_prefix}.model"
        self.kv_path = self.model_path.with_suffix(".kv")
        self.model = None
        self.is_model_trained = False

        self.train_params = {
            "epochs": epochs,
            "bucket": bucket,
            "hs": hs,
            "min_alpha": min_alpha,
            "min_count": min_count,
            "min_n": min_n,
            "max_n": max_n,
            "max_final_vocab": max_final_vocab,
            "sample": sample,
            "seed": seed,
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
        data_loader: Union[FasttextDataLoader, FasttextParquetDataLoader],
    ):
        self.data_loader = data_loader

    def prepare_vocab(self):
        # create model.wv with index_to_key, key_to_index, and set 'count' for each word
        self.model.wv.index_to_key = []
        self.model.wv.key_to_index = {}
        for word, v in self.retain_vocab.items():
            self.model.wv.key_to_index[word] = len(self.model.wv.index_to_key)
            self.model.wv.index_to_key.append(word)
        for word in self.model.wv.index_to_key:
            self.model.wv.set_vecattr(word, "count", self.retain_vocab[word])

        # calculate threadhold_count of sample, and set sample_int for each word
        retain_total = sum(self.retain_vocab.values())

        # 1. `sample_ratio` is used when vocab freqs is not "aligned" with trained corpus,
        #   for example, if vocab freqs is calculated from a large corpus, and the train corpus is a subset of it,
        #   then the sample_ratio should be set close to the corpuse ratio:
        #       train_corpus_count / vocab_freqs_corpus_count
        #   for example, if sample_ratio is about 0.1 ~ 1.0, then this means the count of train corpus is about 5% ~ 100% of the vocab freqs corpus
        # 2. `threshold_count` is inherited from gensim's original naming convention,
        #   which is the threshold of the word frequency to be sampled
        #   the smaller the threshold_count, the more words will be downsampled, and the faster the training speed
        # 3. `downsample_unique` is the count of words that will be downsampled,
        #    next step i would like to enable this (or a new `downsample_vocab_ratio`) as cli arg to enhances flexibility
        threshold_count = self.model.sample * retain_total * self.sample_ratio
        downsample_total, downsample_unique = 0, 0
        for word, v in self.retain_vocab.items():
            word_probability = (np.sqrt(v / threshold_count) + 1) * (
                threshold_count / v
            )
            if word_probability < 1.0:
                downsample_unique += 1
                downsample_total += word_probability * v
            else:
                word_probability = 1.0
                downsample_total += v
            self.model.wv.set_vecattr(
                word, "sample_int", np.uint32(word_probability * (2**32 - 1))
            )

        # delete raw_vocab
        self.model.raw_vocab = defaultdict(int)

        # sort model.wv vocab , create_binary_tree (if hs),  make_cum_table (if negative)
        self.model.wv.sort_by_descending_frequency()
        if self.model.hs:
            # add info about each word's Huffman encoding
            self.model.create_binary_tree()
        if self.model.negative:
            # build the table for drawing random words (for negative sampling)
            self.model.make_cum_table()

        prepare_vocab_info = {
            "hs": self.model.hs,
            "negative": self.model.negative,
            "sample": self.model.sample,
            "sample_ratio": self.sample_ratio,
            "seed": self.model.seed,
            "retain_total": retain_total,
            "threshold_count": threshold_count,
            "downsample_unique": downsample_unique,
            "downsample_total": downsample_total,
        }
        logger.mesg(dict_to_str(prepare_vocab_info), indent=2)

    def dump_wv_info(self):
        # dump debug vocab info to csv
        logger.note("> Dumping wv info:")
        wv_infos = [
            {
                "token": word,
                "index": self.model.wv.key_to_index[word],
                "count": self.model.wv.get_vecattr(word, "count"),
                "sample_int": self.model.wv.get_vecattr(word, "sample_int"),
            }
            for word in self.model.wv.key_to_index
        ]
        df = pd.DataFrame(wv_infos)
        wv_vocab_path = TOKEN_FREQS_ROOT / f"{self.model_prefix}.wv"
        df.to_csv(wv_vocab_path, index=False)
        logger.file(f"  * [{wv_vocab_path}]")

    def load_vocab(self):
        # load vocab from csv or pickle
        if self.vocab_load_format == "csv":
            vocab_dict = self.vocab_loader.load_vocab_from_csv(
                return_format="dict", order="sort", sort_first="term_freq"
            )
        else:
            vocab_dict = self.vocab_loader.load_vocab_from_pickle()

        token_freqs = vocab_dict["term_freqs"]

        # original gensim's implementation set limit of term_freq,
        #   where vocab size smaller than max_final_vocab
        sorted_tokens = sorted(
            token_freqs.keys(), key=lambda word: token_freqs[word], reverse=True
        )
        calc_min_count = token_freqs[sorted_tokens[self.model.max_final_vocab]] + 1
        calc_min_count = max(calc_min_count, self.model.min_count)
        # if would like the vocab size to be exactly max_final_vocab,
        #   comment the above, and uncomment the following line:
        # calc_min_count = self.model.min_count

        # get retain_vocab with filtering term_freqs by min_count and max_final_vocab
        self.retain_vocab = {}
        retain_vocab_count = 0
        for word, v in token_freqs.items():
            if retain_vocab_count >= self.model.max_final_vocab:
                break
            if v >= calc_min_count:
                self.retain_vocab[word] = v
                retain_vocab_count += 1

        # get corpus_count and total_words
        total_words, corpus_count = self.data_loader.get_count()
        # corpus_count = self.data_loader.get_corpus_count()
        # total_words = sum(token_freqs.values())
        self.model.corpus_count = corpus_count
        self.model.corpus_total_words = total_words
        count_info = {
            "total_words": total_words,
            "corpus_count": corpus_count,
        }
        logger.mesg(dict_to_str(count_info), indent=2)

        # prepare vocab and weights
        logger.note("> Preparing vocab and weights:")
        self.prepare_vocab()
        # TODO: need to adapt for post_train
        self.model.prepare_weights(update=False)
        prepare_weights_info = {
            "model_attrs": attrs_to_dict(self.model),
            # "wv_attrs": attrs_to_dict(self.model.wv),
            "top_wv_words": list(self.model.wv.key_to_index.keys())[:5],
        }
        logger.mesg(dict_to_str(prepare_weights_info), indent=2)

    def init_vocab(self):
        logger.note("> Initiating vocab:")
        if self.is_model_trained and self.skip_trained:
            logger.file("  * model already trained, skip init vocab")
            return

        if self.vocab_loader:
            with logger.temp_indent(2):
                self.load_vocab()
            logger.success("  ✓ vocab loaded")
        else:
            self.model.build_vocab(corpus_iterable=self.data_loader)
            logger.success("  ✓ vocab built")

        self.dump_wv_info()

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
        logger.note("> Saving model:")
        if not self.model_path.parent.exists():
            self.model_path.parent.mkdir(parents=True, exist_ok=True)

        if self.train_stage == "post_train":
            if self.post_train_model_prefix:
                save_model_prefix = self.post_train_model_prefix
            else:
                save_model_prefix = f"{self.model_prefix}_post_train"
            model_save_path = FASTTEXT_CKPT_ROOT / f"{save_model_prefix}.model"
        else:
            model_save_path = self.model_path

        self.model.save(str(model_save_path))
        logger.success(f"  * [{model_save_path}]")

        if self.use_kv:
            kv_save_path = model_save_path.with_suffix(".kv")
            self.model.wv.save(str(kv_save_path))
            logger.success(f"  * [{kv_save_path}]")

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
                # total_words=self.model.corpus_total_words,
            )
            logger.success(f"  ✓ model trained")

            self.save_model()

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
            "-ms",
            "--model-class",
            type=str,
            choices=["fasttext", "doc2vec"],
            default="fasttext",
        )
        self.add_argument(
            "-ds",
            "--data-source",
            type=str,
            choices=["mongo", "parquet"],
            default="parquet",
        )
        self.add_argument(
            "-ts",
            "--train-stage",
            type=str,
            choices=["pre_train", "post_train", "test"],
            default="pre_train",
        )
        self.add_argument("-dy", "--dry-run", action="store_true")
        self.add_argument("-pm", "--post-train-model-prefix", type=str, default=None)
        self.add_argument("-ep", "--epochs", type=int, default=1)
        self.add_argument("-bk", "--bucket", type=int, default=2000000)
        self.add_argument("-hs", "--hs", type=int, default=0)
        self.add_argument("-ma", "--min-alpha", type=float, default=0.0001)
        self.add_argument("-mc", "--min-count", type=int, default=20)
        self.add_argument("-mv", "--max-final-vocab", type=int, default=None)
        self.add_argument("-minn", "--min-n", type=int, default=2)
        self.add_argument("-maxn", "--max-n", type=int, default=6)
        self.add_argument("-sg", "--sg", type=int, default=0)
        self.add_argument("-sw", "--shrink-windows", action="store_true")
        self.add_argument("-vs", "--vector-size", type=int, default=256)
        self.add_argument("-vf", "--vocab-file", type=str, default=None)
        self.add_argument(
            "-vl",
            "--vocab-load-format",
            type=str,
            choices=["csv", "pickle"],
            default="csv",
        )
        self.add_argument("-vm", "--vocab-max-count", type=int, default=None)
        self.add_argument("-sr", "--sample-ratio", type=float, default=1.0)
        self.add_argument("-wd", "--window", type=int, default=5)
        self.add_argument("-wk", "--workers", type=int, default=32)
        self.add_argument("-st", "--skip-trained", action="store_true")
        self.add_argument("-lm", "--use-local-model", action="store_true")
        self.add_argument("-kv", "--use-kv", action="store_true")
        self.add_argument("-tm", "--tokenizer-model", type=str, default=None)

    def parse_args(self):
        self.args, self.unknown_args = self.parse_known_args(sys.argv[1:])
        return self.args


class Doc2VecArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("-dm", "--dm", type=int, default=1)
        self.add_argument("-dmm", "--dm-mean", type=int, default=None)
        self.add_argument("-dmc", "--dm-concat", type=int, default=0)
        self.add_argument("-dmt", "--dm-tag-count", type=int, default=1)
        self.add_argument("-dbw", "--dbow-words", type=int, default=0)

    def parse_args(self):
        self.args, self.unknown_args = self.parse_known_args(sys.argv[1:])
        return self.args


def main(args: argparse.Namespace):
    if args.train_stage == "pre_train":
        args.skip_trained = False
        args.use_local_model = False
    elif args.train_stage == "post_train":
        args.skip_trained = False
        args.use_local_model = True
    elif args.train_stage == "test":
        args.skip_trained = True
        args.use_local_model = True
    else:
        raise ValueError(f"× Invalid train_stage: {args.train_stage}")

    if args.vocab_file:
        vocab_loader = FasttextVocabLoader(
            args.vocab_file,
            vocab_max_count=args.vocab_max_count,
            token_format="parquet",
            verbose=True,
        )
    else:
        vocab_loader = None

    train_params = {
        "model_prefix": args.model_prefix,
        "epochs": args.epochs,
        "bucket": args.bucket,
        "hs": args.hs,
        "min_alpha": args.min_alpha,
        "min_count": args.min_count,
        "min_n": args.min_n,
        "max_n": args.max_n,
        "max_final_vocab": args.max_final_vocab,
        "shrink_windows": args.shrink_windows,
        "sg": args.sg,
        "vector_size": args.vector_size,
        "window": args.window,
        "workers": args.workers,
        "skip_trained": args.skip_trained,
        "use_local_model": args.use_local_model,
        "use_kv": args.use_kv,
        "vocab_loader": vocab_loader,
        "vocab_load_format": args.vocab_load_format,
        "sample_ratio": args.sample_ratio,
    }

    if args.model_class == "fasttext":
        trainer = FasttextModelTrainer(**train_params)
    elif args.model_class == "doc2vec":
        doc2vec_params = {
            "dm": args.dm,
            "dm_mean": args.dm_mean,
            "dm_concat": args.dm_concat,
            "dm_tag_count": args.dm_tag_count,
            "dbow_words": args.dbow_words,
        }
        train_params.update(doc2vec_params)
        trainer = Doc2VecModelTrainer(**train_params)
    else:
        raise ValueError(f"× Invalid model_class: {args.model_class}")

    if args.train_stage in ["pre_train", "post_train"]:
        data_source_str = logstr.mesg(brk(args.data_source))
        logger.note(f"> Initiating data loader: {data_source_str}")
        if args.data_source == "mongo":
            tokenizer = ParallelSentenceFullTokenizer(
                SP_MERGED_MODEL_PATH,
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
            data_loader = FasttextDataLoader(**data_params, tokenizer=tokenizer)
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
                "show_at_init": False,
                "show_epoch_bar": True,
                "verbose": True,
            }
            data_loader = FasttextParquetDataLoader(
                **data_params, parquet_reader=parquet_reader
            )
            data_loader.model_class = args.model_class
            logger.mesg(dict_to_str(data_params), indent=2)
        else:
            raise ValueError(f"× Invalid data_source: {args.data_source}")
        trainer.init_data_loader(data_loader)
        trainer.init_vocab()
        if not args.dry_run:
            trainer.train()
        else:
            trainer.save_model()
        if args.data_source == "mongo":
            data_loader.tokenizer.terminate()

    if args.dry_run:
        return

    if args.model_class == "fasttext":
        trainer.test(tokenizer=None, restrict_vocab=150000)
    elif args.model_class == "doc2vec":
        tokenizer = SentenceFullTokenizer(
            SP_MERGED_MODEL_PATH, drop_non_word=True, drop_whitespace=True
        )
        trainer.test(tokenizer=tokenizer)
    else:
        raise ValueError(f"× Invalid model_class: {args.model_class}")


if __name__ == "__main__":
    arg_parser = DATA_LOADER_ARG_PARSER
    arg_parser.add_parser_class(ModelTrainerArgParser, Doc2VecArgParser)
    args = arg_parser.parse_args()
    with Runtimer() as timer:
        main(args)

    # python -m models.fasttext.train -m fasttext_other_game -ep 1 -dr "parquets" -dn "video_texts_other_game" -vf video_texts_other_game_nt -bs 20000 -mv 300000
    # python -m models.fasttext.train -m fasttext_tech_sports_vf -ep 1 -dr "parquets" -dn "video_texts_tech_sports" -vf "video_texts_tech_sports_nt" -bs 20000 -mv 500000

    # python -m models.fasttext.train -m fasttext_tid_all_mv_30w -ts test
    # python -m models.fasttext.train -m fasttext_music_dance -ts test

    # python -m models.fasttext.train -m fasttext_other_game_vf_merged_csv -ep 1 -dr "parquets" -dn "video_texts_other_game" -vf "merged_video_texts" -vl csv -bs 20000 -mv 900000 -bk 5000000 -sr 0.1
