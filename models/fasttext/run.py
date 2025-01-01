import argparse
import importlib
import numpy as np
import Pyro5.api
import Pyro5.core
import Pyro5.server
import sys

from gensim.models import FastText, KeyedVectors
from pathlib import Path

from tclogger import logger, logstr, dict_to_str, brk, brp, Runtimer
from typing import Union, Literal

import models.fasttext.test
from configs.envs import PYRO_ENVS, FASTTEXT_CKPT_ROOT
from models.fasttext.preprocess import FasttextModelPreprocessor
from models.fasttext.preprocess import FasttextModelFrequenizer


PYRO_NS = "fasttext_model_runner"


@Pyro5.server.behavior(instance_mode="single")
class FasttextModelRunner:
    def __init__(
        self,
        model_prefix: Union[str, Path] = "fasttext",
        frequenizer: FasttextModelFrequenizer = None,
        preprocessor: FasttextModelPreprocessor = None,
        restrict_vocab: int = 150000,
        vector_weighted: bool = False,
        verbose: bool = False,
    ):
        self.model_prefix = model_prefix
        self.frequenizer = frequenizer
        self.preprocessor = preprocessor or FasttextModelPreprocessor()
        self.restrict_vocab = restrict_vocab
        self.vector_weighted = vector_weighted
        self.verbose = verbose

    @Pyro5.server.expose
    def list_models(self) -> dict:
        logger.file(f"> List local models:", verbose=self.verbose)
        model_files = list(FASTTEXT_CKPT_ROOT.glob(f"*.model"))
        model_files.sort()
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
        logger.file(dict_to_str(model_info_dict), verbose=self.verbose)
        return model_info_dict

    def load_model(self):
        self.model_path = FASTTEXT_CKPT_ROOT / f"{self.model_prefix}.model"
        if self.verbose:
            logger.note(f"> Loading model:")
            logger.mesg(f"  * {self.model_path}")
        self.model = FastText.load(str(self.model_path))
        model_info = {
            "vector_size": self.model.vector_size,
            "restrict_vocab": self.restrict_vocab,
            "max_final_vocab": self.model.max_final_vocab,
            "window": self.model.window,
            "wv.min_n": self.model.wv.min_n,
            "wv.max_n": self.model.wv.max_n,
            "vector_weighted": self.vector_weighted,
        }
        if self.verbose:
            logger.file(dict_to_str(model_info))

    def is_in_vocab(self, word: str):
        return word in self.model.wv.key_to_index

    def preprocess(self, words: Union[str, list[str]]) -> list[str]:
        return self.preprocessor.preprocess(words)

    @Pyro5.server.expose
    def wv_func(self, func: str, *args, **kwargs):
        if self.verbose:
            logger.note(f"> model.wv.{func}:")
            logger.mesg(f"  * args  : {args}")
            logger.mesg(f"  * kwargs: {kwargs}")
        return getattr(self.model.wv, func)(*args, **kwargs)

    @Pyro5.server.expose
    def max_pool_vectors(self, vectors: list[np.ndarray]) -> np.ndarray:
        pooled_vector = np.max(vectors, axis=0)
        return pooled_vector / np.linalg.norm(pooled_vector)

    @Pyro5.server.expose
    def calc_vector(
        self,
        word: Union[str, list[str]],
        ignore_duplicates: bool = True,
        weight_func: Literal["max", "mean", "freq"] = "max",
    ) -> np.ndarray:
        pwords = self.preprocess(word)
        if ignore_duplicates:
            pwords = list(set(pwords))
        if weight_func == "max":
            pword_vectors = [
                self.model.wv.get_vector(pword, norm=True) for pword in pwords
            ]
            return self.max_pool_vectors(pword_vectors)
        elif weight_func in ["freq", "mean"]:
            if weight_func == "freq" and self.frequenizer:
                weights = self.frequenizer.calc_weights_of_tokens(pwords)
            else:
                weights = None
            return self.model.wv.get_mean_vector(
                pwords, weights=weights, pre_normalize=True, post_normalize=False
            )
        else:
            raise ValueError(f"× Invalid weight_func: {weight_func}")

    @Pyro5.server.expose
    def calc_query_vector(self, word: Union[str, list[str]]) -> np.ndarray:
        return self.calc_vector(word, weight_func="freq")

    @Pyro5.server.expose
    def calc_sample_vector(self, word: Union[str, list[str]]) -> np.ndarray:
        return self.calc_vector(word, weight_func="mean")

    @Pyro5.server.expose
    def vector_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return sim**2

    @Pyro5.server.expose
    def word_similarity(
        self, word1: Union[str, list[str]], word2: Union[str, list[str]]
    ) -> float:
        vec1 = self.calc_query_vector(word1)
        vec2 = self.calc_sample_vector(word2)
        return self.vector_similarity(vec1, vec2)

    @Pyro5.server.expose
    def words_similarities(
        self,
        word1: Union[str, list[str]],
        words: Union[list[str], list[list[str]]],
        sort: bool = True,
    ) -> list[tuple[list[str], float]]:
        vec1 = self.calc_query_vector(word1)
        vecs = [self.calc_sample_vector(row) for row in words]
        scores = [self.vector_similarity(vec1, vec) for vec in vecs]
        res = [(row, score) for row, score in zip(words, scores)]
        if sort:
            res.sort(key=lambda x: x[1], reverse=True)
        return res

    @Pyro5.server.expose
    def most_similar_vocab(
        self,
        positive: list = None,
        negative: list = None,
        topn: int = 10,
        restrict_vocab: int = None,
    ):
        results = self.model.wv.most_similar(
            positive=self.preprocess(positive),
            negative=self.preprocess(negative),
            topn=topn,
            restrict_vocab=restrict_vocab or self.restrict_vocab,
        )
        return results

    @Pyro5.server.expose
    def test(self):
        importlib.reload(models.fasttext.test)
        from models.fasttext.test import TEST_KEYWORDS

        logger.note(f"> Testing:")
        for word in TEST_KEYWORDS:
            word = self.preprocess(word)
            logger.mesg(f"  * [{logstr.file(word)}]:")
            results = self.most_similar_vocab(positive=word, topn=10)[:6]
            for result in results:
                res_word, res_score = result
                logger.success(f"    * {res_score:>.4f}: {res_word}")
        logger.file(f"* {self.model_prefix}")

    @Pyro5.server.expose
    def test_pair_similarities(self):
        importlib.reload(models.fasttext.test)
        from models.fasttext.test import TEST_PAIRS

        def sim2str(num: float, round_digits: int = 2, br: bool = True) -> str:
            num_str = f"{num:.{round_digits}f}"
            if br:
                num_str = brp(num_str)
            return logstr.mesg(num_str)

        def weight2str(num: float, round_digits: int = 2, br: bool = True) -> str:
            num_str = f"{num:.{round_digits}f}"
            if br:
                num_str = brk(num_str)
            return logstr.file(num_str)

        def simweight2str(
            sim: float, weight: float, round_digits: int = 1, br: bool = True
        ) -> str:
            weight_str = weight2str(weight, round_digits, br=False)
            sim_str = sim2str(sim, round_digits, br=False)
            num_str = f"{weight_str}*{sim_str}"
            if br:
                num_str = brp(num_str)
            return logstr.mesg(num_str)

        logger.note(f"> Testing (similarity):")
        for word1, words in TEST_PAIRS:
            pword1 = self.preprocess(word1)
            if self.frequenizer:
                pword1_weights = self.frequenizer.calc_weights_of_tokens(pword1)
            else:
                pword1_weights = [1.0] * len(pword1)
            pword1_str_list = [
                f"{token}{weight2str(weight)}"
                for token, weight in zip(pword1, pword1_weights)
            ]
            pword1_str = " ".join(pword1_str_list)
            logger.note(f"  * [{pword1_str}]:")
            pwords = [self.preprocess(word) for word in words]
            results = self.words_similarities(pword1, pwords)
            for result in results:
                res_row, res_score = result
                sims = [self.word_similarity(pword1, pword) for pword in res_row]
                if self.frequenizer:
                    token_weights = self.frequenizer.calc_weights_of_tokens(res_row)
                else:
                    token_weights = [1.0] * len(res_row)
                tokens_str_list = [
                    f"{token}{simweight2str(sim, weight)}"
                    for token, sim, weight in zip(res_row, sims, token_weights)
                ]
                tokens_str = " ".join(tokens_str_list)
                logger.success(f"    * {res_score:>.4f}: [{tokens_str}]")

    def test_func(self):
        importlib.reload(models.fasttext.test)
        from models.fasttext.test import TEST_KEYWORDS

        logger.note(f"> Testing (func):")
        for word in TEST_KEYWORDS:
            word = self.preprocess(word)
            logger.mesg(f"  * [{logstr.file(word)}]:")
            results = self.wv_func("most_similar", positive=word, topn=10)[:6]
            for result in results:
                res_word, res_score = result
                logger.success(f"    * {res_score:>.4f}: {res_word}")
        logger.file(f"* {self.model_prefix}")


class FasttextModelRunnerServer:
    def __init__(
        self,
        nameserver: str = PYRO_NS,
        host: str = PYRO_ENVS["host"],
        port: int = PYRO_ENVS["port"],
        verbose: bool = False,
    ):
        self.nameserver = nameserver
        self.host = host
        self.port = port
        self.verbose = verbose

    def serve(self, runner: FasttextModelRunner):
        if self.verbose:
            logger.note(f"> Running as remote server:")
            logger.mesg(f"  * {self.host}:{self.port}")
        # Pyro5.server.serve({runner: self.nameserver}, host=self.host, port=self.port)
        daemon = Pyro5.server.Daemon(host=self.host, port=self.port)
        self.uri = daemon.register(runner, objectId=self.nameserver, weak=True)
        logger.file(f"  * {self.uri}", verbose=self.verbose)
        daemon.requestLoop()


class FasttextModelRunnerClient:
    def __init__(
        self,
        nameserver: str = PYRO_NS,
        host: str = PYRO_ENVS["host"],
        port: int = PYRO_ENVS["port"],
        verbose: bool = False,
    ):
        self.verbose = verbose
        self.uri = f"PYRO:{nameserver}@{host}:{port}"
        self.runner = Pyro5.api.Proxy(self.uri)


class FasttextModelRunnerArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("-m", "--model-prefix", type=str, default="fasttext")
        self.add_argument("-k", "--tokenizer-prefix", type=str, default="sp_merged")
        self.add_argument(
            "-q", "--token-freq-prefix", type=str, default="video_texts_freq_all"
        )
        self.add_argument("-w", "--vector-weighted", action="store_true")
        self.add_argument("-v", "--restrict-vocab", type=int, default=150000)
        self.add_argument("-minw", "--min-weight", type=float, default=0.001)
        self.add_argument("-maxw", "--max-weight", type=float, default=1.0)
        self.add_argument("-l", "--list-models", action="store_true")
        self.add_argument("-n", "--nameserver", type=str, default=PYRO_NS)
        self.add_argument("-s", "--host", type=str, default=PYRO_ENVS["host"])
        self.add_argument("-p", "--port", type=int, default=PYRO_ENVS["port"])
        self.add_argument("-r", "--remote", action="store_true")
        self.add_argument("-t", "--test", action="store_true")
        self.add_argument("-tc", "--test-client", action="store_true")

    def parse_args(self):
        self.args, self.unknown_args = self.parse_known_args(sys.argv[1:])
        return self.args


if __name__ == "__main__":
    parser = FasttextModelRunnerArgParser()
    args = parser.parse_args()

    if not args.test_client:
        if args.vector_weighted:
            frequenizer = FasttextModelFrequenizer(
                token_freq_prefix=args.token_freq_prefix,
                min_weight=args.min_weight,
                max_weight=args.max_weight,
                verbose=True,
            )
        else:
            frequenizer = None
        if not args.list_models:
            preprocessor = FasttextModelPreprocessor(
                tokenizer_prefix=args.tokenizer_prefix, verbose=True
            )
        else:
            preprocessor = None
        runner = FasttextModelRunner(
            model_prefix=args.model_prefix,
            frequenizer=frequenizer,
            preprocessor=preprocessor,
            restrict_vocab=args.restrict_vocab,
            vector_weighted=args.vector_weighted,
            verbose=True,
        )

    if args.list_models:
        runner.list_models()
    else:
        remote_args = {
            "nameserver": args.nameserver,
            "host": args.host,
            "port": args.port,
        }

        if not args.test_client:
            with Runtimer() as timer:
                runner.load_model()

        if args.test:
            # runner.test()
            runner.test_func()
        elif args.test_client:
            client = FasttextModelRunnerClient(**remote_args)
            # client.runner.test()
            client.runner.test_pair_similarities()
            # res = client.runner.list_models()
            # logger.success(dict_to_str(res))
        elif args.remote:
            runner.verbose = False
            server = FasttextModelRunnerServer(**remote_args, verbose=True)
            server.serve(runner)
        else:
            raise ValueError("× Invalid running mode!")

    # List models:
    # python -m models.fasttext.run -l

    # Test models:
    # python -m models.fasttext.run -t -m fasttext_tid_all_mv_60w -v 150000
    # python -m models.fasttext.run -t -m fasttext_tid_all_mv_30w -v 150000

    # Run remote server:
    # python -m models.fasttext.run -r -m fasttext_tid_all_mv_30w -v 150000
    # python -m models.fasttext.run -r -m fasttext_tid_all_mv_30w -v 150000 -w

    # Test remote client:
    # python -m models.fasttext.run -tc
