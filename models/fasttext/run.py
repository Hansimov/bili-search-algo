import argparse
import importlib
import numpy as np
import Pyro5.api
import Pyro5.server
import sys

from functools import partial
from gensim.models import FastText, KeyedVectors
from pathlib import Path
from tclogger import logger, logstr, dict_to_str, brk, brp, Runtimer
from typing import Union, Literal

import models.fasttext.test
from configs.envs import PYRO_ENVS, FASTTEXT_CKPT_ROOT
from configs.envs import SP_MERGED_MODEL_PREFIX, TOKEN_FREQ_PREFIX
from configs.envs import FASTTEXT_MERGED_MODEL_PREFIX
from configs.envs import FASTTEXT_MERGED_MODEL_DIMENSION
from models.fasttext.preprocess import FasttextModelPreprocessor
from models.fasttext.preprocess import FasttextModelFrequenizer
from models.fasttext.preprocess import TokenScoreFuncType, FreqScoreFuncType
from models.vectors.calcs import dot_sim
from models.vectors.forms import trunc, stretch_copy, stretch_shift_add, downsample
from models.vectors.forms import calc_padded_downsampled_cols
from models.vectors.structs import replace_items_with_sub_list_and_idxs
from workers.pyros.register import register_pyro_apis

register_pyro_apis()


@Pyro5.server.behavior(instance_mode="single")
class FasttextModelRunner:
    def __init__(
        self,
        model_prefix: Union[str, Path] = FASTTEXT_MERGED_MODEL_PREFIX,
        frequenizer: FasttextModelFrequenizer = None,
        preprocessor: FasttextModelPreprocessor = None,
        run_mode: Literal["local", "remote"] = "local",
        restrict_vocab: int = 150000,
        vector_weighted: bool = False,
        verbose: bool = False,
    ):
        self.model_prefix = model_prefix
        self.frequenizer = frequenizer
        self.preprocessor = preprocessor or FasttextModelPreprocessor()
        self.run_mode = run_mode
        self.restrict_vocab = restrict_vocab
        self.vector_weighted = vector_weighted
        self.verbose = verbose

    @Pyro5.server.expose
    def list_models(self) -> dict:
        logger.file(f"> List local models:", verbose=self.verbose)
        model_files = list(FASTTEXT_CKPT_ROOT.glob(f"*.model"))
        model_files.sort()
        model_info_dict = {}
        logger.file(f"* {FASTTEXT_CKPT_ROOT}")
        logger.note(f"* name: (model_size_in_mb, vocab_size_in_mb)")
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

    @Pyro5.server.expose
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
    def preprocessor_func(self, func: str, *args, **kwargs):
        return getattr(self.preprocessor, func)(*args, **kwargs)

    @Pyro5.server.expose
    def frequenizer_func(self, func: str, *args, **kwargs):
        return getattr(self.frequenizer, func)(*args, **kwargs)

    @Pyro5.server.expose
    def calc_weight_of_token(
        self,
        word: str,
        score_func: TokenScoreFuncType = "one",
        base: Union[int, float] = None,
        min_weight: float = None,
        max_weight: float = None,
    ) -> float:
        return self.frequenizer.calc_weight_of_token(
            word,
            score_func=score_func,
            base=base,
            min_weight=min_weight,
            max_weight=max_weight,
        )

    @Pyro5.server.expose
    def calc_weight_of_words(
        self,
        word: Union[str, list[str]],
        score_func: TokenScoreFuncType = "one",
        base: Union[int, float] = None,
        min_weight: float = None,
        max_weight: float = None,
    ):
        pwords = self.preprocess(word)
        return [
            self.calc_weight_of_token(
                word,
                score_func=score_func,
                base=base,
                min_weight=min_weight,
                max_weight=max_weight,
            )
            for word in pwords
        ]

    @Pyro5.server.expose
    def calc_tokens_and_weights_of_sentence(
        self,
        sentence: str,
        max_char_len: int = None,
        score_func: TokenScoreFuncType = "pos",
        base: Union[int, float] = None,
        min_weight: float = None,
        max_weight: float = None,
    ) -> dict:
        res = {}
        tokens = self.preprocess(sentence)

        weight_params = {
            "score_func": score_func,
            "base": base,
            "min_weight": min_weight,
            "max_weight": max_weight,
        }
        tokens_weights = self.frequenizer.calc_weights_of_tokens(
            tokens, **weight_params
        )
        tokens_freqs = self.frequenizer.get_tokens_freqs(tokens)

        use_pos = score_func == "pos" and self.frequenizer.token_pos_dict

        if use_pos:
            tokens_pos = [
                self.frequenizer.token_pos_dict.get(token, None) for token in tokens
            ]
        else:
            tokens_pos = None

        if max_char_len:
            tokens_splits, splits_idxs = self.preprocessor.get_tokens_splits_and_idxs(
                tokens, max_char_len=max_char_len
            )

            tokens_maxlen = replace_items_with_sub_list_and_idxs(
                tokens, tokens_splits, splits_idxs
            )

            tokens_splits_weights = [
                self.frequenizer.calc_weights_of_tokens(token_splits, **weight_params)
                for token_splits in tokens_splits
            ]
            tokens_splits_freqs = [
                self.frequenizer.get_tokens_freqs(token_splits)
                for token_splits in tokens_splits
            ]
        else:
            tokens_splits = None
            splits_idxs = None
            tokens_splits_weights = None
            tokens_splits_freqs = None
            tokens_maxlen = None

        if max_char_len and use_pos:
            tokens_splits_pos = [
                [
                    self.frequenizer.token_pos_dict.get(token_split, None)
                    for token_split in token_splits
                ]
                for token_splits in tokens_splits
            ]
            tokens_maxlen_pos = replace_items_with_sub_list_and_idxs(
                tokens,
                tokens_splits_pos,
                splits_idxs,
                func=lambda x: self.frequenizer.token_pos_dict.get(x, None),
            )
            tokens_maxlen_freqs = replace_items_with_sub_list_and_idxs(
                tokens_freqs,
                tokens_splits_freqs,
                splits_idxs,
            )
            tokens_maxlen_weights = replace_items_with_sub_list_and_idxs(
                tokens_weights,
                tokens_splits_weights,
                splits_idxs,
            )
        else:
            tokens_splits_pos = None
            tokens_maxlen_pos = None
            tokens_maxlen_freqs = None
            tokens_maxlen_weights = None

        res = {
            "sentence": sentence,
            "max_char_len": max_char_len,
            "weight_params": weight_params,
            "tokens": tokens,
            "tokens_pos": tokens_pos,
            "tokens_freqs": tokens_freqs,
            "tokens_weights": tokens_weights,
            "splits_idxs": splits_idxs,
            "tokens_splits": tokens_splits,
            "tokens_splits_pos": tokens_splits_pos,
            "tokens_splits_freqs": tokens_splits_freqs,
            "tokens_splits_weights": tokens_splits_weights,
            "tokens_maxlen": tokens_maxlen,
            "tokens_maxlen_pos": tokens_maxlen_pos,
            "tokens_maxlen_freqs": tokens_maxlen_freqs,
            "tokens_maxlen_weights": tokens_maxlen_weights,
        }

        return res

    @Pyro5.server.expose
    def max_pool_vectors(self, vectors: list[np.ndarray]) -> np.ndarray:
        pooled_vector = np.max(vectors, axis=0)
        return pooled_vector / np.linalg.norm(pooled_vector)

    @Pyro5.server.expose
    def get_vector(self, word: str) -> np.ndarray:
        vector = self.model.wv.get_vector(word, norm=True)
        return vector

    @Pyro5.server.expose
    def calc_vector(
        self,
        word: Union[str, list[str]],
        ignore_duplicates: bool = True,
        weight_func: Literal["mean", "sum", "score"] = "mean",
        score_func: TokenScoreFuncType = "one",
        base: Union[int, float] = None,
        min_weight: float = None,
        max_weight: float = None,
        normalize: bool = True,
    ) -> np.ndarray:
        pwords = self.preprocess(word)
        if not pwords:
            return np.zeros(self.model.vector_size)
        if ignore_duplicates:
            pwords = list(set(pwords))
        pword_vectors = [self.model.wv.get_vector(pword, norm=True) for pword in pwords]
        vector = np.zeros(self.model.vector_size)
        if weight_func == "score" and self.frequenizer:
            weight_params = {
                "score_func": score_func,
                "base": base,
                "min_weight": min_weight,
                "max_weight": max_weight,
            }
            weights = np.array(
                self.frequenizer.calc_weights_of_tokens(pwords, **weight_params)
            )
        else:
            weights = np.ones(len(pwords))
        for pword_vector, weight in zip(pword_vectors, weights):
            vector += pword_vector * weight
        if normalize:
            vector /= np.linalg.norm(vector)
        else:
            if weight_func not in ["sum"]:
                weights_abs_sum = np.sum(np.abs(weights))
                if weights_abs_sum > 0:
                    vector /= weights_abs_sum
        return vector

    @Pyro5.server.expose
    def calc_query_vector(self, word: Union[str, list[str]]) -> np.ndarray:
        return self.calc_vector(word, weight_func="score")

    @Pyro5.server.expose
    def calc_sample_vector(self, word: Union[str, list[str]]) -> np.ndarray:
        return self.calc_vector(word, weight_func="sum", normalize=False)

    @Pyro5.server.expose
    def word_similarity(
        self, word1: Union[str, list[str]], word2: Union[str, list[str]]
    ) -> float:
        vec1 = self.calc_query_vector(word1)
        vec2 = self.calc_sample_vector(word2)
        return dot_sim(vec1, vec2) ** 2

    @Pyro5.server.expose
    def calc_pairs_scores(
        self,
        query_words: list[str],
        sample_words_list: list[list[str]],
        level: Literal["word", "sentence"] = "word",
        sort: bool = True,
    ) -> list[tuple[list[str], float, list[float], list[float]]]:
        if level == "sentence":
            query_vector = self.calc_query_vector(query_words)
            sample_vectors = [
                self.calc_sample_vector(sample_words)
                for sample_words in sample_words_list
            ]
            sample_scores = [
                dot_sim(query_vector, sample_vector) ** 2
                for sample_vector in sample_vectors
            ]
            word_scores_of_samples = []
            word_weights_of_samples = []
        elif level == "word":
            query_vector = self.calc_query_vector(query_words)
            sample_scores: list[float] = []
            word_scores_of_samples: list[list[float]] = []
            word_weights_of_samples: list[list[float]] = []
            for sample_words in sample_words_list:
                word_scores = []
                word_weights = []
                sample_score_weight_dict = {}
                for sample_word in sample_words:
                    if sample_word in sample_score_weight_dict:
                        word_score = sample_score_weight_dict[sample_word]["score"]
                        word_weight = sample_score_weight_dict[sample_word]["weight"]
                    else:
                        sample_vector_by_word = self.calc_sample_vector(sample_word)
                        word_score = dot_sim(query_vector, sample_vector_by_word) ** 2
                        word_score = trunc(word_score, trunc_at=0.15, trunc_to=0)
                        if self.frequenizer:
                            word_weight = self.frequenizer.calc_weight_of_token(
                                sample_word
                            )
                        else:
                            word_weight = 1.0
                        sample_score_weight_dict[sample_word] = {
                            "score": word_score,
                            "weight": word_weight,
                        }
                    word_scores.append(word_score)
                    word_weights.append(word_weight)
                sample_score = sum(
                    [
                        score_weight_dict["score"] * score_weight_dict["weight"]
                        for score_weight_dict in sample_score_weight_dict.values()
                    ]
                )
                sample_scores.append(sample_score)
                word_scores_of_samples.append(word_scores)
                word_weights_of_samples.append(word_weights)
        else:
            raise ValueError(f"× Invalid score level: {level}")
        res = list(
            zip(
                sample_words_list,
                sample_scores,
                word_scores_of_samples,
                word_weights_of_samples,
            )
        )
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

        def score2str(num: float, round_digits: int = 2, br: bool = True) -> str:
            num_str = f"{num:.{round_digits}f}"
            if br:
                num_str = brp(num_str)
            if num < 0.25:
                str_func = logstr.warn
            elif num < 0.5:
                str_func = logstr.mesg
            else:
                str_func = logstr.hint
            return str_func(num_str)

        def weight2str(num: float, round_digits: int = 2, br: bool = True) -> str:
            num_str = f"{num:.{round_digits}f}"
            if br:
                num_str = brk(num_str)
            return logstr.file(num_str)

        def token2str(token: str, score: float, weight: float) -> str:
            if score < 0.25 or weight < 0.25:
                str_func = logstr.warn
            elif score < 0.5 or weight < 0.5:
                str_func = logstr.mesg
            else:
                str_func = logstr.success
            return str_func(token)

        def scoreweight2str(
            score: float, weight: float, round_digits: int = 1, br: bool = True
        ) -> str:
            score_str = score2str(score, round_digits, br=False)
            weight_str = weight2str(weight, round_digits, br=False)
            num_str = f"{score_str}*{weight_str}"
            if br:
                num_str = brp(num_str)
            return logstr.line(num_str)

        logger.note(f"> Testing (similarity):")
        for query, samples in TEST_PAIRS:
            query_tokens = self.preprocess(query)
            if self.frequenizer:
                query_token_weights = self.frequenizer.calc_weights_of_tokens(
                    query_tokens
                )
            else:
                query_token_weights = [1.0] * len(query_tokens)
            query_tokens_str_list = [
                f"{token}{weight2str(weight)}"
                for token, weight in zip(query_tokens, query_token_weights)
            ]
            query_tokens_str = " ".join(query_tokens_str_list)
            logger.note(f"  * [{query_tokens_str}]:")
            sample_tokens_list = [self.preprocess(sample) for sample in samples]
            results = self.calc_pairs_scores(
                query_tokens, sample_tokens_list, level="word"
            )
            for result in results:
                sample_tokens, sample_score, token_scores, token_weights = result
                tokens_str_list = []
                for token, token_weight, token_score in zip(
                    sample_tokens, token_weights, token_scores
                ):
                    token_str = token2str(token, score=token_score, weight=token_weight)
                    score_weight_str = scoreweight2str(
                        score=token_score, weight=token_weight
                    )
                    token_score_str = f"{token_str}{score_weight_str}"
                    tokens_str_list.append(token_score_str)
                tokens_str = " ".join(tokens_str_list)
                logger.line(f"    * {sample_score:>.4f}: [{tokens_str}]")

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


@Pyro5.server.behavior(instance_mode="single")
class FasttextDocVecModelRunner(FasttextModelRunner):
    dim = FASTTEXT_MERGED_MODEL_DIMENSION

    def __init__(
        self,
        *args,
        dim_scale: int = 6,
        downsample_nume_deno: tuple[int, int] = (3, 3),
        max_char_len: int = 3,
        token_split_wr: tuple[float, float] = (1.0, 0.8),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dim_scale = dim_scale
        self.downsample_nume_deno = downsample_nume_deno
        self.docvec_dim = (
            calc_padded_downsampled_cols(self.dim, downsample_nume_deno) * dim_scale
        )
        self.max_char_len = max_char_len
        self.token_wr, self.split_wr = token_split_wr
        self.downsample = partial(
            downsample, nume_deno=self.downsample_nume_deno, method="window"
        )

    @Pyro5.server.expose
    def calc_query_vector(self, doc: Union[str, list[str]]) -> np.ndarray:
        return self.calc_vector(
            doc,
            ignore_duplicates=False,
            weight_func="score",
            score_func="pos",
            normalize=False,
        )

    @Pyro5.server.expose
    def calc_weight_of_sample_token(self, token: str) -> float:
        return self.calc_weight_of_token(token, score_func="pos")

    def calc_tokens_splits_vectors(
        self, tokens_splits: list[list[str]]
    ) -> list[np.ndarray]:
        vectors = []
        for token_splits in tokens_splits:
            token_splits_vector = np.zeros(self.dim)
            weight_sum = 0
            for split in token_splits:
                vector = self.get_vector(split)
                weight = self.calc_weight_of_sample_token(split)
                token_splits_vector += vector * weight
                weight_sum += weight
            vectors.append(token_splits_vector / weight_sum)
        return vectors

    @Pyro5.server.expose
    def calc_sample_token_vectors(self, doc: Union[str, list[str]]) -> np.ndarray:
        tokens = self.preprocess(doc)
        if not tokens:
            return np.zeros((1, self.dim))
        vectors = [self.get_vector(token) for token in tokens]
        weights = [self.calc_weight_of_sample_token(token) for token in tokens]
        res = [vector * weight for vector, weight in zip(vectors, weights)]
        if self.max_char_len:
            tokens_splits, token_idxs = self.preprocessor.get_tokens_splits_and_idxs(
                tokens, max_char_len=self.max_char_len
            )
            token_splits_vectors = self.calc_tokens_splits_vectors(tokens_splits)
            for i, token_idx in enumerate(token_idxs):
                res[token_idx] = (
                    res[token_idx] * self.token_wr
                    + token_splits_vectors[i] * self.split_wr
                )
        return np.array(res)

    # @Pyro5.server.expose
    # def calc_stretch_query_vector(self, doc: Union[str, list[str]]) -> np.ndarray:
    #     query_vector = self.calc_query_vector(doc)
    #     downsampled_vector = self.downsample(query_vector)
    #     stretched_vector = stretch_copy(downsampled_vector, scale=self.dim_scale)
    #     return stretched_vector

    @Pyro5.server.expose
    def calc_stretch_query_vector(self, doc: Union[str, list[str]]) -> np.ndarray:
        token_vectors = self.calc_sample_token_vectors(doc)
        query_vector = np.sum(token_vectors, axis=0)
        downsampled_vector = self.downsample(query_vector)
        stretched_vector = stretch_copy(downsampled_vector, scale=self.dim_scale)
        return stretched_vector

    @Pyro5.server.expose
    def calc_stretch_sample_vector(self, doc: Union[str, list[str]]) -> np.ndarray:
        token_vectors = self.calc_sample_token_vectors(doc)
        downsampled_vector = self.downsample(token_vectors)
        stretched_vector = stretch_shift_add(downsampled_vector, scale=self.dim_scale)
        return stretched_vector

    @Pyro5.server.expose
    def test_doc_sims(self):
        importlib.reload(models.fasttext.test)
        from models.fasttext.test import TEST_PAIRS

        query_vec_func = self.calc_stretch_query_vector
        sample_vec_func = self.calc_stretch_sample_vector
        logger.note(f"> Testing doc-level similarity:")
        for query, samples in TEST_PAIRS:
            query_vec = query_vec_func(query).astype(np.float16)
            sample_vecs = [
                sample_vec_func(sample).astype(np.float16) for sample in samples
            ]
            sims = [dot_sim(query_vec, sample_vec) for sample_vec in sample_vecs]
            sample_sims = list(zip(samples, sims))
            sample_sims.sort(key=lambda x: x[1], reverse=True)
            logger.note(f"> [{query}]")
            for sample, sim in sample_sims:
                sim_str = logstr.mesg(f"{sim:.4f}")
                logger.mesg(f"  * {sim_str}: {sample}")
        logger.mesg(
            f"* docvec_dim: {logstr.success(self.docvec_dim)}\n"
            f"* vector_dim: {logstr.success(query_vec.shape)}"
        )


PYRO_NS = {
    "word": "fasttext_model_runner_word",
    "doc": "fasttext_model_runner_doc",
}


class FasttextModelRunnerServer:
    def __init__(
        self,
        model_class: Literal["word", "doc"] = "word",
        host: str = PYRO_ENVS["host"],
        port: int = PYRO_ENVS["port"],
        verbose: bool = False,
    ):
        self.model_class = model_class
        self.nameserver = PYRO_NS.get(model_class, "fasttext_model_runner")
        self.host = host
        self.port = port
        self.verbose = verbose

    def serve(self, runner: Union[FasttextModelRunner, FasttextDocVecModelRunner]):
        if self.verbose:
            logger.note(f"> Running as remote server:")
            logger.mesg(f"  * {self.host}:{self.port}")
        # Pyro5.server.serve({runner: self.nameserver}, host=self.host, port=self.port)
        daemon = Pyro5.server.Daemon(host=self.host, port=self.port)
        runner.run_mode = "remote"
        self.uri = daemon.register(runner, objectId=self.nameserver, weak=True)
        logger.file(f"  * {self.uri}", verbose=self.verbose)
        daemon.requestLoop()


class FasttextModelRunnerClient:
    def __init__(
        self,
        model_class: Literal["word", "doc"] = "word",
        host: str = PYRO_ENVS["host"],
        port: int = PYRO_ENVS["port"],
        verbose: bool = False,
    ):
        self.model_class = model_class
        self.nameserver = PYRO_NS.get(model_class, "fasttext_model_runner")
        self.uri = f"PYRO:{self.nameserver}@{host}:{port}"
        self.runner = Pyro5.api.Proxy(self.uri)
        self.verbose = verbose


class FasttextModelRunnerArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument(
            "-m", "--model-prefix", type=str, default=FASTTEXT_MERGED_MODEL_PREFIX
        )
        self.add_argument(
            "-ms",
            "--model-class",
            type=str,
            choices=["word", "doc"],
            default="word",
        )
        self.add_argument(
            "-k", "--tokenizer-prefix", type=str, default=SP_MERGED_MODEL_PREFIX
        )
        self.add_argument(
            "-q", "--token-freq-prefix", type=str, default=TOKEN_FREQ_PREFIX
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


def main(args: argparse.Namespace):
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
        if args.model_class == "word":
            runner_class = FasttextModelRunner
        elif args.model_class == "doc":
            runner_class = FasttextDocVecModelRunner
        else:
            raise ValueError("× Invalid model class!")
        runner = runner_class(
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
            "model_class": args.model_class,
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
            if args.model_class == "word":
                client.runner.test_pair_similarities()
            elif args.model_class == "doc":
                client.runner.test_doc_sims()
            else:
                raise ValueError("× Invalid model class!")
            # res = client.runner.list_models()
            # logger.success(dict_to_str(res))
        elif args.remote:
            runner.verbose = False
            server = FasttextModelRunnerServer(**remote_args, verbose=True)
            server.serve(runner)
        else:
            raise ValueError("× Invalid running mode!")


if __name__ == "__main__":
    parser = FasttextModelRunnerArgParser()
    args = parser.parse_args()
    main(args)

    # List models:
    # python -m models.fasttext.run -l

    # Test models:
    # python -m models.fasttext.run -t -m fasttext_tid_all_mv_60w -v 150000
    # python -m models.fasttext.run -t -m fasttext_tid_all_mv_30w -v 150000

    # Run remote server:
    # python -m models.fasttext.run -r -m fasttext_tid_all_mv_30w -v 150000
    # python -m models.fasttext.run -r -m fasttext_merged -v 150000 -w
    # python -m models.fasttext.run -r -ms doc -m fasttext_merged -v 150000 -w

    # Test remote client:
    # python -m models.fasttext.run -tc
    # python -m models.fasttext.run -tc -ms doc
