import math
import pandas as pd

from pathlib import Path
from tclogger import logger, logstr, brk, dict_to_str, Runtimer
from typing import Union, Literal

from configs.envs import SENTENCEPIECE_CKPT_ROOT, TOKEN_FREQ_ROOT
from configs.envs import SP_MERGED_MODEL_PREFIX
from models.sentencepiece.tokenizer import SentenceFullTokenizer


class FasttextModelFrequenizer:
    def __init__(
        self,
        token_freq_prefix: str = "video_texts_freq_all",
        tf_max_rows: int = 600000,
        tf_min_freq: int = 100,
        min_weight: float = 0.01,
        max_weight: float = 1.0,
        verbose: bool = False,
    ):
        self.token_freq_prefix = token_freq_prefix
        self.tf_max_rows = tf_max_rows
        self.tf_min_freq = tf_min_freq
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.verbose = verbose
        self.load_token_freq()
        self.init_tf_min_max()

    def load_token_freq(self):
        self.token_freq_path = TOKEN_FREQ_ROOT / f"{self.token_freq_prefix}.csv"
        if self.verbose:
            logger.note(f"> Loading token freq csv:")
            logger.file(f"  * {self.token_freq_path}")
            logger.mesg(f"  * tf_max_rows: {self.tf_max_rows}")
            logger.mesg(f"  * tf_min_freq: {self.tf_min_freq}")
            logger.mesg(f"  * min_weight: {self.min_weight}")
            logger.mesg(f"  * max_weight: {self.max_weight}")
        self.tf_df = pd.read_csv(self.token_freq_path)

    def init_tf_min_max(self):
        if self.tf_max_rows:
            self.tf_df = self.tf_df.head(self.tf_max_rows)
        if self.tf_min_freq:
            self.tf_df = self.tf_df[self.tf_df["doc_freq"] >= self.tf_min_freq]
        self.quantiles = [
            *[0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            *[0.7, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999, 0.9999, 0.99999, 1.0],
        ]
        self.tf_df_quantiles = self.tf_df["doc_freq"].quantile(
            self.quantiles, interpolation="nearest"
        )
        self.tf_df_min_freq = self.tf_df["doc_freq"].min()
        self.tf_df_max_freq = self.tf_df["doc_freq"].max()
        self.tf_df_qmin_freq = self.tf_df_quantiles[0]
        self.tf_df_qmax_freq = self.tf_df_quantiles[self.quantiles[-2]]
        self.tf_df_min_log_freq = self.calc_log_of_freq(self.tf_df_min_freq)
        self.tf_df_max_log_freq = self.calc_log_of_freq(self.tf_df_max_freq)
        self.token_doc_freq_dict = dict(
            zip(self.tf_df["token"], self.tf_df["doc_freq"])
        )
        if self.verbose:
            logger.mesg(f"  * tf_df_min_freq: {self.tf_df_min_freq}")
            logger.mesg(f"  * tf_df_max_freq: {self.tf_df_max_freq}")
            logger.mesg(f"  * tf_df_qmin_freq: {self.tf_df_qmin_freq}")
            logger.mesg(f"  * tf_df_qmax_freq: {self.tf_df_qmax_freq}")
            # logger.mesg(f"  * tf_df_quantiles:")
            # logger.file(self.tf_df_quantiles, indent=4)
            logger.mesg(f"  * tf_df_min_log_freq: {self.tf_df_min_log_freq}")
            logger.mesg(f"  * tf_df_max_log_freq: {self.tf_df_max_log_freq}")

    def get_token_freq(self, word: str) -> int:
        return self.token_doc_freq_dict.get(word, None)

    def get_tokens_freqs(self, words: list[str]) -> dict[str, int]:
        return [self.get_token_freq(word) for word in words]

    def calc_ratio_of_freq(self, freq: int) -> float:
        return (freq - self.tf_df_min_freq) / (
            self.tf_df_max_freq - self.tf_df_min_freq
        )

    def calc_quantile_of_freq(self, freq: int) -> float:
        for i, q in enumerate(self.tf_df_quantiles):
            if q > freq:
                break
        if i == 0:
            quantile = self.quantiles[0]
        else:
            quantile = self.quantiles[i - 1]
        return quantile

    def calc_log_of_freq(self, freq: int) -> float:
        return math.log(freq + 10, 10)

    def calc_log_ratio_of_freq(self, freq: int) -> float:
        return (self.calc_log_of_freq(freq) - self.tf_df_min_log_freq) / (
            self.tf_df_max_log_freq - self.tf_df_min_log_freq
        )

    def calc_weight_of_score(self, score: float) -> float:
        weight = self.min_weight + (self.max_weight - self.min_weight) * (1 - score)
        return weight

    def calc_weight_of_freq(
        self, freq: int, score_func: Literal["ratio", "quantile", "log"] = "log"
    ) -> float:
        if freq <= self.tf_df_qmin_freq:
            weight = self.max_weight
        elif freq >= self.tf_df_qmax_freq:
            weight = self.min_weight
        else:
            if score_func == "ratio":
                score = self.calc_ratio_of_freq(freq)
            elif score_func == "quantile":
                score = self.calc_quantile_of_freq(freq)
            else:
                score = self.calc_log_ratio_of_freq(freq)
            weight = self.calc_weight_of_score(score)
        return weight

    def calc_weight_of_token(self, word: str) -> float:
        token_freq = self.get_token_freq(word)
        if token_freq is None:
            weight = (self.min_weight + self.max_weight) / 2
        else:
            weight = self.calc_weight_of_freq(token_freq)
        return round(weight, 4)

    def calc_weights_of_tokens(self, words: list[str]) -> list[float]:
        return [self.calc_weight_of_token(word) for word in words]


class FasttextModelPreprocessor:
    def __init__(
        self, tokenizer_prefix: str = SP_MERGED_MODEL_PREFIX, verbose: bool = False
    ):
        self.tokenizer_prefix = tokenizer_prefix
        self.verbose = verbose
        self.load_tokenizer()

    def load_tokenizer(self):
        self.tokenizer_path = SENTENCEPIECE_CKPT_ROOT / f"{self.tokenizer_prefix}.model"
        if self.verbose:
            logger.note(f"> Loading tokenizer:")
            logger.file(f"  * {self.tokenizer_path}")
        self.tokenizer = SentenceFullTokenizer(
            model_path=self.tokenizer_path, drop_non_word=True, drop_whitespace=True
        )

    def tokenize(self, word: str) -> list[str]:
        return self.tokenizer.tokenize(word)

    def get_sep_indexes(self, words: list[str]) -> set[tuple[int, int]]:
        sep_indexes = []
        start = 0
        for word in words:
            end = start + len(word)
            sep_indexes.append((start, end))
            start = end
        return set(sep_indexes)

    def concat_singles(
        self, words: list[str], sep_indexes: set[tuple[int, int]] = {}
    ) -> list[str]:
        if words is None:
            return None
        res = []
        start = 0
        for i, word in enumerate(words):
            word_len = len(word)
            sep_index = (start, start + word_len)
            start += word_len
            if (
                i > 0
                and word_len == 1
                and len(words[i - 1]) == 1
                and sep_index not in sep_indexes
            ):
                res[-1] += word
            else:
                res.append(word)
        return res

    def preprocess(
        self, words: Union[str, list[str]], tokenize: bool = True
    ) -> list[str]:
        if words is None:
            return None
        if not isinstance(words, list):
            words = [words]
            sep_indexes = {}
        else:
            sep_indexes = self.get_sep_indexes(words)

        words = [word.lower() for word in words]
        if tokenize:
            words = [token for word in words for token in self.tokenize(word)]
            words = self.concat_singles(words, sep_indexes=sep_indexes)
        return words


if __name__ == "__main__":
    from models.fasttext.test import TEST_KEYWORDS, TEST_PAIRS

    timer = Runtimer()
    timer.__enter__()
    frequenizer = FasttextModelFrequenizer(
        token_freq_prefix="video_texts_freq_all",
        tf_max_rows=600000,
        tf_min_freq=100,
        min_weight=0.001,
        max_weight=1.0,
        verbose=True,
    )
    preprocessor = FasttextModelPreprocessor(
        tokenizer_prefix=SP_MERGED_MODEL_PREFIX, verbose=True
    )
    res = {}
    for word, words in TEST_PAIRS:
        # tword = words[0]
        tword = word
        pwords = preprocessor.preprocess(tword)
        freqs = frequenizer.get_tokens_freqs(pwords)
        weights = frequenizer.calc_weights_of_tokens(pwords)
        res[str(tword)] = {
            "tokens": pwords,
            "freqs": freqs,
            "weights": weights,
        }
        # logger.mesg(f"* {logstr.mesg(brk(word))}: {logstr.success(pword)}")
    logger.mesg(dict_to_str(res, align_list=False))
    timer.__exit__(None, None, None)

    # python -m models.fasttext.preprocess
