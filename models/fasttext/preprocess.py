import math

from tclogger import logger, logstr, brk, dict_to_str, Runtimer
from typing import Union, Literal

from configs.envs import SENTENCEPIECE_CKPT_ROOT, TOKEN_FREQS_ROOT
from configs.envs import SP_MERGED_MODEL_PREFIX, TOKEN_FREQ_PREFIX
from data_utils.videos.freq import read_token_freq_csv
from models.sentencepiece.tokenizer import SentenceFullTokenizer
from models.sentencepiece.tokenizer import PT_DIGITS_ZH_WITH_UNIT, PT_DIGITS_ZH
from models.sentencepiece.tokenizer import calc_cjk_char_len
from models.word.pos import INCLUDE_POS_NAMES, MID_POS_NAMES, EXCLUDE_POS_NAMES
from models.word.prefix import PrefixMatcher

TokenScoreFuncType = Literal["one", "ratio", "quantile", "log", "power", "pos"]
FreqScoreFuncType = Literal["ratio", "quantile", "log", "power"]


class FasttextModelFrequenizer:
    def __init__(
        self,
        token_freq_prefix: str = TOKEN_FREQ_PREFIX,
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
        self.init_tf_vars()

    def load_token_freq(self):
        self.token_freq_path = TOKEN_FREQS_ROOT / f"{self.token_freq_prefix}.csv"
        if self.verbose:
            logger.note(f"> Loading token freq csv:")
            logger.file(f"  * {self.token_freq_path}")
            logger.mesg(f"  * tf_max_rows: {self.tf_max_rows}")
            logger.mesg(f"  * tf_min_freq: {self.tf_min_freq}")
            logger.mesg(f"  * min_weight: {self.min_weight}")
            logger.mesg(f"  * max_weight: {self.max_weight}")
        self.tf_df = read_token_freq_csv(self.token_freq_path)

    def init_tf_vars(self):
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
        self.tf_df_min_freq = self.tf_df["doc_freq"].min().astype(int)
        self.tf_df_max_freq = self.tf_df["doc_freq"].max().astype(int)
        self.tf_df_qmin_freq = self.tf_df_quantiles[0]
        self.tf_df_qmax_freq = self.tf_df_quantiles[self.quantiles[-2]]
        self.tf_df_min_log_freq = self.calc_log_of_freq(self.tf_df_min_freq)
        self.tf_df_max_log_freq = self.calc_log_of_freq(self.tf_df_max_freq)
        self.tf_df_min_power_freq = self.calc_power_of_freq(self.tf_df_min_freq)
        self.tf_df_max_power_freq = self.calc_power_of_freq(self.tf_df_max_freq)
        self.token_doc_freq_dict = dict(
            zip(self.tf_df["token"], self.tf_df["doc_freq"])
        )
        if "pos" in self.tf_df.columns:
            self.token_pos_dict = dict(zip(self.tf_df["token"], self.tf_df["pos"]))
        else:
            self.token_pos_dict = None
        if self.verbose:
            logger.mesg(f"  * tf_df_min_freq: {self.tf_df_min_freq}")
            logger.mesg(f"  * tf_df_max_freq: {self.tf_df_max_freq}")
            # logger.mesg(f"  * tf_df_qmin_freq: {self.tf_df_qmin_freq}")
            # logger.mesg(f"  * tf_df_qmax_freq: {self.tf_df_qmax_freq}")
            # logger.mesg(f"  * tf_df_quantiles:")
            # logger.file(self.tf_df_quantiles, indent=4)
            # logger.mesg(f"  * tf_df_min_log_freq: {self.tf_df_min_log_freq}")
            # logger.mesg(f"  * tf_df_max_log_freq: {self.tf_df_max_log_freq}")
            # logger.mesg(f"  * tf_df_min_power_freq: {self.tf_df_min_power_freq}")
            # logger.mesg(f"  * tf_df_max_power_freq: {self.tf_df_max_power_freq}")

    def get_token_freq(self, word: str) -> int:
        return self.token_doc_freq_dict.get(word, None)

    def get_tokens_freqs(self, words: list[str]) -> list[int]:
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

    def calc_log_of_freq(self, freq: int, base: Union[int, float] = None) -> float:
        return math.log(freq + 10, base or 10)

    def calc_log_ratio_of_freq(
        self, freq: int, base: Union[int, float] = None
    ) -> float:
        min_log = self.calc_log_of_freq(self.tf_df_min_freq, base)
        max_log = self.calc_log_of_freq(self.tf_df_max_freq, base)
        return (self.calc_log_of_freq(freq, base) - min_log) / (max_log - min_log)

    def calc_power_of_freq(self, freq: int, base: Union[int, float] = None) -> float:
        return math.pow(freq, base or 0.5)

    def calc_power_ratio_of_freq(
        self, freq: int, base: Union[int, float] = None
    ) -> float:
        min_power = self.calc_power_of_freq(self.tf_df_min_freq, base)
        max_power = self.calc_power_of_freq(self.tf_df_max_freq, base)
        return (self.calc_power_of_freq(freq, base) - min_power) / (
            max_power - min_power
        )

    def calc_weight_of_freq_score(
        self,
        score: float,
        min_weight: float = None,
        max_weight: float = None,
    ) -> float:
        min_weight = min_weight or self.min_weight
        max_weight = max_weight or self.max_weight
        weight = min_weight + (max_weight - min_weight) * (1 - score)
        return weight

    def calc_weight_of_freq(
        self,
        freq: int,
        score_func: FreqScoreFuncType = "log",
        base: Union[int, float] = None,
        min_weight: float = None,
        max_weight: float = None,
    ) -> float:
        min_weight = min_weight or self.min_weight
        max_weight = max_weight or self.max_weight
        if freq <= self.tf_df_qmin_freq:
            weight = max_weight
        elif freq >= self.tf_df_qmax_freq:
            weight = min_weight
        else:
            if score_func == "ratio":
                score = self.calc_ratio_of_freq(freq)
            elif score_func == "quantile":
                score = self.calc_quantile_of_freq(freq)
            elif score_func == "log":
                score = self.calc_log_ratio_of_freq(freq, base=base)
            else:
                score = self.calc_power_ratio_of_freq(freq, base=base)
            weight = self.calc_weight_of_freq_score(
                score, min_weight=min_weight, max_weight=max_weight
            )
            logger.mesg(
                # f"score_func: {score_func}, base: {base}, "
                # f"min_weight: {min_weight}, max_weight: {max_weight}, "
                f"score: {round(score,4)}, weight: {round(weight,4)}, freq: {freq}",
                verbose=self.verbose,
            )
        return weight

    def calc_weight_of_pos(
        self, word: str, min_weight: float = None, max_weight: float = None
    ) -> float:
        min_weight = min_weight or self.min_weight
        max_weight = max_weight or self.max_weight
        word_pos = self.token_pos_dict.get(word, None)
        if word_pos in EXCLUDE_POS_NAMES:
            weight = min_weight
        elif word_pos in MID_POS_NAMES:
            weight = (min_weight + max_weight) / 2
        else:
            weight = max_weight
        return weight

    def calc_weight_of_token(
        self,
        word: str,
        score_func: TokenScoreFuncType = "one",
        base: Union[int, float] = None,
        min_weight: float = None,
        max_weight: float = None,
    ) -> float:
        if score_func == "one":
            return base or 1.0
        elif score_func == "pos":
            return self.calc_weight_of_pos(
                word, min_weight=min_weight, max_weight=max_weight
            )
        else:
            token_freq = self.get_token_freq(word)
            min_weight = min_weight or self.min_weight
            max_weight = max_weight or self.max_weight
            if token_freq is None:
                weight = (min_weight + max_weight) / 2
            else:
                logger.note(word, end=": ", verbose=self.verbose)
                weight = self.calc_weight_of_freq(
                    token_freq,
                    score_func=score_func,
                    base=base,
                    min_weight=min_weight,
                    max_weight=max_weight,
                )
            return round(weight, 4)

    def calc_weights_of_tokens(
        self,
        words: list[str],
        score_func: TokenScoreFuncType = "one",
        base: Union[int, float] = None,
        min_weight: float = None,
        max_weight: float = None,
    ) -> list[float]:
        return [
            self.calc_weight_of_token(
                word,
                score_func=score_func,
                base=base,
                min_weight=min_weight,
                max_weight=max_weight,
            )
            for word in words
        ]


class FasttextModelPreprocessor:
    def __init__(
        self,
        tokenizer_prefix: str = SP_MERGED_MODEL_PREFIX,
        token_freq_prefix: str = TOKEN_FREQ_PREFIX,
        verbose: bool = False,
    ):
        self.tokenizer_prefix = tokenizer_prefix
        self.token_freq_prefix = token_freq_prefix
        self.verbose = verbose
        self.load_tokenizer()
        self.load_frquenizer()
        self.load_prefixer()

    def load_tokenizer(self):
        self.tokenizer_path = SENTENCEPIECE_CKPT_ROOT / f"{self.tokenizer_prefix}.model"
        if self.verbose:
            logger.note(f"> Loading tokenizer:")
            logger.file(f"  * {self.tokenizer_path}")
        self.tokenizer = SentenceFullTokenizer(
            model_path=self.tokenizer_path, drop_non_word=True, drop_whitespace=True
        )
        self.model_tokenizer = self.tokenizer.model_tokenizer

    def load_prefixer(self):
        self.token_freq_path = TOKEN_FREQS_ROOT / f"{self.token_freq_prefix}.csv"
        self.tf_df = read_token_freq_csv(self.token_freq_path)
        tokens = self.tf_df["token"].tolist()
        scores = dict(zip(self.tf_df["token"], self.tf_df["doc_freq"]))
        self.prefixer = PrefixMatcher(
            tokens,
            scores=scores,
            df=self.tf_df,
            token_freq_path=self.token_freq_path,
            verbose=True,
        )

    def load_frquenizer(self):
        self.frequenizer = FasttextModelFrequenizer(
            token_freq_prefix=self.token_freq_prefix,
            tf_max_rows=600000,
            tf_min_freq=100,
            min_weight=0.001,
            max_weight=1.0,
            verbose=True,
        )

    def tokenize(self, word: str) -> list[str]:
        return self.tokenizer.tokenize(word)

    def get_words_with_prefix(
        self,
        prefix: str,
        top_k: int = None,
        use_pinyin: bool = False,
        use_short: bool = False,
    ) -> list[str]:
        return self.prefixer.get_words_with_prefix(
            prefix, top_k=top_k, use_pinyin=use_pinyin, use_short=use_short
        )

    def get_sep_indexes(self, words: list[str]) -> set[tuple[int, int]]:
        """This function is to get indexes of separated chars in original sentence,
        which serves later single-char words concat."""
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

    def split_by_maxlen(self, words: str, max_char_len: int) -> list[str]:
        res = []
        for word in words:
            if max_char_len >= calc_cjk_char_len(word):
                res.append(word)
            else:
                word_segs = self.model_tokenizer.tokenize_maxlen(
                    word, max_char_len=max_char_len
                )
                res.extend(word_segs)
        return res

    def get_tokens_splits_and_idxs(
        self, tokens: list[str], max_char_len: int
    ) -> tuple[list[list[str]], list[int]]:
        """Similar to `split_max_len`, but there are two differences:
        1.  (a) input of this func is `list[str]`, which is tokenized sequences,
            (b) while `split_max_len` accpets `str`, which is a not-tokenized sentence;
        2.  (a) output of this func `tuple`, which is subset-list of splitale-only tokens and indexes, which is meant to reduce the times of calling `calc_vector`.
            (b) while `split_max_len` returns `list[str]`, which is a full-list of all splitted tokens

        This function should be used after `preprocess`.
        And the following two stages should not be done at the same time:
        1. specify `max_char_len` in `preprocess`;
        2. call `get_sub_tokens_and_idxs` after `preprocess`,
        Otherwise it would tokenize the same words list twice.
        """
        tokens_splits = []
        idxs = []
        for idx, word in enumerate(tokens):
            if (
                max_char_len >= calc_cjk_char_len(word)
                or PT_DIGITS_ZH_WITH_UNIT.match(word)
                or PT_DIGITS_ZH.match(word)
            ):
                pass
            else:
                word_splits = self.model_tokenizer.tokenize_maxlen(
                    word, max_char_len=max_char_len
                )
                tokens_splits.append(word_splits)
                idxs.append(idx)
        return tokens_splits, idxs

    def preprocess(
        self,
        words: Union[str, list[str]],
        tokenize: bool = True,
        concat_singles: bool = True,
        max_char_len: int = None,
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
            if max_char_len:
                with self.tokenizer.temp_max_char_len(max_char_len):
                    words = [token for word in words for token in self.tokenize(word)]
            else:
                words = [token for word in words for token in self.tokenize(word)]
                if concat_singles:
                    words = self.concat_singles(words, sep_indexes=sep_indexes)
        return words


def test_frequenizer(test_sentences: list[str]):
    frequenizer = FasttextModelFrequenizer(
        token_freq_prefix=TOKEN_FREQ_PREFIX,
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
    for sentence in test_sentences:
        tokens = preprocessor.preprocess(sentence)
        logger.mesg(f"* {tokens}")
        freqs = frequenizer.get_tokens_freqs(tokens)
        weights = frequenizer.calc_weights_of_tokens(tokens)
        res[str(sentence)] = {
            "tokens": tokens,
            "freqs": freqs,
            "weights": weights,
        }
    logger.mesg(dict_to_str(res, align_list=False))


def test_pos_tagger(test_sentences: list[str]):
    preprocessor = FasttextModelPreprocessor(
        tokenizer_prefix=SP_MERGED_MODEL_PREFIX, verbose=True
    )
    res = {}
    tagger = HanlpPosTagger(verbose=True)
    for sentence in test_sentences:
        tokens = preprocessor.preprocess(sentence, concat_singles=True, max_char_len=3)
        logger.mesg(f"* {tokens}")
        tags = tagger.tag_pos(tokens)
        token_tags = " ".join(
            [
                f"{logstr.success(token)}/{logstr.file(tagger.tags_to_names(tag))}"
                for token, tag in zip(tokens, tags)
            ]
        )
        logger.mesg(f"  * {token_tags}")
    logger.mesg(dict_to_str(res, align_list=False))


def test_split_tokens(test_sentences: list[str]):
    preprocessor = FasttextModelPreprocessor(
        tokenizer_prefix=SP_MERGED_MODEL_PREFIX, verbose=True
    )
    for sentence in test_sentences:
        tokens = preprocessor.preprocess(sentence)
        tokens_splits, token_idxs = preprocessor.get_tokens_splits_and_idxs(
            tokens, max_char_len=3
        )
        logger.mesg(f"  * {tokens_splits}")
        # logger.mesg(f"  * {list(zip(tokens_splits, token_idxs))}")


def test_max_len(test_sentences: list[str]):
    preprocessor = FasttextModelPreprocessor(
        tokenizer_prefix=SP_MERGED_MODEL_PREFIX, verbose=True
    )
    for sentence in test_sentences:
        tokens = preprocessor.preprocess(sentence, max_char_len=3)
        logger.mesg(f"* {tokens}")


def test_prefix():
    preprocessor = FasttextModelPreprocessor(
        tokenizer_prefix=SP_MERGED_MODEL_PREFIX,
        token_freq_prefix=TOKEN_FREQ_PREFIX,
        verbose=True,
    )
    prefixes = [
        *["大语言", "欣小", "咬人", "红警", "尤里"],
        *["小米", "影视", "游戏", "cs", "小鸟", "上海", "hbk"],
    ]
    pinyins = ["ysjf", "yingshi", "hongj", "anye"]
    for prefix in prefixes + pinyins:
        logger.note(f"> {prefix}")
        if prefix in pinyins:
            pinyin_params = {"use_pinyin": True, "use_short": True}
        else:
            pinyin_params = {}
        words = preprocessor.get_words_with_prefix(prefix, top_k=10, **pinyin_params)
        for word in words:
            logger.mesg(f"  * {word}")


if __name__ == "__main__":
    from itertools import chain
    from models.hanlp.pos import HanlpPosTagger
    from models.fasttext.test import TEST_KEYWORDS, TEST_PAIRS
    from models.sentencepiece.test import TEST_WORDS, TEST_SENTENCES

    TEST_SENTS = [
        *list(chain.from_iterable([words for word, words in TEST_PAIRS])),
        *TEST_SENTENCES,
        *TEST_WORDS,
    ]
    with Runtimer():
        # test_frequenizer(TEST_SENTS)
        # test_pos_tagger(TEST_SENTS)
        # test_split_tokens(TEST_SENTS)
        # test_max_len(TEST_SENTS)
        test_prefix()

    # python -m models.fasttext.preprocess
