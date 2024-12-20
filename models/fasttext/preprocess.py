from pathlib import Path
from tclogger import logger, logstr, brk, dict_to_str
from typing import Union

from configs.envs import SENTENCEPIECE_CKPT_ROOT
from models.sentencepiece.tokenizer import SentenceFullTokenizer


class FasttextModelPreprocessor:
    def __init__(
        self,
        tokenizer_prefix: Union[str, Path] = "sp_400k_merged",
        verbose: bool = False,
    ):
        self.tokenizer_prefix = tokenizer_prefix
        self.verbose = verbose
        self.load_tokenizer()

    def load_tokenizer(self):
        self.tokenizer_path = SENTENCEPIECE_CKPT_ROOT / f"{self.tokenizer_prefix}.model"
        if self.verbose:
            logger.note(f"> Loading tokenizer:")
            logger.mesg(f"  * {self.tokenizer_path}")
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
            if i == len(words) - 1 and word_len == 1:
                if res and sep_indexes and sep_index not in sep_indexes:
                    res[-1] += word
                else:
                    res.append(word)
            elif (
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

    preprocessor = FasttextModelPreprocessor(
        tokenizer_prefix="sp_400k_merged", verbose=True
    )
    res = {}
    for word, words in TEST_PAIRS:
        pword = preprocessor.preprocess(word)
        res[str(word)] = pword
        # logger.mesg(f"* {logstr.mesg(brk(word))}: {logstr.success(pword)}")
    logger.mesg(dict_to_str(res, align_list=False))

    # python -m models.fasttext.preprocess
