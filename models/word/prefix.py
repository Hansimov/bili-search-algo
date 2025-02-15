import marisa_trie
import pandas as pd
import pickle

from pathlib import Path
from tclogger import logger

from models.word.pinyin import ChinesePinyinizer


class PrefixMatcher:
    def __init__(
        self,
        words: list[str],
        scores: dict[str, float] = None,
        df: pd.DataFrame = None,
        token_freq_path: Path = None,
        verbose: bool = False,
    ):
        self.words = words
        self.scores = scores
        self.verbose = verbose
        self.df = df
        self.token_freq_path = token_freq_path
        self.pinyinizer = ChinesePinyinizer()
        self.build_words_trie()
        self.build_pinyin_trie()

    def build_words_trie(self):
        logger.note(f"> Building words trie: {len(self.words)}", verbose=self.verbose)
        self.words_trie = marisa_trie.Trie(self.words)

    def construct_pinyin_word_dict(self):
        if self.token_freq_path:
            self.word_dict_path = self.token_freq_path.with_suffix(".word_dict.pkl")
            if self.word_dict_path.exists():
                if self.verbose:
                    logger.note(f"> Loading pinyin word dict from:")
                    logger.file(f"  * {self.word_dict_path}")
                with open(self.word_dict_path, "rb") as rf:
                    data = pickle.load(rf)
                    self.pinyin_word_dict = data["pinyin_word_dict"]
                    self.short_word_dict = data["short_word_dict"]
                return

        logger.note(
            f"> Building pinyin word dict: {len(self.df)}", verbose=self.verbose
        )
        if "pinyin" in self.df.columns:
            self.pinyin_word_dict = (
                self.df.groupby("pinyin")["token"].apply(list).to_dict()
            )
        else:
            self.pinyin_word_dict = None
        if "short" in self.df.columns:
            self.short_word_dict = (
                self.df.groupby("short")["token"].apply(list).to_dict()
            )
        else:
            self.short_word_dict = None

        if self.token_freq_path:
            logger.note(f"> Saving pinyin word dict to:")
            logger.file(f"* {self.word_dict_path}")
            with open(self.word_dict_path, "wb") as wf:
                data = {
                    "pinyin_word_dict": self.pinyin_word_dict,
                    "short_word_dict": self.short_word_dict,
                }
                pickle.dump(data, wf)

    def build_pinyin_trie(self):
        self.pinyins_trie = None
        self.shorts_trie = None
        if self.df is None:
            return
        self.construct_pinyin_word_dict()
        if not self.pinyin_word_dict and not self.short_word_dict:
            return
        logger.note(f"> Building pinyins trie: {len(self.words)}", verbose=self.verbose)
        if self.pinyin_word_dict:
            self.pinyins = list(self.pinyin_word_dict.keys())
            self.pinyins_trie = marisa_trie.Trie(self.pinyins)
        if self.short_word_dict:
            self.shorts = list(self.short_word_dict.keys())
            self.shorts_trie = marisa_trie.Trie(self.shorts)

    def get_words_with_prefix(
        self,
        prefix: str,
        top_k: int = None,
        sort_by_score: bool = True,
        use_pinyin: bool = True,
        use_short: bool = False,
    ) -> list[str]:
        words_res = self.words_trie.keys(prefix)
        if use_pinyin or use_short:
            pinyin, short = self.pinyinizer.to_pinyin_str_and_short(prefix)
        if use_pinyin and self.pinyins_trie:
            pinyins_keys = self.pinyins_trie.keys(pinyin)
            pinyin_res = [
                word
                for pinyin in pinyins_keys
                for word in self.pinyin_word_dict[pinyin]
            ]
        else:
            pinyin_res = []
        if use_short and self.shorts_trie:
            shorts_keys = self.shorts_trie.keys(short)
            short_res = [
                word for short in shorts_keys for word in self.short_word_dict[short]
            ]
        else:
            short_res = []

        res = words_res + pinyin_res + short_res
        res = list(set(res))

        if sort_by_score and self.scores:
            res.sort(key=lambda x: self.scores[x], reverse=True)
        if top_k:
            return res[:top_k]
        else:
            return res


if __name__ == "__main__":
    from tclogger import logger

    words = ["apple", "apples", "banana", "bananas", "orange", "oranges"]
    matcher = PrefixMatcher(words)
    prefix = "app"
    res = matcher.get_words_with_prefix(prefix)
    logger.mesg(f"prefixed by [{prefix}]: {res}")

    # python -m models.sentencepiece.prefix
