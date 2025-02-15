import marisa_trie
import pypinyin
import re
import zhconv

from collections import defaultdict
from tclogger import logger

RE_ALPHA_DIGITS = re.compile(r"^[a-zA-Z0-9]+$")


class ChinesePinyinizer:
    def to_pinyin_choices(self, text: str) -> list[list[str]]:
        text = zhconv.convert(text, "zh-cn")
        pinyin_choices = pypinyin.pinyin(
            text, style=pypinyin.STYLE_NORMAL, heteronym=True
        )
        return pinyin_choices

    def to_pinyin_segs(self, text: str) -> list[str]:
        pinyin_choices = self.to_pinyin_choices(text)
        pinyin_segs = [choice[0] for choice in pinyin_choices]
        return pinyin_segs

    def to_pinyin_str(self, text: str, sep: str = "") -> str:
        if RE_ALPHA_DIGITS.match(text):
            return text
        pinyin_segs = self.to_pinyin_segs(text)
        pinyin_str = sep.join(pinyin_segs)
        return pinyin_str

    def to_pinyin_str_and_short(self, text: str, sep: str = "") -> tuple[str, str]:
        if RE_ALPHA_DIGITS.match(text):
            return text, text
        pinyin_segs = self.to_pinyin_segs(text)
        pinyin_str = sep.join(pinyin_segs)
        pinyin_short = "".join([seg[0] for seg in pinyin_segs])
        return pinyin_str, pinyin_short


class PrefixMatcher:
    def __init__(
        self,
        words: list[str],
        scores: dict[str, float] = None,
        use_pinyin: bool = False,
        verbose: bool = False,
    ):
        self.words = words
        self.scores = scores
        self.use_pinyin = use_pinyin
        self.verbose = verbose
        self.build_words_trie()
        if self.use_pinyin:
            self.build_pinyin_trie()
        else:
            self.pinyins = None

    def build_words_trie(self):
        logger.note(f"> Building words trie: {len(self.words)}", verbose=self.verbose)
        self.words_trie = marisa_trie.Trie(self.words)

    def build_pinyin_trie(self):
        logger.note(f"> Building pinyins trie: {len(self.words)}", verbose=self.verbose)
        self.pinyinizer = ChinesePinyinizer()
        self.pinyin_word_dict = defaultdict(list)
        self.pinyins = []
        self.shorts = []
        self.shorts_word_dict = defaultdict(list)
        for word in self.words:
            pinyin, short = self.pinyinizer.to_pinyin_str_and_short(word)
            if pinyin and pinyin != word:
                self.pinyins.append(pinyin)
                self.pinyin_word_dict[pinyin].append(word)
                self.shorts.append(short)
                self.shorts_word_dict[short].append(word)
        self.pinyins_trie = marisa_trie.Trie(self.pinyins)
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
        if use_pinyin and self.pinyins:
            pinyin, short = self.pinyinizer.to_pinyin_str_and_short(prefix)
            pinyins_keys = self.pinyins_trie.keys(pinyin)
            pinyin_words = [
                word
                for pinyin in pinyins_keys
                for word in self.pinyin_word_dict[pinyin]
            ]
            if use_short:
                shorts_keys = self.shorts_trie.keys(short)
                short_words = [
                    word
                    for short in shorts_keys
                    for word in self.shorts_word_dict[short]
                ]
            else:
                short_words = []
            res = words_res + pinyin_words + short_words
        else:
            res = words_res
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
    res = matcher.get_strs_with_prefix(prefix)
    logger.mesg(f"prefixed by [{prefix}]: {res}")

    # python -m models.sentencepiece.prefix
