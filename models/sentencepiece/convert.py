import math
import re

from tclogger import dict_get


class DocSentenceConverter:
    """
    GB2312 编码表:
        - https://www.toolhelper.cn/Encoding/GB2312
        - A1A0~A3FE (JP), A6A0~A9FE (ZH)
    CJK Unicode Tables:
        - https://www.khngai.com/chinese/charmap/tbluni.php
        - 0x4E00~0x9FFF (ZH)
    Unicode Kanji Table:
        - http://www.rikai.com/library/kanjitables/kanji_codes.unicode.shtml
        - 0x3040~0x30FF (JP)
    """

    RE_CJK = r"\u4E00-\u9FFF\u3040-\u30FF"
    RE_EN = r"0-9a-zA-Z"
    RE_DASH = r"\-\_\."

    RE_CJK_SPACE = rf"(?<=[{RE_CJK}])\s+(?=[{RE_CJK}])"
    RE_NON_WORD = rf"[^({RE_CJK})|({RE_EN})]"

    PT_CJK_SPACE = re.compile(RE_CJK_SPACE)
    PT_NON_WORD = re.compile(RE_NON_WORD)
    PT_WHITESPACES = re.compile(r"\s{2,}")

    def doc_to_sentence(self, doc: dict) -> str:
        author = dict_get(doc, "owner.name", "")
        author_str = f"{author}" if author else ""

        title = dict_get(doc, "title", "")
        title_str = f"{title}" if title else ""

        desc = dict_get(doc, "desc", "")
        desc_str = f"{desc}" if desc else ""

        rtags = dict_get(doc, "rtags", "")
        tags = dict_get(doc, "tags", "")
        tags_str = f"{rtags}, {tags}" if tags else f"{rtags}"

        sentence = f"{author_str} | {title_str} | {desc_str} | {tags_str}"
        return sentence

    def remove_whitespaces_among_cjk(self, sentence: str) -> str:
        return self.PT_CJK_SPACE.sub("", sentence)

    def replace_non_word_with_whitespaces(self, sentence: str) -> str:
        return self.PT_NON_WORD.sub(" ", sentence)

    def merge_whitespaces(self, sentence: str) -> str:
        return self.PT_WHITESPACES.sub(" ", sentence).strip()

    def multiply_sentence(self, doc: dict, sentence: str) -> str:
        view = dict_get(doc, "stat.view", 0)
        if view <= 100:
            multi = 1
        elif view >= 1e7:
            multi = 8
        else:
            view_log = max(math.log(max(view, 0) + 1, 10), 0)
            multi = int(view_log) + 1
        sentence = f"{sentence} " * multi
        return sentence

    def convert_sentence(self, sentence: str) -> str:
        sentence = sentence.lower()
        # sentence = self.remove_whitespaces_among_cjk(sentence)
        sentence = self.replace_non_word_with_whitespaces(sentence)
        # sentence = self.merge_whitespaces(sentence)
        return sentence

    def convert(self, doc: dict) -> str:
        sentence = self.doc_to_sentence(doc)
        sentence = self.convert_sentence(sentence)
        sentence = self.multiply_sentence(doc, sentence)
        return sentence


if __name__ == "__main__":
    from tclogger import logger

    sentence = " 这是 一段 中文。这是日语：これは 日本語 です。 Here is some English. https://www.google.com"
    logger.note(sentence)
    converter = DocSentenceConverter()
    sentence = converter.convert_sentence(sentence)
    logger.success(sentence)

    # python -m models.sentencepiece.convert
