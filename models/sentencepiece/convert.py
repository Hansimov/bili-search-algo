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
    RE_EN = r"0-9a-zA-Zα-ωΑ-Ω"
    RE_DASH = r"\-\_\."

    RE_CJK_SPACE = rf"(?<=[{RE_CJK}])\s+(?=[{RE_CJK}])"
    RE_NON_WORD = rf"[^{RE_CJK}{RE_EN}]+"
    RE_WHITESPACES = r"\s{2,}"

    RE_DIGIT_PREFIX = r"[第前这那每]"
    RE_UNIT_NUM = r"毫厘分个十百千万兆亿"
    RE_UNIT_DATE = r"年岁月周日天夜号点分秒"
    RE_UNIT_WEIGHT = r"吨斤升两克磅里平米尺寸吋"
    RE_UNIT_PAPER = r"集章篇部卷节回页张句行词字"
    RE_UNIT_OTHRES = r"季阶级系路元块折期课题届次名人份只头种件位辆楼层套间室厅厨卫杀袋包箱台倍星枚连"
    RE_UNIT_COMBO = rf"小时|分钟|周[年岁]|倍[速镜]|平米|[公海英]里|英[镑尺寸吋]|[美日欧]元|[{RE_UNIT_NUM}][{RE_UNIT_WEIGHT}]"
    RE_UNIT_EN = rf"([mck]m|[km]w|h|min|[ukm]g|[nmu]s|[km]hz|kwh)(?<!a-zA-Z)"
    RE_UNITS = rf"({RE_UNIT_COMBO}|{RE_UNIT_EN}|[{RE_UNIT_NUM}{RE_UNIT_DATE}{RE_UNIT_WEIGHT}{RE_UNIT_PAPER}{RE_UNIT_OTHRES}])"
    RE_DIGIT_UNIT = rf"{RE_DIGIT_PREFIX}?\d+{RE_UNITS}"
    RE_DIGIT_PURE = r"(^|\b)\d+(\b|$)"

    RE_DIGITS_ALL = rf"({RE_DIGIT_UNIT}|{RE_DIGIT_PURE})"

    PT_CJK_SPACE = re.compile(RE_CJK_SPACE)
    PT_NON_WORD = re.compile(RE_NON_WORD)
    PT_WHITESPACES = re.compile(RE_WHITESPACES)
    RE_DIGITS_ALL = re.compile(RE_DIGITS_ALL)

    def __init__(self, fields: list[str] = None):
        if fields:
            self.fields = fields
        else:
            self.fields = ["owner.name", "title", "desc", "tags"]

    def doc_to_sentence(self, doc: dict) -> str:
        if "owner.name" in self.fields:
            author = dict_get(doc, "owner.name", "")
            author_str = f"{author}" if author else ""
        else:
            author_str = ""

        if "title" in self.fields:
            title = dict_get(doc, "title", "")
            title_str = f"{title}" if title else ""
        else:
            title_str = ""

        if "desc" in self.fields:
            desc = dict_get(doc, "desc", "")
            desc_str = f"{desc}" if desc else ""
        else:
            desc_str = ""

        if "tags" in self.fields:
            rtags = dict_get(doc, "rtags", "")
            tags = dict_get(doc, "tags", "")
            tags_str = f"{rtags}, {tags}" if tags else f"{rtags}"
        else:
            tags_str = ""

        sentence_list = [s for s in [author_str, title_str, desc_str, tags_str] if s]
        sentence = " | ".join(sentence_list)
        return sentence

    def remove_whitespaces_among_cjk(self, sentence: str) -> str:
        return self.PT_CJK_SPACE.sub("", sentence)

    def replace_non_word_with_whitespaces(self, sentence: str) -> str:
        return self.PT_NON_WORD.sub(" ", sentence)

    def replace_digits(self, sentence: str) -> str:
        return self.RE_DIGITS_ALL.sub("", sentence)

    def merge_whitespaces(self, sentence: str) -> str:
        return self.PT_WHITESPACES.sub(" ", sentence).strip()

    def multiply_sentence(self, doc: dict, sentence: str) -> str:
        view = dict_get(doc, "stat.view", 0)
        if view <= 100:
            return sentence
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
        sentence = self.replace_digits(sentence)
        # sentence = self.merge_whitespaces(sentence)
        return sentence

    def convert(self, doc: dict) -> str:
        sentence = self.doc_to_sentence(doc)
        sentence = self.convert_sentence(sentence)
        sentence = self.multiply_sentence(doc, sentence)
        return sentence


if __name__ == "__main__":
    from tclogger import logger
    from models.sentencepiece.test import TEST_TOKENS
    import timeit

    sentence = TEST_TOKENS
    logger.note(sentence)
    converter = DocSentenceConverter()
    sentence = converter.convert_sentence(sentence)
    logger.success(sentence)

    logger.note("> Benchmarking ...")
    sum_time = 0
    epochs, iterations = 5, 10000
    for i in range(epochs):
        res = timeit.timeit(
            lambda: converter.convert_sentence(sentence), number=iterations
        )
        logger.file(f"* {res:.6f}")
        sum_time += res
    logger.success(f"{sum_time/epochs:.6f}")

    # python -m models.sentencepiece.convert
