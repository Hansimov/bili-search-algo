import argparse
import math
import re
import sys
import zhconv

from functools import partial
from tclogger import dict_get
from typing import Literal, Union

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

CH_CJK = r"\u4E00-\u9FFF\u3040-\u30FF"
CH_AB = r"0-9a-zA-Zα-ωΑ-Ω"
CH_DASH = r"\-\_\."
CH_LB = r"\(\[\{"
CH_RB = r"\)\]\}"

RE_SPACE_IN_CJK = rf"(?<=[{CH_CJK}])\s+(?=[{CH_CJK}])"
RE_NOT_DIGIT_DOT = r"\.(?!\d)"
# RE_NON_WORD = rf"[^{CH_CJK}{CH_AB}]+"
RE_NON_WORD = rf"[^{CH_CJK}〇{CH_AB}\-\.%‰]+|{RE_NOT_DIGIT_DOT}"
RE_WHITESPACES = r"\s{2,}"

CH_DIGIT_PREFIX = r"第前这那每"
CH_UNIT_NUM = r"毫厘分个十百千万兆亿"
CH_UNIT_DATE = r"年岁月周日天夜号点分秒"
CH_UNIT_WEIGHT = r"吨斤升两克磅里平米尺寸吋亩"
CH_UNIT_PAPER = r"集章篇部卷节回页张句行词字"
CH_UNIT_OTHRES = r"季阶级系路元块折期课题届次名人份只头种件位艘架辆楼层套间室厅厨卫杀袋包箱台倍星枚连"
RE_UNIT_COMBO = rf"小时|分钟|年代|周[年岁]|世纪|倍[速镜]|平米|平方米|平方公里|[公海英]里|光年|公斤|英[镑尺寸吋]|[美日欧]元|[{CH_UNIT_NUM}][{CH_UNIT_WEIGHT}]"
RE_UNIT_EN = rf"([mck]m|[km]w|h|min|[ukm]g|[nmu]s|[km]hz|kwh)(?<!a-zA-Z)"

RE_UNITS_ALL = rf"({RE_UNIT_COMBO}|{RE_UNIT_EN}|[{CH_UNIT_NUM}{CH_UNIT_DATE}{CH_UNIT_WEIGHT}{CH_UNIT_PAPER}{CH_UNIT_OTHRES}])"
RE_DIGITS_WITH_PREFIX_AND_UNIT = rf"[{CH_DIGIT_PREFIX}]?\d+{RE_UNITS_ALL}"
RE_DIGITS_WITH_BOUND = r"(^|\b)\d+(\b|$)"
RE_DIGITS_ALL = rf"({RE_DIGITS_WITH_PREFIX_AND_UNIT}|{RE_DIGITS_WITH_BOUND})"


PT_SPACE_IN_CJK = re.compile(RE_SPACE_IN_CJK)
PT_NON_WORD = re.compile(RE_NON_WORD)
PT_WHITESPACES = re.compile(RE_WHITESPACES)
PT_DIGITS_ALL = re.compile(RE_DIGITS_ALL)


class DocSentenceConverter:
    def __init__(
        self,
        collect_name: Literal["videos_texts", "users", "pages"] = "videos_texts",
        fields: Union[str, list[str]] = None,
        is_replace_non_word: bool = False,
        is_replace_digits: bool = False,
        is_simplify_chinese: bool = False,
        is_multiply_sentence: bool = False,
    ):
        self.collect_name = collect_name
        if fields:
            self.fields = fields
        else:
            self.fields = None
        self.is_replace_non_word = is_replace_non_word
        self.is_replace_digits = is_replace_digits
        self.is_simplify_chinese = is_simplify_chinese
        self.is_multiply_sentence = is_multiply_sentence
        self.init_doc_to_sentence()

    def init_doc_to_sentence(self):
        if self.collect_name == "users":
            self.doc_to_sentence = partial(self.get_doc_field, field="name")
            self.multiply_sentence = self.multiply_sentence_by_videos_count
        elif self.collect_name == "videos_texts":
            if self.fields:
                self.doc_to_sentence = partial(self.get_doc_fields, fields=self.fields)
            else:
                self.doc_to_sentence = self.get_doc_all_fields
            self.multiply_sentence = self.multiply_sentence_by_stat_view
        elif self.collect_name == "pages":
            self.doc_to_sentence = partial(self.get_doc_field, field="revision.text")
            self.multiply_sentence = lambda doc, sentence: sentence
        else:
            raise ValueError(f"× Invalid collect_name: {self.collect_name}")

    def get_doc_field(self, doc: dict, field: str) -> str:
        return dict_get(doc, field, "")

    def get_doc_fields(self, doc: dict, fields: list[str]) -> str:
        field_strs = [self.get_doc_field(doc, field) for field in fields]
        fields_str = " | ".join([field_str for field_str in field_strs if field_str])
        return fields_str

    def get_doc_all_fields(self, doc: dict) -> str:
        return self.get_doc_fields(
            doc, ["owner.name", "title", "desc", "rtags", "tags"]
        )

    def remove_whitespaces_among_cjk(self, sentence: str) -> str:
        return PT_SPACE_IN_CJK.sub("", sentence)

    def replace_non_word_with_whitespaces(self, sentence: str) -> str:
        return PT_NON_WORD.sub(" ", sentence)

    def replace_digits(self, sentence: str) -> str:
        return PT_DIGITS_ALL.sub("", sentence)

    def merge_whitespaces(self, sentence: str) -> str:
        return PT_WHITESPACES.sub(" ", sentence).strip()

    def multiply_sentence_by_stat_view(self, doc: dict, sentence: str) -> str:
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

    def multiply_sentence_by_videos_count(self, doc: dict, sentence: str) -> str:
        videos = dict_get(doc, "videos", [])
        videos_count = min(len(videos), 1)
        sentence = f"{sentence} " * videos_count
        return sentence

    def convert_sentence(self, sentence: str) -> str:
        sentence = sentence.lower()
        # sentence = self.remove_whitespaces_among_cjk(sentence)
        if self.is_replace_non_word:
            sentence = self.replace_non_word_with_whitespaces(sentence)
        if self.is_replace_digits:
            sentence = self.replace_digits(sentence)
        if self.is_simplify_chinese:
            sentence = zhconv.convert(sentence, "zh-cn")
        # sentence = self.merge_whitespaces(sentence)
        return sentence

    def convert(self, doc: dict) -> str:
        sentence = self.doc_to_sentence(doc)
        sentence = self.convert_sentence(sentence)
        if self.is_multiply_sentence:
            sentence = self.multiply_sentence(doc, sentence)
        return sentence


class ArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("-t", "--test", action="store_true")

    def parse_args(self):
        self.args, self.unknown_args = self.parse_known_args(sys.argv[1:])
        return self.args


if __name__ == "__main__":
    from tclogger import logger
    from models.sentencepiece.test import TEST_TOKENS, TEST_SENTENCES
    import timeit

    args = ArgParser().parse_args()
    converter = DocSentenceConverter(is_simplify_chinese=True, is_replace_non_word=True)

    if args.test:
        logger.note("> Testing ...")
        sentence = TEST_TOKENS
        logger.note(sentence)
        sentence = converter.convert_sentence(sentence)
        logger.success(sentence)
        for sentence in TEST_SENTENCES:
            converted_sentence = converter.convert_sentence(sentence)
            logger.mesg(f"  * {converted_sentence}")
    else:
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

    # python -m data_utils.videos.convert
