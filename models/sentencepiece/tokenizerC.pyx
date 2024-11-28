# cython: profile=True

import re
import sentencepiece as spm

from pathlib import Path
from tclogger import logstr
from typing import Union, List, Tuple

# global regex
cdef unicode CH_CJK = u"\u4E00-\u9FFF\u3040-\u30FF"
cdef unicode CH_AB = u"0-9a-zA-Zα-ωΑ-Ω"
cdef unicode CH_DIGIT_PREFIX = u"第前这那每"
cdef unicode CH_UNIT_NUM = u"毫厘分个十百千万兆亿"
cdef unicode CH_UNIT_DATE = u"年岁月周日天夜号点分秒"
cdef unicode CH_UNIT_WEIGHT = u"吨斤升两克磅里平米尺寸吋"
cdef unicode CH_UNIT_PAPER = u"集章篇部卷节回页张句行词字"
cdef unicode CH_UNIT_OTHRES = u"季阶级系路元块折期课题届次名人份只头种件位辆楼层套间室厅厨卫杀袋包箱台倍星枚连"
cdef unicode RE_UNIT_COMBO = f"小时|分钟|周[年岁]|倍[速镜]|平米|平方米|平方公里|[公海英]里|公斤|英[镑尺寸吋]|[美日欧]元|[{CH_UNIT_NUM}][{CH_UNIT_WEIGHT}]"
cdef unicode RE_UNIT_EN = f"([mck]m|[km]w|h|min|[ukm]g|[nmu]s|[km]hz|kwh)(?<!a-zA-Z)"
cdef unicode RE_UNITS_ALL = f"({RE_UNIT_COMBO}|{RE_UNIT_EN}|[{CH_UNIT_NUM}{CH_UNIT_DATE}{CH_UNIT_WEIGHT}{CH_UNIT_PAPER}{CH_UNIT_OTHRES}])"

# pre-tokenizer regex
cdef str CH_LB = r"\(\[\{"
cdef str CH_RB = r"\)\]\}"

cdef unicode CH_DIGIT_ZH = u"〇零一二两三四五六七八九十"
cdef unicode CH_DIGIT_ZH_MUL = u"十百千万亿"

cdef unicode RE_DIGITS_PURE = u"\d+"
cdef unicode RE_DOT_DIGITS = u"\d*\.\d+"
cdef unicode RE_DIGITS_AND_DOTS = u"[\d\.]+(?<!\.)"
cdef unicode RE_DIGITS_NUMBER = f"({RE_DOT_DIGITS}|{RE_DIGITS_AND_DOTS})"
cdef unicode RE_DIGITS_AND_DOTS_WITH_PREFIX_AND_UNIT = (
    f"[{CH_DIGIT_PREFIX}]?\s*{RE_DIGITS_AND_DOTS}\s*{RE_UNITS_ALL}"
)
cdef unicode RE_DIGITS_WITH_DOTS_AND_BRS = (
    f"\[{RE_DIGITS_AND_DOTS}\]|\({RE_DIGITS_AND_DOTS}\)|{{{RE_DIGITS_AND_DOTS}}}"
)
cdef unicode RE_DIGITS_ALL = f"(?:{RE_DIGITS_AND_DOTS_WITH_PREFIX_AND_UNIT}|{RE_DIGITS_WITH_DOTS_AND_BRS}|{RE_DIGITS_NUMBER})"

cdef unicode RE_NON_WORD = f"[^{CH_CJK}〇{CH_AB}\.{CH_LB}{CH_RB}]+"

cdef unicode RE_DIGITS_ZH = (
    f"(([{CH_DIGIT_ZH}][{CH_DIGIT_ZH_MUL}])+[{CH_DIGIT_ZH}]?|[{CH_DIGIT_ZH}]+)"
)
cdef unicode RE_DIGITS_ZH_WITH_UNIT = f"[{CH_DIGIT_PREFIX}]?{RE_DIGITS_ZH}{RE_UNITS_ALL}"

cdef unicode RE_DIGITS_UNITS_AND_NON_WORD = f"(?P<digits_with_unit>{RE_DIGITS_AND_DOTS_WITH_PREFIX_AND_UNIT})|(?P<digits_zh_with_unit>{RE_DIGITS_ZH_WITH_UNIT})|(?P<non_word>{RE_NON_WORD})"

cdef object PT_NON_WORD = re.compile(RE_NON_WORD)
cdef object PT_DIGITS_ALL = re.compile(RE_DIGITS_ALL)
cdef object PT_DIGITS_ZH_WITH_UNIT = re.compile(RE_DIGITS_ZH_WITH_UNIT)
cdef object PT_DIGITS_UNITS_AND_NON_WORDS = re.compile(RE_DIGITS_UNITS_AND_NON_WORD)

cdef class SentencePreTokenizer:
    def fill_str_parts(self, List[Tuple[int, int, str, str]] parts, str sentence) -> List[Tuple[str, str]]:
        cdef List[Tuple[str, str]] res = []
        parts.sort()
        cdef int start = 0
        for part_start, part_end, part, part_type in parts:
            if start < part_start:
                res.append((sentence[start:part_start], "str"))
            res.append((part, part_type))
            start = part_end
        if start < len(sentence):
            res.append((sentence[start:], "str"))
        return res

    def tokenize(self, str sentence) -> List[Tuple[str, str]]:
        cdef List[Tuple[int, int, str, str]] parts = []
        cdef List[str] group_names = ["digits_with_unit", "digits_zh_with_unit", "non_word"]
        for match in PT_DIGITS_UNITS_AND_NON_WORDS.finditer(sentence):
            for name in group_names:
                if match.group(name):
                    parts.append((match.start(), match.end(), match.group(name), name))
                    break
        res = self.fill_str_parts(parts, sentence)
        return res


cdef class SentencePieceModelTokenizer:
    cdef object sp
    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor(model_file=model_path)

    def tokenize(self, str sentence) -> List[str]:
        cdef List[str] tokens = self.sp.EncodeAsPieces(sentence)
        return tokens


# post-tokenizer regex
cdef str CH_ATOZ = r"a-zA-Z"
cdef unicode RE_ATOZ = f"[{CH_ATOZ}]+"

cdef unicode RE_ATOZ_DIGITS_NUMBER = f"({RE_ATOZ}|{RE_DIGITS_NUMBER})"

cdef unicode RE_DIGITS_UNITS_TAIL = f"[{CH_DIGIT_PREFIX}]?{RE_DIGITS_AND_DOTS}"
cdef unicode RE_DIGITS_UNITS_HEAD = f"{RE_UNITS_ALL}"

cdef unicode RE_DIGITS_ZH_UNITS_HEAD = (
    f"({RE_DIGITS_ZH}{RE_UNITS_ALL}|{RE_DIGITS_ZH}|{RE_UNITS_ALL})"
)

cdef str RE_WORD_EXCPET_ATOZ_OR_DIGITS = r"[^\da-zA-Z]+"
cdef unicode RE_ATOZ_DIGITS_WORD = f"(?P<atoz>{RE_ATOZ})|(?P<digits_with_unit>{RE_DIGITS_ALL})|(?P<digits_number>{RE_DIGITS_NUMBER})|(?P<digits_zh_with_unit>{RE_DIGITS_ZH_WITH_UNIT})|(?P<word>{RE_WORD_EXCPET_ATOZ_OR_DIGITS})"

cdef object PT_ATOZ = re.compile(RE_ATOZ)
cdef object PT_DIGITS_NUMBER = re.compile(RE_DIGITS_NUMBER)
cdef object PT_ATOZ_DIGITS_NUMBER = re.compile(RE_ATOZ_DIGITS_NUMBER)
cdef object PT_ATOZ_DIGITS_WORD = re.compile(RE_ATOZ_DIGITS_WORD)

cdef unicode RE_ATOZ_CONCAT = f"{RE_ATOZ}(?:<SPT>{RE_ATOZ})+"
cdef unicode RE_DIGITS_UNITS_CONCAT = f"{RE_DIGITS_UNITS_TAIL}<SPT>{RE_DIGITS_UNITS_HEAD}"
cdef unicode RE_DIGITS_NUMBER_CONCAT = f"{RE_DIGITS_NUMBER}(?:<SPT>{RE_DIGITS_NUMBER})+"
cdef unicode RE_DIGITS_ZH_UNITS_CONCAT = f"{RE_DIGITS_ZH}<SPT>{RE_DIGITS_ZH_UNITS_HEAD}(?:<SPT>|$)"

cdef unicode RE_CONCAT = f"(?P<atoz>{RE_ATOZ_CONCAT})|(?P<digits_number>{RE_DIGITS_NUMBER_CONCAT})"
cdef object PT_CONCAT = re.compile(RE_CONCAT)

cdef unicode RE_SINGLE_CJK_HEAD = f"(^|<SPT>)[{CH_CJK}]"
cdef unicode RE_SINGLE_CJK_TAIL = f"[{CH_CJK}](<SPT>|$)"
cdef unicode RE_DIGITS_ZH_WITH_UNIT_SPT = f"<SPT>{RE_DIGITS_ZH_WITH_UNIT}<SPT>"
cdef unicode RE_DIGITS_ZH_WITH_UNIT_SINGLE_CHAR = f"{RE_SINGLE_CJK_HEAD}{RE_DIGITS_ZH_WITH_UNIT_SPT}{RE_SINGLE_CJK_TAIL}|{RE_SINGLE_CJK_HEAD}{RE_DIGITS_ZH_WITH_UNIT_SPT}|{RE_DIGITS_ZH_WITH_UNIT_SPT}{RE_SINGLE_CJK_TAIL}"

cdef object PT_DIGITS_ZH_WITH_UNIT_SINGLE_CHAR = re.compile(RE_DIGITS_ZH_WITH_UNIT_SINGLE_CHAR)


class SentencePostTokenizer:
    def concat_same_types(self, List[str] tokens) -> List[str]:
        cdef str spt_str = "<SPT>".join(tokens)
        cdef str res_str = spt_str
        cdef str new_value
        for match in PT_CONCAT.finditer(spt_str):
            value = match.group()
            new_value = value.replace("<SPT>", "")
            res_str = res_str.replace(value, new_value)
        return res_str.split("<SPT>")

    def concat_digits_zh_units_single_chars(
        self, List[str] tokens, model_tokenizer
    ) -> List[str]:
        cdef str spt_str = "<SPT>".join(tokens)
        cdef str res_str = spt_str
        cdef str value
        cdef str raw_value
        cdef List[str] new_tokens
        cdef str new_value
        for match in PT_DIGITS_ZH_WITH_UNIT_SINGLE_CHAR.finditer(spt_str):
            value = match.group()
            raw_value = value.replace("<SPT>", "")
            new_tokens = model_tokenizer.tokenize(raw_value)
            if len(new_tokens) == 1:
                new_value = "<SPT>" + "".join(new_tokens) + "<SPT>"
                res_str = res_str.replace(value, new_value)
        return res_str.split("<SPT>")


class SentenceFullTokenizer:
    def __init__(
        self,
        model_path: Union[Path, str],
        drop_non_word: bool = False,
        verbose: bool = False,
    ):
        self.model_path = model_path
        self.drop_non_word = drop_non_word
        self.verbose = verbose
        self.pre_tokenizer = SentencePreTokenizer()
        self.post_tokenizer = SentencePostTokenizer()
        self.model_tokenizer = SentencePieceModelTokenizer(self.model_path)

    def tokenize_parts(self, List[Tuple[str, str]] parts) -> List[str]:
        cdef list res = []
        for token, token_type in parts:
            if token_type == "str":
                segs = self.model_tokenizer.tokenize(token)
                res.extend(segs)
            else:
                res.append(token)
        return res

    def stringify(self, tokens: list[str]) -> unicode:
        return f"{logstr.note('_')}".join(tokens)

    def parts_to_tokens(self, parts: List[Tuple[str, str]]) -> List[str]:
        cdef list tokens = []
        cdef str token
        if not self.drop_non_word:
            tokens = [part for part, type in parts]
        else:
            for part, type in parts:
                token = PT_NON_WORD.sub("", part)
                if token:
                    tokens.append(token)
        return tokens

    def tokenize(self, str sentence) -> List[str]:
        sentence = sentence.lower()
        cdef list parts = self.pre_tokenizer.tokenize(sentence)
        cdef list tokens = self.tokenize_parts(parts)
        tokens = self.post_tokenizer.concat_same_types(tokens)
        tokens = self.post_tokenizer.concat_digits_zh_units_single_chars(tokens, self.model_tokenizer)
        return tokens