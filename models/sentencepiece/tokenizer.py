import re
import sentencepiece as spm

from pathlib import Path
from tclogger import logger, logstr
from typing import Union

from configs.envs import SP_MERGED_MODEL_PATH
from datasets.videos.convert import CH_CJK, CH_AB, CH_DIGIT_PREFIX, RE_UNITS_ALL

"""Naming conventions by typings:
- tokenize: str -> list[str]
- tokenize_parts: list[tuple] -> list[str]
- transform: list[str] -> list[str]
"""

# pre-tokenizer regex
CH_LB = r"\(\[\{"
CH_RB = r"\)\]\}"

CH_DIGIT_ZH = r"〇零一二两三四五六七八九十"
CH_DIGIT_ZH_MUL = r"十百千万亿"

RE_DIGITS_PURE = r"\d+"
RE_DOT_DIGITS = r"\d*(\.\d+)+"
RE_COMMA_DIGITS = r"\d+(\,\d+)+"
RE_DASHED_DIGITS = r"\d+(\-\d+)+"
RE_PERCENT = r"[%‰]*"
RE_DIGITS_AND_DOTS = rf"{RE_DOT_DIGITS}|{RE_DIGITS_PURE}"
RE_DIGITS_NUMBER = rf"(({RE_DOT_DIGITS}|{RE_COMMA_DIGITS}|{RE_DASHED_DIGITS}|{RE_DIGITS_PURE}){RE_PERCENT})"
RE_DIGITS_NUMBER_WITH_PREFIX_AND_UNIT = (
    rf"([{CH_DIGIT_PREFIX}]\s*)?{RE_DIGITS_NUMBER}\s*{RE_UNITS_ALL}"
)
RE_DIGITS_WITH_DOTS_AND_BRS = (
    rf"\[{RE_DIGITS_AND_DOTS}\]|\({RE_DIGITS_AND_DOTS}\)|{{{RE_DIGITS_AND_DOTS}}}"
)
RE_DIGITS_ALL = rf"(?:{RE_DIGITS_NUMBER_WITH_PREFIX_AND_UNIT}|{RE_DIGITS_WITH_DOTS_AND_BRS}|{RE_DIGITS_NUMBER})"

RE_NOT_DIGIT_DOT = r"\.(?!\d)"
RE_NON_WORD = rf"[^{CH_CJK}〇{CH_AB}\-\.%‰{CH_LB}{CH_RB}]+|{RE_NOT_DIGIT_DOT}"

RE_DIGITS_ZH = (
    rf"(([{CH_DIGIT_ZH}][{CH_DIGIT_ZH_MUL}])+[{CH_DIGIT_ZH}]?|[{CH_DIGIT_ZH}]+)"
)
RE_DIGITS_ZH_WITH_UNIT = rf"[{CH_DIGIT_PREFIX}]?{RE_DIGITS_ZH}{RE_UNITS_ALL}"

RE_DASHED_ATOZ_AND_DIGITS = rf"[a-z0-9]+(\-[a-z0-9]+)+"

RE_DIGITS_UNITS_AND_NON_WORD = rf"(?P<dashed_atoz_and_digits>{RE_DASHED_ATOZ_AND_DIGITS}{RE_UNITS_ALL}?)|(?P<digits_with_unit>{RE_DIGITS_NUMBER_WITH_PREFIX_AND_UNIT})|(?P<digits_zh_with_unit>{RE_DIGITS_ZH_WITH_UNIT})|(?P<non_word>{RE_NON_WORD})"

PT_NON_WORD = re.compile(RE_NON_WORD)
PT_DIGITS_ALL = re.compile(RE_DIGITS_ALL)
PT_DIGITS_ZH_WITH_UNIT = re.compile(RE_DIGITS_ZH_WITH_UNIT)
PT_DIGITS_UNITS_AND_NON_WORDS = re.compile(RE_DIGITS_UNITS_AND_NON_WORD)


class SentencePreTokenizer:
    def fill_str_parts(
        self, parts: list[tuple[int, int, str, str]], sentence: str
    ) -> list[tuple[str, str]]:
        """Fill str parts between non-word and digits
        Input: list of tuple: [(start:int, end:int, part:str, type:str), ...]
        Output: list of tuple: [(part:str, type:str), ...]
        """
        res: list[tuple[str, str]] = []
        parts.sort()
        start = 0
        for part_start, part_end, part, part_type in parts:
            if start < part_start:
                res.append((sentence[start:part_start], "str"))
            res.append((part, part_type))
            start = part_end
        if start < len(sentence):
            res.append((sentence[start:], "str"))
        return res

    def tokenize(self, sentence: str) -> list[tuple[str, str]]:
        """Split sentence by multiple parts, non-word, digits and non-digits
        Output: list of tuple (part:str, type:str)
        - part: str: digits, non-word, other string
        - type: "digits_with_unit", "digits_zh_with_unit", "non_word"
        """
        parts = []
        group_names = [
            "dashed_atoz_and_digits",
            "digits_with_unit",
            "digits_zh_with_unit",
            "non_word",
        ]
        for match in PT_DIGITS_UNITS_AND_NON_WORDS.finditer(sentence):
            for name in group_names:
                if match.group(name):
                    parts.append((match.start(), match.end(), match.group(name), name))
                    break
        res = self.fill_str_parts(parts, sentence)
        return res


class SentencePieceModelTokenizer:
    def __init__(self, model_path: Union[Path, str]):
        self.model_file = str(model_path)
        self.sp = spm.SentencePieceProcessor(model_file=self.model_file)

    def tokenize(self, sentence: str) -> list[str]:
        tokens = self.sp.EncodeAsPieces(sentence)
        return tokens


# post-tokenizer regex
CH_ATOZ = r"a-zA-Z"
RE_ATOZ = rf"[{CH_ATOZ}]+"

RE_ATOZ_DIGITS_NUMBER = rf"({RE_ATOZ}|{RE_DIGITS_NUMBER})"

RE_DIGITS_UNITS_TAIL = rf"[{CH_DIGIT_PREFIX}]?{RE_DIGITS_NUMBER}"
RE_DIGITS_UNITS_HEAD = rf"{RE_UNITS_ALL}"

RE_DIGITS_ZH_UNITS_HEAD = (
    rf"({RE_DIGITS_ZH}{RE_UNITS_ALL}|{RE_DIGITS_ZH}|{RE_UNITS_ALL})"
)

RE_WORD_EXCPET_ATOZ_OR_DIGITS = r"[^\da-zA-Z]+"
RE_ATOZ_DIGITS_WORD = rf"(?P<atoz>{RE_ATOZ})|(?P<digits_with_unit>{RE_DIGITS_ALL})|(?P<digits_number>{RE_DIGITS_NUMBER})|(?P<digits_zh_with_unit>{RE_DIGITS_ZH_WITH_UNIT})|(?P<word>{RE_WORD_EXCPET_ATOZ_OR_DIGITS})"

PT_ATOZ = re.compile(RE_ATOZ)
PT_DIGITS_NUMBER = re.compile(RE_DIGITS_NUMBER)
PT_ATOZ_DIGITS_NUMBER = re.compile(RE_ATOZ_DIGITS_NUMBER)
PT_ATOZ_DIGITS_WORD = re.compile(RE_ATOZ_DIGITS_WORD)

RE_ATOZ_CONCAT = rf"{RE_ATOZ}(?:<SPT>{RE_ATOZ})+"
RE_DIGITS_UNITS_CONCAT = rf"{RE_DIGITS_UNITS_TAIL}<SPT>{RE_DIGITS_UNITS_HEAD}"
RE_DIGITS_NUMBER_CONCAT = rf"{RE_DIGITS_NUMBER}(?:<SPT>{RE_DIGITS_NUMBER})+"
RE_DIGITS_ZH_UNITS_CONCAT = rf"{RE_DIGITS_ZH}<SPT>{RE_DIGITS_ZH_UNITS_HEAD}(?:<SPT>|$)"

RE_CONCAT = rf"(?P<atoz>{RE_ATOZ_CONCAT})|(?P<digits_number>{RE_DIGITS_NUMBER_CONCAT})"
PT_CONCAT = re.compile(RE_CONCAT)

RE_SINGLE_CJK_HEAD = rf"(^|<SPT>)[{CH_CJK}]"
RE_SINGLE_CJK_TAIL = rf"[{CH_CJK}](<SPT>|$)"
RE_DIGITS_ZH_WITH_UNIT_SPT = rf"<SPT>{RE_DIGITS_ZH_WITH_UNIT}<SPT>"
RE_DIGITS_ZH_WITH_UNIT_SINGLE_CHAR = rf"{RE_SINGLE_CJK_HEAD}{RE_DIGITS_ZH_WITH_UNIT_SPT}{RE_SINGLE_CJK_TAIL}|{RE_SINGLE_CJK_HEAD}{RE_DIGITS_ZH_WITH_UNIT_SPT}|{RE_DIGITS_ZH_WITH_UNIT_SPT}{RE_SINGLE_CJK_TAIL}"

PT_DIGITS_ZH_WITH_UNIT_SINGLE_CHAR = re.compile(RE_DIGITS_ZH_WITH_UNIT_SINGLE_CHAR)


class SentencePostTokenizer:
    def is_same_type(self, a: str, b: str) -> tuple[bool, str]:
        ab = f"{a}<SPT>{b}"
        match = PT_CONCAT.match(ab)
        if match:
            for name, value in match.groupdict().items():
                if value:
                    return True, name
        return False, "word"

    def split_atoz_and_digits(self, token: str) -> list[tuple[str, str]]:
        parts: list[tuple[str, str]] = []
        group_names = [
            "atoz",
            "digits_with_unit",
            "digits_number",
            "digits_zh_with_unit",
            "word",
        ]
        for match in PT_ATOZ_DIGITS_WORD.finditer(token):
            for name in group_names:
                if match.group(name):
                    parts.append((match.group(name), name))
        return parts

    def concat_same_types(self, tokens: list[str]) -> list[str]:
        """Examples:
        - [hb,k0,8,是] -> [hbk08,是]
        - [2024lbw,nb] -> [2024lbwnb]
        - [2024lbw, ,nb]-> [2024,lbw, ,nb]
        - [abc100,0,def123] -> [abc1000,def123]
        - [5a10,0d,def123d,?] -> [5a100,ddef123,d,?]
        """
        spt_str = "<SPT>".join(tokens)
        res_str = spt_str
        for match in PT_CONCAT.finditer(spt_str):
            value = match.group()
            new_value = value.replace("<SPT>", "")
            res_str = res_str.replace(value, new_value)
        return res_str.split("<SPT>")

    def get_token_type(self, token: str) -> str:
        for pattern, name in [
            (PT_ATOZ, "atoz"),
            (PT_DIGITS_NUMBER, "digits_number"),
            (PT_ATOZ_DIGITS_NUMBER, "atoz_digits_number"),
            (PT_DIGITS_ZH_WITH_UNIT, "digits_zh_with_unit"),
        ]:
            if pattern.fullmatch(token):
                return name
        return "raw"

    def merge_atoz_and_digits(self, tokens: list[str]) -> list[tuple[str, str]]:
        merges: list[tuple[str, str]] = []
        for token in tokens:
            token_type = self.get_token_type(token)
            if not merges:
                merges.append((token, token_type))
                continue
            last_token, last_token_type = merges[-1]
            if token_type in ["digits_number"] and last_token_type in [
                "atoz",
                "digits_number",
                "atoz_digits_number",
            ]:
                merges[-1] = (last_token + token, "atoz_digits_number")
            else:
                merges.append((token, token_type))
        return merges

    def concat_digits_zh_units_single_chars(
        self, tokens: list[str], model_tokenizer: SentencePieceModelTokenizer
    ) -> list[str]:
        spt_str = "<SPT>".join(tokens)
        res_str = spt_str
        for match in PT_DIGITS_ZH_WITH_UNIT_SINGLE_CHAR.finditer(spt_str):
            value = match.group()
            raw_value = value.replace("<SPT>", "")
            new_tokens = model_tokenizer.tokenize(raw_value)
            if len(new_tokens) == 1:
                new_value = "<SPT>" + "".join(new_tokens) + "<SPT>"
                res_str = res_str.replace(value, new_value)
        concat_tokens = res_str.split("<SPT>")
        return concat_tokens


class SentenceFullTokenizer:
    def __init__(
        self,
        model_path: Union[Path, str],
        drop_non_word: bool = False,
        drop_whitespace: bool = False,
        verbose: bool = False,
    ):
        self.model_path = model_path
        self.drop_non_word = drop_non_word
        self.drop_whitespace = drop_whitespace
        self.verbose = verbose
        self.pre_tokenizer = SentencePreTokenizer()
        self.post_tokenizer = SentencePostTokenizer()
        self.model_tokenizer = SentencePieceModelTokenizer(model_path)

    def tokenize_parts(self, parts: list[tuple]) -> list[str]:
        res: list[str] = []
        for token, token_type in parts:
            if token_type == "str":
                segs = self.model_tokenizer.tokenize(token)
                res.extend(segs)
            else:
                res.append(token)
        return res

    def stringify(self, tokens: list[str]) -> str:
        tokens_str = f"{logstr.note('_')}".join(tokens)
        return tokens_str

    def parts_to_tokens(self, parts: list[tuple]) -> list[str]:
        if not self.drop_non_word and not self.drop_whitespace:
            tokens = [part for part, type in parts]
        else:
            tokens = []
            for part, type in parts:
                if self.drop_non_word:
                    token = PT_NON_WORD.sub("", part)
                if self.drop_whitespace:
                    # token = token.strip()
                    token = token.replace(" ", "")
                if token:
                    tokens.append(token)
        return tokens

    def remove_non_words(self, tokens: list[str]):
        res = []
        for token in tokens:
            token = PT_NON_WORD.sub("", token)
            if token:
                res.append(token)
        return res

    def remove_whitespaces(self, tokens: list[str]):
        return [token for token in tokens if token.strip()]

    def clean_tokens(self, tokens: list[str]):
        if self.drop_non_word:
            tokens = self.remove_non_words(tokens)
        if self.drop_whitespace:
            tokens = self.remove_whitespaces(tokens)
        return tokens

    def tokenize(self, sentence: str) -> list[str]:
        sentence = sentence.lower()
        parts = self.pre_tokenizer.tokenize(sentence)
        tokens = self.tokenize_parts(parts)
        tokens = self.post_tokenizer.concat_same_types(tokens)
        # parts = self.post_tokenizer.merge_atoz_and_digits(tokens)
        tokens = self.post_tokenizer.concat_digits_zh_units_single_chars(
            tokens, self.model_tokenizer
        )
        tokens = self.clean_tokens(tokens)
        return tokens


if __name__ == "__main__":
    from tclogger import logger
    from models.sentencepiece.test import TEST_TOKENS, TEST_SENTENCES

    sentence = TEST_TOKENS
    logger.note("> Pre-Tokenizing ...")
    pre_tokenizer = SentencePreTokenizer()
    parts = pre_tokenizer.tokenize(sentence)
    logger.success(parts)

    logger.note("> Full-Tokenizing ...")
    tokenizer = SentenceFullTokenizer(
        SP_MERGED_MODEL_PATH, drop_non_word=True, verbose=True
    )
    for sentence in TEST_SENTENCES:
        tokens = tokenizer.tokenize(sentence)
        pretty_tokens = tokenizer.stringify(tokens)
        logger.mesg(f"  * {pretty_tokens}")

    # python -m models.sentencepiece.tokenizer
