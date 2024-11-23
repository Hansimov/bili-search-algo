import re
import sentencepiece as spm

from pathlib import Path
from tclogger import logger, logstr
from typing import Union

from models.sentencepiece.convert import CH_CJK, CH_AB, CH_DIGIT_PREFIX, RE_UNITS_ALL

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
RE_DOT_DIGITS = r"\d*\.\d+"
RE_DIGITS_AND_DOTS = r"[\d\.]+(?<!\.)"
RE_DIGITS_NUMBER = rf"({RE_DOT_DIGITS}|{RE_DIGITS_AND_DOTS})"
RE_DIGITS_AND_DOTS_WITH_PREFIX_AND_UNIT = (
    rf"[{CH_DIGIT_PREFIX}]?\s*{RE_DIGITS_AND_DOTS}\s*{RE_UNITS_ALL}"
)
RE_DIGITS_WITH_DOTS_AND_BRS = (
    rf"\[{RE_DIGITS_AND_DOTS}\]|\({RE_DIGITS_AND_DOTS}\)|{{{RE_DIGITS_AND_DOTS}}}"
)
RE_DIGITS_ALL = rf"(?:{RE_DIGITS_AND_DOTS_WITH_PREFIX_AND_UNIT}|{RE_DIGITS_WITH_DOTS_AND_BRS}|{RE_DIGITS_NUMBER})"

RE_NON_WORD = rf"[^{CH_CJK}〇{CH_AB}\.{CH_LB}{CH_RB}]+"

RE_DIGITS_ZH = (
    rf"(([{CH_DIGIT_ZH}][{CH_DIGIT_ZH_MUL}])+[{CH_DIGIT_ZH}]?|[{CH_DIGIT_ZH}]+)"
)
RE_DIGITS_ZH_WITH_UNIT = rf"[{CH_DIGIT_PREFIX}]?{RE_DIGITS_ZH}{RE_UNITS_ALL}"

RE_DIGITS_UNITS_AND_NON_WORD = rf"(?P<digits_with_unit>{RE_DIGITS_AND_DOTS_WITH_PREFIX_AND_UNIT})|(?P<digits_zh_with_unit>{RE_DIGITS_ZH_WITH_UNIT})|(?P<non_word>{RE_NON_WORD})"

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
        parts.sort()
        start = 0
        for i in range(len(parts)):
            end, _, part, type = parts[i]
            if start < end:
                parts.append((start, end, sentence[start:end], "str"))
            start = end + len(part)
        if start < len(sentence):
            parts.append((start, len(sentence), sentence[start:], "str"))
        parts.sort()
        parts = [(part, type) for _, _, part, type in parts]
        return parts

    def tokenize(self, sentence: str) -> list[tuple]:
        """Split sentence by multiple parts, non-word, digits and non-digits
        Output: list of tuple (part:str, type:str)
        - part: str: digits, non-word, other string
        - type: "digits_with_unit", "digits_zh_with_unit", "non_word"
        """
        parts = []
        group_names = ["digits_with_unit", "digits_zh_with_unit", "non_word"]
        for match in PT_DIGITS_UNITS_AND_NON_WORDS.finditer(sentence):
            for name in group_names:
                if match.group(name):
                    parts.append((match.start(), match.end(), match.group(name), name))
        parts = self.fill_str_parts(parts, sentence)
        return parts


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
RE_ATOZ_HEAD = rf"^{RE_ATOZ}"
RE_ATOZ_TAIL = rf"{RE_ATOZ}$"

RE_DIGITS_NUMBER_TAIL = rf"{RE_DIGITS_NUMBER}$"
RE_DIGITS_NUMBER_HEAD = rf"^{RE_DIGITS_NUMBER}"

RE_ATOZ_DIGITS_NUMBER = rf"({RE_ATOZ}|{RE_DIGITS_NUMBER})"

RE_DIGITS_UNITS_TAIL = rf"[{CH_DIGIT_PREFIX}]?{RE_DIGITS_AND_DOTS}{RE_UNITS_ALL}?"
RE_DIGITS_UNITS_HEAD = (
    rf"{RE_DIGITS_AND_DOTS}?{RE_UNITS_ALL}|{RE_DIGITS_AND_DOTS}{RE_UNITS_ALL}?"
)

RE_DIGITS_ZH_UNITS_TAIL = rf"{RE_DIGITS_ZH}$"
RE_DIGITS_ZH_UNITS_HEAD = (
    rf"^({RE_DIGITS_ZH}{RE_UNITS_ALL}|{RE_DIGITS_ZH}|{RE_UNITS_ALL})$"
)

RE_WORD_EXCPET_ATOZ_OR_DIGITS = r"[^\da-zA-Z]+"
RE_ATOZ_DIGITS_WORD = rf"(?P<atoz>{RE_ATOZ})|(?P<digits_with_unit>{RE_DIGITS_ALL})|(?P<digits_number>{RE_DIGITS_NUMBER})|(?P<digits_zh_with_unit>{RE_DIGITS_ZH_WITH_UNIT})|(?P<word>{RE_WORD_EXCPET_ATOZ_OR_DIGITS})"

PT_ATOZ = re.compile(RE_ATOZ)
PT_DIGITS_NUMBER = re.compile(RE_DIGITS_NUMBER)
PT_ATOZ_DIGITS_NUMBER = re.compile(RE_ATOZ_DIGITS_NUMBER)
PT_DIGITS_NUMBER_TAIL = re.compile(RE_DIGITS_NUMBER_TAIL)
PT_DIGITS_NUMBER_HEAD = re.compile(RE_DIGITS_NUMBER_HEAD)
PT_DIGITS_UNITS_TAIL = re.compile(RE_DIGITS_UNITS_TAIL)
PT_DIGITS_UNITS_HEAD = re.compile(RE_DIGITS_UNITS_HEAD)
PT_DIGITS_ZH_UNITS_TAIL = re.compile(RE_DIGITS_ZH_UNITS_TAIL)
PT_DIGITS_ZH_UNITS_HEAD = re.compile(RE_DIGITS_ZH_UNITS_HEAD)
PT_ATOZ_TAIL = re.compile(RE_ATOZ_TAIL)
PT_ATOZ_HEAD = re.compile(RE_ATOZ_HEAD)
PT_ATOZ_DIGITS_WORD = re.compile(RE_ATOZ_DIGITS_WORD)


class SentencePostTokenizer:
    def is_same_type(self, a: str, b: str) -> tuple[bool, str]:
        if PT_ATOZ_TAIL.match(a) and PT_ATOZ_HEAD.match(b):
            return True, "atoz"
        if PT_DIGITS_UNITS_TAIL.match(a) and PT_DIGITS_UNITS_HEAD.match(b):
            return True, "digits_with_unit"
        if PT_DIGITS_NUMBER_TAIL.match(a) and PT_DIGITS_NUMBER_HEAD.match(b):
            return True, "digits_number"
        if PT_DIGITS_ZH_UNITS_TAIL.match(a) and PT_DIGITS_ZH_UNITS_HEAD.match(b):
            return True, "digits_zh_with_unit"
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

    def concat_same_types(self, parts: list[tuple[str, str]]) -> list[tuple[str, str]]:
        """Examples:
        - [hb,k0,8,是] -> [hbk,08,是]
        - [2024lbw,nb] -> [2024,lbwnb]
        - [2024lbw, ,nb]-> [2024lbw, ,nb] # no merge as adjacent token type not same
        - [abc100,0, def123] -> [abc,1000,def123]
        - [5a10,0d, def123d,?] -> [5a,100,ddef,123,d,?]
        """
        merges: list[tuple[str, str]] = []
        for i in range(len(parts)):
            token, token_type = parts[i]
            if token == "":
                continue
            if not merges:
                merges.append((token, "raw"))
                continue
            last_token = merges[-1][0]
            is_same, same_type = self.is_same_type(last_token, token)
            if is_same:
                merges[-1] = (last_token + token, "merged")
            else:
                merges.append((token, "raw"))

        splits: list[tuple[str, str]] = []
        for token, type in merges:
            if type == "merged":
                split_tokens = self.split_atoz_and_digits(token)
                splits.extend(split_tokens)
            else:
                splits.append((token, type))
        return splits

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

    def merge_atoz_and_digits(
        self, parts: list[tuple[str, str]]
    ) -> list[tuple[str, str]]:
        merges: list[tuple[str, str]] = []
        for part in parts:
            token = part[0]
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

    def merge_single_char_around_digits_zh_with_units(
        self, parts: list[tuple[str, str]], model_tokenizer: SentencePieceModelTokenizer
    ) -> list[tuple[str, str]]:
        merges: list[tuple[str, str]] = []
        for part in parts:
            token, token_type = part
            if not merges:
                merges.append((token, token_type))
                continue
            last_token, last_token_type = merges[-1]
            if (
                len(last_token) == 1
                and last_token_type == "raw"
                and token_type == "digits_zh_with_unit"
            ) or (
                len(token) == 1
                and token_type == "raw"
                and last_token_type
                in ["digits_zh_with_unit", "raw_digits_zh_with_unit"]
            ):
                new_tokens = model_tokenizer.tokenize(last_token + token)
                if len(new_tokens) == 1:
                    new_tokens_str = "".join(new_tokens)
                    if token_type == "raw":
                        new_token_type = "digits_zh_with_unit_raw"
                    else:
                        new_token_type = "raw_digits_zh_with_unit"
                    merges[-1] = (new_tokens_str, new_token_type)
                else:
                    merges.append((token, token_type))
            else:
                merges.append((token, token_type))

        return merges


class SentenceFullTokenizer:
    def __init__(self, model_path: Union[Path, str], verbose: bool = False):
        self.model_path = model_path
        self.verbose = verbose
        self.pre_tokenizer = SentencePreTokenizer()
        self.post_tokenizer = SentencePostTokenizer()
        self.model_tokenizer = SentencePieceModelTokenizer(model_path)

    def tokenize_parts(self, parts: list[tuple]) -> list[str]:
        new_parts = []
        for part, type in parts:
            if type == "str":
                tokens = self.model_tokenizer.tokenize(part)
                new_parts.extend([(token, "token") for token in tokens])
            else:
                new_parts.append((part, type))
        return new_parts

    def stringify(self, tokens: list[str]) -> str:
        tokens_str = f"{logstr.note('_')}".join(tokens)
        return tokens_str

    def tokenize(self, sentence: str) -> list[str]:
        sentence = sentence.lower()
        parts = self.pre_tokenizer.tokenize(sentence)
        parts = self.tokenize_parts(parts)
        parts = self.post_tokenizer.concat_same_types(parts)
        parts = self.post_tokenizer.merge_atoz_and_digits(parts)
        parts = self.post_tokenizer.merge_single_char_around_digits_zh_with_units(
            parts, self.model_tokenizer
        )
        tokens = [token for token, type in parts]
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
    model_path = str(Path(__file__).parents[2] / "sp_10m_10k.model")
    tokenizer = SentenceFullTokenizer(model_path, verbose=True)
    for sentence in TEST_SENTENCES:
        tokens = tokenizer.tokenize(sentence)
        pretty_tokens = tokenizer.stringify(tokens)
        logger.mesg(f"  * {pretty_tokens}")

    # python -m models.sentencepiece.tokenize
