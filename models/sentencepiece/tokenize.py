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
    rf"[{CH_DIGIT_PREFIX}]?{RE_DIGITS_AND_DOTS}{RE_UNITS_ALL}"
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
        Output: list of tuple (start:int, end:int, part:str, type:str)
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
        parts = [(part, type) for _, _, part, type in parts]
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

    def split_atoz_and_digits(self, token: str) -> list[str]:
        """Examples:
        - "abc100" -> ["abc", "100"]
        - "5a10" -> ["5", "a", "10"]
        - "1000" -> ["1000"]
        """
        parts = []
        group_names = ["atoz", "digits", "digits_zh_with_unit", "word"]
        for match in self.PT_ATOZ_DIGITS_WORD.finditer(token):
            for name in group_names:
                if match.group(name):
                    parts.append(match.group(name))
        return parts

    def merge_same_types(self, tokens: list[str]) -> list[tuple[str, str]]:
        """Examples:
        - [hb,k0,8,是] -> [hbk,08,是]
        - [2024lbw,nb] -> [2024,lbwnb]
        - [2024lbw, ,nb]-> [2024lbw, ,nb] # no merge as adjacent token type not same
        - [abc100,0, def123] -> [abc,1000,def123]
        - [5a10,0d, def123d,?] -> [5a,100,ddef,123,d,?]
        """
        merges = []
        for i in range(len(tokens)):
            token = tokens[i]
            if token == "":
                continue
            if not merges:
                merges.append((token, "raw"))
                continue
            last_token = merges[-1][0]
            if self.is_same_type(last_token, token):
                merges[-1] = (last_token + token, "merged")
            else:
                merges.append((tokens[i], "raw"))
        merged_tokens = []
        for token, type in merges:
            if type == "raw":
                merged_tokens.append(token)
            else:
                split_tokens = self.split_atoz_and_digits(token)
                merged_tokens.extend(split_tokens)
        return merged_tokens

    def transform(self, tokens: list[str]) -> list[str]:
        tokens = self.merge_same_types(tokens)
        return tokens


class SentenceFullTokenizer:
    def __init__(self, model_path: Union[Path, str], verbose: bool = False):
        self.model_path = model_path
        self.verbose = verbose
        self.pre_tokenizer = SentencePreTokenizer()
        self.post_tokenizer = SentencePostTokenizer()
        self.model_tokenizer = SentencePieceModelTokenizer(model_path)

    def tokenize_parts(self, parts: list[tuple]) -> list[str]:
        tokens = []
        for part, type in parts:
            if type == "str":
                tokens.extend(self.model_tokenizer.tokenize(part))
            else:
                tokens.append(part)
        return tokens

    def stringify(self, tokens: list[str]) -> str:
        tokens_str = f"{logstr.note('_')}".join(tokens)
        return tokens_str

    def tokenize(self, sentence: str) -> list[str]:
        sentence = sentence.lower()
        parts = self.pre_tokenizer.tokenize(sentence)
        tokens = self.tokenize_parts(parts)
        tokens = self.post_tokenizer.transform(tokens)
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
