import re
import sentencepiece as spm

from pathlib import Path
from tclogger import logger, logstr
from typing import Union

from models.sentencepiece.convert import DocSentenceConverter

"""Naming conventions by typings:
- tokenize: str -> list[str]
- tokenize_parts: list[tuple] -> list[str]
- transform: list[str] -> list[str]
"""


class SentencePreTokenizer:
    RE_CJK = DocSentenceConverter.RE_CJK
    RE_EN = DocSentenceConverter.RE_EN
    RE_NON_WORD = rf"[^{RE_CJK}{RE_EN}\.]+"

    RE_DIGIT_PREFIX = DocSentenceConverter.RE_DIGIT_PREFIX
    RE_UNITS = DocSentenceConverter.RE_UNITS
    RE_DIGITS = r"[\d\.]+(?<!\.)"
    RE_DIGIT_DOTS = r"\d*\.\d+"
    RE_DIGIT_UNIT = rf"[{RE_DIGIT_PREFIX}]?{RE_DIGITS}{RE_UNITS}"
    RE_DIGITS_ALL = rf"(?:{RE_DIGIT_UNIT}|{RE_DIGIT_DOTS})"

    PT_NON_WORD = re.compile(RE_NON_WORD)
    PT_DIGITS_ALL = re.compile(RE_DIGITS_ALL)

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
        - part: str: digits, non-word, normal string
        - type: "non_word", "digits", "str"
        """
        parts = []
        for match in self.PT_DIGITS_ALL.finditer(sentence):
            parts.append((match.start(), match.end(), match.group(), "digits"))
        for match in self.PT_NON_WORD.finditer(sentence):
            parts.append((match.start(), match.end(), match.group(), "non_word"))
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


class SentencePostTokenizer:
    RE_ATOZ = r"[a-zA-Z]+"
    RE_ATOZ_HEAD = rf"^{RE_ATOZ}"
    RE_ATOZ_TAIL = rf"{RE_ATOZ}$"

    RE_DIGITS = SentencePreTokenizer.RE_DIGITS
    RE_DIGITS_HEAD = rf"^{RE_DIGITS}"
    RE_DIGITS_TAIL = rf"{RE_DIGITS}$"

    RE_WORD = r"[^\da-zA-Z]+"
    RE_ATOZ_DIGITS = rf"(?P<atoz>{RE_ATOZ})|(?P<digits>{RE_DIGITS})|(?P<word>{RE_WORD})"

    PT_ATOZ = re.compile(RE_ATOZ)
    PT_ATOZ_HEAD = re.compile(RE_ATOZ_HEAD)
    PT_ATOZ_TAIL = re.compile(RE_ATOZ_TAIL)

    PT_DIGITS = re.compile(RE_DIGITS)
    PT_DIGITS_HEAD = re.compile(RE_DIGITS_HEAD)
    PT_DIGITS_TAIL = re.compile(RE_DIGITS_TAIL)

    PT_ATOZ_DIGITS = re.compile(RE_ATOZ_DIGITS)

    def is_same_type(self, a: str, b: str) -> bool:
        if self.PT_DIGITS_TAIL.match(a) and self.PT_DIGITS_HEAD.match(b):
            return True
        if self.PT_ATOZ_TAIL.match(a) and self.PT_ATOZ_HEAD.match(b):
            return True
        return False

    def split_atoz_and_digits(self, token: str) -> list[str]:
        """Examples:
        - "abc100" -> ["abc", "100"]
        - "5a10" -> ["5", "a", "10"]
        - "1000" -> ["1000"]
        """
        parts = []
        for match in self.PT_ATOZ_DIGITS.finditer(token):
            atoz = match.group("atoz")
            digits = match.group("digits")
            word = match.group("word")
            if atoz:
                parts.append(atoz)
            if digits:
                parts.append(digits)
            if word:
                parts.append(word)
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
            if self.is_same_type(last_token[-1], token[0]):
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
            if type in ["non_word", "digits"]:
                tokens.append(part)
            else:
                tokens.extend(self.model_tokenizer.tokenize(part))
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
