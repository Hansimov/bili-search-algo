import re
import sentencepiece as spm

from pathlib import Path
from tclogger import logger, logstr
from typing import Union

from models.sentencepiece.convert import DocSentenceConverter
from models.sentencepiece.test import TEST_SENTENCES


class SentencePreTokenizer:
    RE_NON_WORD = DocSentenceConverter.RE_NON_WORD
    RE_DIGITS = DocSentenceConverter.RE_DIGITS

    PT_NON_WORD = re.compile(RE_NON_WORD)
    PT_DIGITS = re.compile(RE_DIGITS)

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
        for match in self.PT_DIGITS.finditer(sentence):
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


class SentenceFullTokenizer:
    def __init__(self, model_path: Union[Path, str], verbose: bool = False):
        self.model_path = model_path
        self.verbose = verbose
        self.pre_tokenizer = SentencePreTokenizer()
        self.model_tokenizer = SentencePieceModelTokenizer(model_path)

    def join_tokens(self, tokens: list[str], sep: str = " ") -> str:
        return sep.join(tokens)

    def prettify_tokens(self, tokens: list[str]) -> str:
        tokens_str = f"{logstr.note('_')}".join(tokens)
        return tokens_str

    def tokenize_parts(self, parts: list[tuple]) -> list[str]:
        tokens = []
        for part, type in parts:
            if type in ["non_word", "digits"]:
                tokens.append(part)
            else:
                tokens.extend(self.model_tokenizer.tokenize(part))
        return tokens

    def tokenize(self, sentence: str) -> list[str]:
        sentence = sentence.lower()
        parts = self.pre_tokenizer.tokenize(sentence)
        tokens = self.tokenize_parts(parts)
        return tokens


if __name__ == "__main__":
    from tclogger import logger

    sentence = (
        " 这是 一段 中文。这是日语：これは 日本語 です。 Here is some English. https://www.google.com \n"
        "3g gta5 红警HBK08 (1) [34] 1999年11月11日 300勇士 3小时5分多钟300吨 2万海里 122毫米 2万 100"
    )
    logger.note("> Pre-Tokenizing ...")
    pre_tokenizer = SentencePreTokenizer()
    parts = pre_tokenizer.tokenize(sentence)
    logger.success(parts)

    logger.note("> Full-Tokenizing ...")
    model_path = str(Path(__file__).parents[2] / "sp_10m_10k.model")
    tokenizer = SentenceFullTokenizer(model_path, verbose=True)
    for sentence in TEST_SENTENCES:
        tokens = tokenizer.tokenize(sentence)
        pretty_tokens = tokenizer.prettify_tokens(tokens)
        logger.mesg(f"  * {pretty_tokens}")

    # python -m models.sentencepiece.tokenize
