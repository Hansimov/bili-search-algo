import hanlp
import hanlp.pretrained

from tclogger import logger, logstr, brk
from typing import Literal, Union


class HanlpTokenizer:
    MODEL_LEVELS = {
        "coarse": hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH,
        "fine": hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH,
        "msr": hanlp.pretrained.tok.MSR_TOK_ELECTRA_BASE_CRF,
    }

    def __init__(
        self, level: Literal["coarse", "fine", "msr"] = "coarse", verbose: bool = False
    ):
        self.level = level
        self.verbose = verbose
        self.load_tokenizer()

    def load_tokenizer(self):
        logger.note(
            f"> Loading Hanlp tokenizer: {logstr.mesg(brk(self.level))}",
            verbose=self.verbose,
        )
        self.tokenizer = hanlp.load(self.MODEL_LEVELS[self.level.lower()])

    def stringify(self, tokens: list[str]) -> str:
        tokens_str = f"{logstr.note('_')}".join(tokens)
        return tokens_str

    def tokenize(self, text: Union[str, list[str]], **kwargs):
        if isinstance(text, list):
            tokenize_res = [self.tokenizer(subtext, **kwargs) for subtext in text]
        else:
            tokenize_res = self.tokenizer(text, **kwargs)
        return tokenize_res


if __name__ == "__main__":
    from models.sentencepiece.test import TEST_SENTENCES

    tokenizer = HanlpTokenizer(verbose=True)
    for sentence in TEST_SENTENCES:
        tokens = tokenizer.tokenize(sentence)
        tokens_str = tokenizer.stringify(tokens)
        logger.mesg(f"  * {tokens_str}")

    # python -m models.hanlp.tokenize