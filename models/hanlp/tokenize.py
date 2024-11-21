import hanlp
import hanlp.pretrained

from tclogger import logger, logstr, brk
from typing import Literal, Union


class HanlpTokenizer:
    MODEL_LEVELS = {
        "albert": hanlp.pretrained.tok.LARGE_ALBERT_BASE,
        "coarse": hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH,
        "fine": hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH,
        "msr": hanlp.pretrained.tok.MSR_TOK_ELECTRA_BASE_CRF,
        "pku": hanlp.pretrained.tok.SIGHAN2005_PKU_BERT_BASE_ZH,
        "ctb9_small": hanlp.pretrained.tok.CTB9_TOK_ELECTRA_SMALL,
        "ctb9_base": hanlp.pretrained.tok.CTB9_TOK_ELECTRA_BASE,
        "ctb9_base_crf": hanlp.pretrained.tok.CTB9_TOK_ELECTRA_BASE_CRF,
        "ud_l6": hanlp.pretrained.tok.UD_TOK_MMINILMV2L6,
        "ud_l12": hanlp.pretrained.tok.UD_TOK_MMINILMV2L12,
    }

    def __init__(
        self,
        level: Literal["coarse", "fine", "ctb9_base"] = "coarse",
        verbose: bool = False,
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
