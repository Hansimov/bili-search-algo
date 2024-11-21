import hanlp
import hanlp.pretrained

from tclogger import logger, logstr, brk
from typing import Literal, Union


class HanlpPosTagger:
    MODEL_LEVELS = {
        "ctb5": hanlp.pretrained.pos.CTB5_POS_RNN,
        "ctb9": hanlp.pretrained.pos.CTB9_POS_ALBERT_BASE,
        "c863": hanlp.pretrained.pos.C863_POS_ELECTRA_SMALL,
        "pku": hanlp.pretrained.pos.PKU_POS_ELECTRA_SMALL,
    }

    def __init__(
        self,
        level: Literal["ctb5", "ctb9", "c863", "pku"] = "ctb5",
        verbose: bool = False,
    ):
        self.level = level
        self.verbose = verbose
        self.load_pos_tagger()

    def load_pos_tagger(self):
        logger.note(
            f"> Loading Hanlp POS tagger: {logstr.mesg(brk(self.level))}",
            verbose=self.verbose,
        )
        self.pos_tagger = hanlp.load(self.MODEL_LEVELS[self.level.lower()])

    def pos_tag(self, text: Union[str, list[str]], **kwargs):
        if isinstance(text, list):
            pos_res = [self.pos_tagger(subtext, **kwargs) for subtext in text]
        else:
            pos_res = self.pos_tagger(text, **kwargs)
        return pos_res


if __name__ == "__main__":
    from models.sentencepiece.test import TEST_SENTENCES
    from models.hanlp.tokenize import HanlpTokenizer

    tokenizer = HanlpTokenizer(level="coarse", verbose=True)
    pos_tagger = HanlpPosTagger(verbose=True)
    for sentence in TEST_SENTENCES:
        tokens = tokenizer.tokenize(sentence)
        tags = pos_tagger.pos_tag(tokens)

    # python -m models.hanlp.pos
