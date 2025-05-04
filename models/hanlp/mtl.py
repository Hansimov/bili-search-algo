import hanlp
import hanlp.pretrained
import hanlp.pretrained.mtl as hp_mtl

from tclogger import logger, logstr, brk
from typing import Literal, Union


MTL_MODELS = {
    "open_small_zh": hp_mtl.OPEN_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH,
    "open_base_zh": hp_mtl.OPEN_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH,
    "close_small_zh": hp_mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH,
    "close_small_zh_u": hp_mtl.CLOSE_TOK_POS_NER_SRL_UDEP_SDP_CON_ELECTRA_SMALL_ZH,
    "close_base_zh": hp_mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH,
    "close_gram_zh": hp_mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ERNIE_GRAM_ZH,
    "ancient_lzh": hp_mtl.KYOTO_EVAHAN_TOK_LEM_POS_UDEP_LZH,
    "minilm_l6": hp_mtl.UD_ONTONOTES_TOK_POS_LEM_FEA_NER_SRL_DEP_SDP_CON_MMINILMV2L6,
    "minilm_l12": hp_mtl.UD_ONTONOTES_TOK_POS_LEM_FEA_NER_SRL_DEP_SDP_CON_MMINILMV2L12,
    "xlmr_base": hp_mtl.UD_ONTONOTES_TOK_POS_LEM_FEA_NER_SRL_DEP_SDP_CON_XLMR_BASE,
    "char_ja": hp_mtl.NPCMJ_UD_KYOTO_TOK_POS_CON_BERT_BASE_CHAR_JA,
    "modern_base_en": hp_mtl.EN_TOK_LEM_POS_NER_SRL_UDEP_SDP_CON_MODERNBERT_BASE,
    "modern_large_en": hp_mtl.EN_TOK_LEM_POS_NER_SRL_UDEP_SDP_CON_MODERNBERT_LARGE,
}

MTL_MODEL_TYPE = Literal[
    "open_small_zh",
    "open_base_zh",
    "close_small_zh",
    "close_small_zh_u",
    "close_base_zh",
    "close_gram_zh",
    "ancient_lzh",
    "minilm_l6",
    "minilm_l12",
    "xlmr_base",
    "char_ja",
    "modern_base_en",
    "modern_large_en",
]


class HanlpMultiTasker:
    def __init__(
        self,
        model: MTL_MODEL_TYPE = "minilm_l12",
        vocab_dict: dict = None,
        verbose: bool = False,
    ):
        self.model = model
        self.vocab_dict = vocab_dict
        self.verbose = verbose
        self.load_mtl()

    def load_mtl(self):
        logger.note(
            f"> Loading Hanlp tokenizer: {logstr.mesg(brk(self.model))}",
            verbose=self.verbose,
        )
        self.mtl = hanlp.load(MTL_MODELS[self.model.lower()])

    def stringify(self, tokens: list[str]) -> str:
        tokens_str = f"{logstr.note('_')}".join(tokens)
        return tokens_str

    def tokenize(self, sentences: Union[str, list[str]], task_name: str = "tok"):
        if isinstance(sentences, str):
            sentences = [sentences]
        tokenize_res = [self.mtl(sent, tasks=task_name) for sent in sentences]
        tok_res = [res.get("tok", []) for res in tokenize_res]
        if len(tok_res) == 1:
            return tok_res[0]
        else:
            return tok_res


if __name__ == "__main__":
    from models.sentencepiece.test import TEST_SENTENCES

    tasker = HanlpMultiTasker(verbose=True)
    for sentence in TEST_SENTENCES:
        tokens = tasker.tokenize(sentence)
        tokens_str = tasker.stringify(tokens)
        logger.mesg(f"  * {tokens_str}")

    # python -m models.hanlp.mtl
