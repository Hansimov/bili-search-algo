import hanlp
import hanlp.pretrained

from tclogger import logger, logstr, brk
from typing import Literal, Union

# https://hanlp.hankcs.com/docs/annotations/pos/pku.html
PKU_POS_NAMES = {
    "Ag": "形语素",
    "a": "形容词",
    "ad": "副形词",
    "an": "名形词",
    "b": "区别语素",
    "c": "连词",
    "Dg": "副语素",
    "d": "副词",
    "e": "叹词",
    "f": "方位词",
    "h": "前接成分",
    "i": "成语",
    "j": "简称略语",
    "k": "后接成分",
    "l": "习用语",
    "Mg": "数语素",
    "m": "数词",
    "Ng": "名语素",
    "n": "名词",
    "nr": "人名",
    "ns": "地名",
    "nt": "机构团体",
    "nx": "外文字符",
    "nz": "其他专名",
    "o": "拟声词",
    "p": "介词",
    "q": "量词",
    "Rg": "代语素",
    "r": "代词",
    "s": "处所词",
    "Tg": "时语素",
    "t": "时间词",
    "u": "助词",
    "Vg": "动语素",
    "v": "动词",
    "vd": "副动词",
    "vn": "名动词",
    "w": "标点符号",
    "x": "非语素字",
    "Yg": "语气语素",
    "y": "语气词",
    "z": "状态词",
}

CTB_POS_NAMES = {}


class HanlpPosTagger:
    MODEL_NAMES = {
        "ctb5": hanlp.pretrained.pos.CTB5_POS_RNN,
        "ctb9": hanlp.pretrained.pos.CTB9_POS_ALBERT_BASE,
        "c863": hanlp.pretrained.pos.C863_POS_ELECTRA_SMALL,
        "pku": hanlp.pretrained.pos.PKU_POS_ELECTRA_SMALL,
    }

    def __init__(
        self,
        model_name: Literal["ctb5", "ctb9", "c863", "pku"] = "pku",
        verbose: bool = False,
    ):
        self.model_name = model_name
        self.verbose = verbose
        self.load_model()

    def load_model(self):
        logger.note(
            f"> Loading Hanlp POS tagger: {logstr.mesg(brk(self.model_name))}",
            verbose=self.verbose,
        )
        self.model = hanlp.load(self.MODEL_NAMES[self.model_name.lower()])

    def tag_pos(self, text: Union[str, list[str]], **kwargs) -> list[str]:
        if isinstance(text, str):
            res = self.model([text], **kwargs)
        else:
            res = self.model(text, **kwargs)
        return res

    def tags_to_names(self, tags: Union[str, list[str]]) -> Union[str, list[str]]:
        if self.model_name == "pku":
            POS_NAMES = PKU_POS_NAMES
        elif self.model_name.startswith("ctb"):
            POS_NAMES = CTB_POS_NAMES
        else:
            POS_NAMES = {}
        if isinstance(tags, str):
            return POS_NAMES.get(tags, tags)
        else:
            return [POS_NAMES.get(tag, tag) for tag in tags]


if __name__ == "__main__":
    from models.sentencepiece.test import TEST_SENTENCES
    from models.hanlp.tokenize import HanlpTokenizer

    tokenizer = HanlpTokenizer(model="coarse", verbose=True)
    tagger = HanlpPosTagger(model_name="pku", verbose=True)
    # tokens = ["我", "爱", "你"]
    # tags = pos_tagger.tag_pos(tokens)
    # logger.mesg(f"  * {tags}")
    for sentence in TEST_SENTENCES:
        tokens = tokenizer.tokenize(sentence)
        tags = tagger.tag_pos(tokens)
        logger.mesg(f"  * {tokens}")
        # logger.mesg(f"  * {tags}")
        token_tags = " ".join(
            [
                f"{logstr.success(token)}/{logstr.file(tagger.tags_to_names(tag))}"
                for token, tag in zip(tokens, tags)
            ]
        )
        logger.mesg(f"  * {token_tags}")

    # python -m models.hanlp.pos
