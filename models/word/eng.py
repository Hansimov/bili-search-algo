import re

from sedb import MongoDocsGenerator
from tclogger import logger, dict_get, dict_to_str

from configs.envs import MONGO_ENVS

# 匹配完整的英文单词，包含字母、数字、连字符、空格、点号
# 要求：
# 1. 只能包含 0-9, a-z, A-Z, -(连字符), " "(空格), .(点号)
# 2. 开头和结尾必须是字母或数字
# 3. 前后必须有边界（非字母数字字符）
# 4. 如果以数字开头，必须包含至少一个字母
RE_ENG = r"""
    (?<![0-9a-zA-Z])          # 前面不能是字母或数字（负向后查）
    (?:                       # 非捕获组
        # 情况1：以字母开头的单词
        [a-zA-Z]              # 以字母开头  
        [0-9a-zA-Z\-\ \.]*    # 后面可以跟字母、数字、连字符、空格、点号
        [0-9a-zA-Z]           # 以字母或数字结尾
        # |
        # [a-zA-Z]              # 单个字母
        |
        # 情况2：以数字开头（但不通过空格分隔的形式包含字母）
        [0-9]                 # 以数字开头
        (?=                   # 正向前查：在空格前必须包含字母
            [0-9a-zA-Z\-\.]*      # 不包含空格的字符
            [a-zA-Z]              # 必须包含字母
        )
        [0-9a-zA-Z\-\.]*      # 不包含空格的紧密字符序列
        [0-9a-zA-Z]           # 以字母或数字结尾
        (?:\ [a-zA-Z][0-9a-zA-Z\-\ \.]*[0-9a-zA-Z])*  # 后面可以跟空格+字母开头的部分
    )
    (?![0-9a-zA-Z])           # 后面不能是字母或数字（负向前查）
"""

REP_ENG = re.compile(RE_ENG, re.VERBOSE)


class EnglishWordExtractor:
    def extract(self, text: str) -> list[str]:
        matches = REP_ENG.findall(text)
        return matches


class EnglishWordsCounter:
    def __init__(self, generator: MongoDocsGenerator):
        self.generator = generator
        self.extractor = EnglishWordExtractor()

    def run(self):
        for doc in self.generator.doc_generator():
            # logger.line(doc)
            text = ""
            for field in ["title", "tags"]:
                field_str = dict_get(doc, field)
                if field_str:
                    if text:
                        text += " | " + doc[field]
                    else:
                        text = doc[field]
            if text:
                eng_words = self.extractor.extract(text)
                if eng_words:
                    logger.okay(eng_words)


def main():
    generator = MongoDocsGenerator()
    generator.init_cli_args(
        ikvs={
            **MONGO_ENVS,
            "mongo_collection": "videos",
            "include_fields": "title,tags",
            "extra_filters": "u:stat.view>1w;d:pubdate>=2025-10-05",
        }
    )
    logger.okay(generator.args)
    generator.init_all_with_cli_args(set_count=False, set_bar=False)
    counter = EnglishWordsCounter(generator)
    counter.run()


if __name__ == "__main__":
    main()

    # python -m models.word.eng
    # python -m models.word.eng -m 20
