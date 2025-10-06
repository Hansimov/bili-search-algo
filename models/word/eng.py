import re

from sedb import MongoDocsGenerator
from tclogger import logger, dict_to_str

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
        # 情况1：以字母开头
        [a-zA-Z]              # 以字母开头  
        [0-9a-zA-Z\-\ \.]*    # 后面可以跟字母、数字、连字符、空格、点号
        [0-9a-zA-Z]           # 以字母或数字结尾
        |                     # 或者
        # 情况2：以数字开头但包含字母
        [0-9]                 # 以数字开头
        (?=.*[a-zA-Z])        # 必须包含至少一个字母（正向前查）
        [0-9a-zA-Z\-\ \.]*    # 后面可以跟字母、数字、连字符、空格、点号
        [0-9a-zA-Z]           # 以字母或数字结尾
    )
    (?![0-9a-zA-Z])           # 后面不能是字母或数字（负向前查）
"""

REP_ENG = re.compile(RE_ENG, re.VERBOSE)


class EnglishWordExtractor:
    pass


class EnglishWordsCounter:
    def __init__(self, generator: MongoDocsGenerator):
        self.generator = generator


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
    for doc in generator.doc_generator():
        logger.line(doc)


if __name__ == "__main__":
    main()

    # python -m models.word.eng
    # python -m models.word.eng -m 5
