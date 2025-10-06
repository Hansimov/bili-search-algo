import re

from itertools import islice
from sedb import MongoDocsGenerator
from tclogger import logger, logstr, dict_get
from typing import TypedDict

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
        return REP_ENG.findall(text)


class RecordType(TypedDict):
    word: str
    doc_freq: int
    term_freq: int
    first_seen: int
    last_seen: int


TEXT_FIELDS = ["title", "tags", "desc"]


class EnglishWordsRecorder:
    def __init__(
        self,
        generator: MongoDocsGenerator,
        text_fields: list[str] = TEXT_FIELDS,
        sort_key: str = "doc_freq",
        min_count: int = 2,
    ):
        self.generator = generator
        self.text_fields = text_fields
        self.sort_key = sort_key
        self.min_count = min_count
        self.extractor = EnglishWordExtractor()
        self.records: dict[str, RecordType] = {}

    def doc_to_text(self, doc: dict):
        text = ""
        for field in self.text_fields:
            field_str = dict_get(doc, field)
            if field_str == "-":
                # speed-up for desc
                continue
            if field_str:
                if text:
                    text += " | " + field_str
                else:
                    text = field_str
        return text.strip().lower()

    def update_record_by_doc(self, word: str, doc_idx: int):
        if word not in self.records:
            self.records[word] = {
                "word": word,
                "doc_freq": 1,
                "term_freq": 0,
                "first_seen": doc_idx,
                "last_seen": doc_idx,
            }
        last_seen = self.records[word]["last_seen"]
        if last_seen != doc_idx:
            self.records[word]["doc_freq"] += 1
            self.records[word]["last_seen"] = doc_idx

    def update_record_by_term(self, word: str):
        self.records[word]["term_freq"] += 1

    def sort_records(self, reverse: bool = True):
        self.records = dict(
            sorted(
                self.records.items(),
                key=lambda item: item[1][self.sort_key],
                reverse=reverse,
            )
        )

    def filter_records(self):
        logger.note(f"> Filter records:")
        logger.mesg(f"  * {self.sort_key} < {self.min_count}")
        old_count = len(self.records)
        self.records = {
            word: record
            for word, record in self.records.items()
            if record[self.sort_key] >= self.min_count
        }
        new_count = len(self.records)
        diff_count = old_count - new_count
        if old_count > 0:
            diff_ratio = diff_count / old_count * 100
        else:
            diff_ratio = 0
        count_str = f"{logstr.okay(new_count)}/{logstr.warn(diff_count)}/{logstr.mesg(old_count)}"
        ratio_str = logstr.warn(f"-{diff_ratio:.1f}%")
        logger.okay(f"  * {count_str} ({ratio_str})")

    def log_results(self):
        top_k = 15
        total_n = len(self.records)
        count_str = f"{logstr.file(top_k)}/{logstr.mesg(total_n)}"
        logger.note(f"> Top words: ({count_str})")
        for i, v in enumerate(islice(self.records.values(), top_k)):
            print(f"[{i}]: {v}")

    def run(self):
        for doc_idx, doc in enumerate(self.generator.doc_generator()):
            text = self.doc_to_text(doc)
            if not text:
                continue
            words = self.extractor.extract(text)
            # if words:
            #     logger.okay(f"[{doc_idx}]: {words}")
            unique_words = set(words)
            for w in unique_words:
                self.update_record_by_doc(w, doc_idx)
            for w in words:
                self.update_record_by_term(w)
        print()
        self.filter_records()
        self.sort_records()
        self.log_results()


def main():
    generator = MongoDocsGenerator()
    generator.init_cli_args(
        ikvs={
            **MONGO_ENVS,
            "mongo_collection": "videos",
            "include_fields": "title,tags,desc",
            "extra_filters": "u:stat.view>1k;d:pubdate>=2025-09-01",
        }
    )
    logger.okay(generator.args)
    # generator.init_all_with_cli_args(set_count=False, set_bar=False)
    generator.init_all_with_cli_args()
    recorder = EnglishWordsRecorder(generator)
    recorder.run()


if __name__ == "__main__":
    main()

    # python -m models.word.eng
    # python -m models.word.eng -m 20
