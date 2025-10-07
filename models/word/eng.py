import polars as pl
import re

from itertools import islice
from pathlib import Path
from sedb import MongoDocsGenerator
from tclogger import logger, logstr, dict_get
from typing import TypedDict

from configs.envs import MONGO_ENVS

"""匹配完整的英文单词，包含字母、数字、连字符、空格、点号
要求：
1. 只能包含 0-9, a-z, A-Z, -(连字符), " "(空格), .(点号)
2. 开头和结尾必须是字母或数字
3. 前后必须有边界（非字母数字字符）
4. 如果以数字开头，必须包含至少一个字母
5. 不能包含 \s\-\s 模式（连字符两侧同时有空格时作为分隔符），分为：
    * 字母数字点号（紧密连接）
    * 连字符（紧密，前后不能同时是空格）
    * 空格（后面必须跟字母数字点号，不能跟连字符）
"""
RE_ENG = r"""
    (?<![0-9a-zA-Z])          # 前面不能是字母或数字（负向后查）
    (?:                       # 非捕获组
        # 情况1：以字母开头
        [a-zA-Z]              # 以字母开头
        (?:
            [0-9a-zA-Z\.]     # 字母、数字、点号
            | \-(?!\ )        # 连字符，后面不能是空格
            | (?<!\ )\-       # 连字符，前面不能是空格
            | \ (?!\-)        # 空格，后面不能是连字符
        )*
        [0-9a-zA-Z]           # 以字母或数字结尾
        |
        # 情况2：以数字开头（则必须包含字母）
        [0-9]                 # 以数字开头
        (?=                   # 正向前查：必须包含字母
            [0-9a-zA-Z\.\-]*  # 不包含空格的紧密字符
            [a-zA-Z]          # 必须包含字母
        )
        [0-9a-zA-Z\.\-]*      # 紧密字符序列
        [0-9a-zA-Z]           # 以字母或数字结尾
        (?:
            \ [a-zA-Z]        # 空格+字母开头
            (?:
                [0-9a-zA-Z\.]
                | \-(?!\ )
                | (?<!\ )\-
                | \ (?!\-)
            )*
            [0-9a-zA-Z]
        )*
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
        min_freq: int = 3,
        docs_count: int = None,
    ):
        self.generator = generator
        self.text_fields = text_fields
        self.sort_key = sort_key
        self.min_freq = min_freq
        self.docs_count = docs_count
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
        logger.mesg(f"  * {self.sort_key} < {self.min_freq}")
        old_count = len(self.records)
        self.records = {
            word: {
                "word": word,
                "doc_freq": record["doc_freq"],
                "term_freq": record["term_freq"],
            }
            for word, record in self.records.items()
            if record[self.sort_key] >= self.min_freq
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

    def set_dump_path(self):
        # keep first two digits of docs_count as suffix (12345 -> 12000)
        if self.docs_count:
            n_bits = len(str(self.docs_count))
            div = 10 ** (n_bits - 2)
            suffix = self.docs_count // div * div
        else:
            suffix = "latest"
        self.csv_path = Path(__file__).parent / "eng" / f"eng_freq_{suffix}.csv"

    def dump_records(self):
        logger.note(f"> Dump records:")
        df = pl.DataFrame(list(self.records.values()))
        logger.line(df, indent=2)
        self.set_dump_path()
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_csv(self.csv_path)
        logger.okay(f"  * {self.csv_path}")

    def log_results(self):
        top_k = 15
        total_n = len(self.records)
        count_str = f"{logstr.file(top_k)}/{logstr.mesg(total_n)}"
        logger.note(f"> Top words: ({count_str})")
        for i, v in enumerate(islice(self.records.values(), top_k)):
            record_str = f"{logstr.mesg(v['word'])} {logstr.file(v['doc_freq'])}"
            logger.line(f"  * [{i}]: {record_str}")

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
        self.dump_records()


def main():
    generator = MongoDocsGenerator()
    generator.init_cli_args(
        ikvs={
            **MONGO_ENVS,
            "mongo_collection": "videos",
            "include_fields": "title,tags,desc",
            # "extra_filters": "u:stat.view>1k;d:pubdate>=2025-08-01",
            # "extra_filters": "u:stat.view>1k",
        }
    )
    logger.okay(generator.args)
    # generator.init_all_with_cli_args(set_count=False, set_bar=False)
    generator.init_all_with_cli_args()
    recorder = EnglishWordsRecorder(
        generator,
        min_freq=5,
        docs_count=generator.total_count,
    )
    recorder.run()


if __name__ == "__main__":
    main()

    # python -m models.word.eng
    # python -m models.word.eng -m 20
    # python -m models.word.eng -t
