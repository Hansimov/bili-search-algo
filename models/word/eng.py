import argparse
import polars as pl
import re
import signal

from itertools import islice
from pathlib import Path
from sedb import MongoDocsGenerator, MongoDocsGeneratorArgParser
from tclogger import PathType, logger, logstr, brk, dict_get, MergedArgParser
from typing import TypedDict, Union, Literal

from configs.envs import MONGO_ENVS

"""匹配完整的英文单词，包含字母、数字、连字符、空格、点号
要求：
1. 只能包含 0-9, a-z, A-Z, -(连字符), " "(空格), .(点号)
2. 开头和结尾必须是字母或数字
3. 前后必须有边界（非字母数字字符）
4. 如果以数字开头，必须包含至少一个字母
5. 对连字符进行预处理
"""
RE_ENG = r"""
    (?<![0-9a-zA-Z])          # 前面不能是字母或数字（负向后查）
    (?:                       # 非捕获组
        # 情况1：以字母开头
        [a-zA-Z]              # 以字母开头
        [0-9a-zA-Z\-\.\ ]*    # 后面可以跟字母、数字、连字符、空格、点号
        [0-9a-zA-Z]           # 以字母或数字结尾
        |
        # 情况2：以数字开头（但必须包含字母）
        [0-9]                 # 以数字开头
        (?=                   # 正向前查：必须包含字母
            [0-9a-zA-Z\-\.]*  # 紧密字符（不含空格）
            [a-zA-Z]          # 必须包含字母
        )
        [0-9a-zA-Z\-\.\ ]*    # 后面可以跟字母、数字、连字符、空格、点号
        [0-9a-zA-Z]           # 以字母或数字结尾
    )
    (?![0-9a-zA-Z])           # 后面不能是字母或数字（负向前查）
"""

REP_ENG = re.compile(RE_ENG, re.VERBOSE)
REP_DASHES = re.compile(r"\-{2,}")
REP_DASH_WS = re.compile(r"(\ \-\ |\ \-|\-\ )")


class EnglishWordsExtractor:
    def timeout_handler(self, signum, frame):
        raise RuntimeError("re.match timeout")

    def set_signal(self):
        signal.signal(signal.SIGALRM, self.timeout_handler)
        signal.alarm(1)

    def clear_signal(self):
        signal.alarm(0)

    def clear_dash(self, text: str) -> str:
        if "--" in text:
            text = REP_DASHES.sub("|", text)
        if "- " in text or " -" in text:
            text = REP_DASH_WS.sub("|", text)
        return text

    def extract(self, text: str) -> list[str]:
        text = self.clear_dash(text)
        # self.set_signal()
        try:
            result = REP_ENG.findall(text)
            # self.clear_signal()
            return result
        except Exception as e:
            # self.clear_signal()
            raise e


REP_PURE_ENG = re.compile(r"^[0-9a-zA-Z\_\-\.\ ]+$")


class ChineseWordsExtractor:
    def extract(self, text: str) -> list[str]:
        words = re.split(r"[,#@]", text)
        res = []
        for w in words:
            w = w.strip()
            if REP_PURE_ENG.match(w):
                continue
            res.append(w)
        return res


class RecordType(TypedDict):
    word: str
    doc_freq: int
    term_freq: int
    first_seen: int
    last_seen: int


TEXT_FIELDS = ["title", "tags", "desc"]


class WordsRecorder:
    def __init__(
        self,
        generator: MongoDocsGenerator,
        extractor: Union[EnglishWordsExtractor, ChineseWordsExtractor],
        text_fields: list[str] = TEXT_FIELDS,
        sort_key: str = "doc_freq",
        min_freq: int = 3,
        dump_path: PathType = None,
    ):
        self.generator = generator
        self.text_fields = text_fields
        self.sort_key = sort_key
        self.min_freq = min_freq
        self.dump_path = dump_path
        self.extractor = extractor
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

    def dump_records(self):
        logger.note(f"> Dump records:")
        df = pl.DataFrame(list(self.records.values()))
        logger.line(df, indent=2)
        self.dump_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_csv(self.dump_path)
        logger.okay(f"  * {self.dump_path}")

    def log_results(self):
        top_k = 15
        total_n = len(self.records)
        count_str = f"{logstr.file(top_k)}/{logstr.mesg(total_n)}"
        logger.note(f"> Top words: ({count_str})")
        for i, v in enumerate(islice(self.records.values(), top_k)):
            record_str = f"{logstr.mesg(v['word'])} {logstr.file(v['doc_freq'])}"
            logger.line(f"  * [{i}]: {record_str}")

    def log_error(self, e: Exception, text: str, doc: dict):
        print()
        logger.warn(f"{logstr.mesg(brk(doc['_id']))}: {text}")

    def run(self):
        for doc_idx, doc in enumerate(self.generator.doc_generator()):
            text = self.doc_to_text(doc)
            if not text:
                continue

            try:
                words = self.extractor.extract(text)
            except Exception as e:
                self.log_error(e, text=text, doc=doc)
                raise e
                # continue

            if not words:
                continue
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


class RecordArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("-mf", "--min-freq", type=int, default=3)
        self.add_argument("-en", "--english", action="store_true")
        self.add_argument("-zh", "--chinese", action="store_true")
        self.args, _ = self.parse_known_args()


def get_dump_path(docs_count: int = None, lang: Literal["zh", "en"] = None) -> Path:
    if docs_count:
        # keep first 2 digits of docs_count as suffix (12345 -> 12000)
        n_bits = len(str(docs_count))
        div = 10 ** (n_bits - 2)
        suffix = docs_count // div * div
    else:
        suffix = "latest"
    if not lang:
        lang = "en"
    dump_path = Path(__file__).parent / "eng" / f"{lang}_freq_{suffix}.csv"
    return dump_path


def main(args: argparse.Namespace):
    if args.chinese:
        lang = "zh"
        include_fields = "tags"
        extractor = ChineseWordsExtractor()
    else:
        lang = "en"
        include_fields = "title,tags,desc"
        extractor = EnglishWordsExtractor()
    text_fields = include_fields.split(",")

    generator = MongoDocsGenerator()
    generator.init_cli_args(
        ikvs={
            **MONGO_ENVS,
            "mongo_collection": "videos",
            "include_fields": include_fields,
            # "extra_filters": "u:stat.view>1k;d:pubdate>=2025-08-01",
            # "extra_filters": "u:stat.view>1k",
        }
    )
    logger.okay(generator.args)
    # generator.init_all_with_cli_args(set_count=False, set_bar=False)
    generator.init_all_with_cli_args()

    dump_path = get_dump_path(docs_count=generator.total_count, lang=lang)

    recorder = WordsRecorder(
        generator=generator,
        extractor=extractor,
        text_fields=text_fields,
        min_freq=args.min_freq,
        dump_path=dump_path,
    )
    recorder.run()


if __name__ == "__main__":
    arg_parser = MergedArgParser(RecordArgParser, MongoDocsGeneratorArgParser)
    args = arg_parser.parse_args()
    main(args)

    # python -m models.word.eng -ed -en -mn 50000
    # python -m models.word.eng -ec -zh -mf 6 -mn 5000000
    # python -m models.word.eng -ec -en -mf 6
    # python -m models.word.eng -ec -zh -mf 6
