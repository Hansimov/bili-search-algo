import argparse
import json
import math
import polars as pl
import re
import signal
import time
import unicodedata

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from itertools import islice
from pathlib import Path
from sedb import MongoDocsGenerator, MongoDocsGeneratorArgParser
from tclogger import PathType, brk, dict_get, logger, logstr, MergedArgParser
from typing import Callable, Literal, TypedDict, Union

from configs.envs import MONGO_ENVS
from data_utils.videos.filter import REGION_MONGO_FILTERS

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
REP_MULTISPACE = re.compile(r"\s+")
REP_DOMAIN = re.compile(r"(?:[a-z0-9-]+\.)+[a-z]{2,}$")
REP_VERSION = re.compile(r"^[a-z]{1,10}\d+(?:\.\d+)+$")
REP_ENG_ALLOWED = re.compile(r"^[a-z0-9\-\.\ ]+$")
REP_ALPHA = re.compile(r"^[a-z]+$")
REP_ALNUM = re.compile(r"^[a-z0-9]+$")
REP_ASCII_TOKEN_CHARS = re.compile(r"[a-z0-9\-.]")
REP_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
REP_DIGITS = re.compile(r"\d")
REP_LATIN = re.compile(r"[a-zA-Z]")
REP_ASCII_MIXED = re.compile(r"[0-9a-zA-Z\-.]")
REP_CJK_ALLOWED = re.compile(
    r"^[0-9a-zA-Z\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff：:，,、·・\-\+&\s]+$"
)
REP_CJK_SPLITTER = re.compile(r"[#@|｜/／]+")
REP_CJK_CONNECTORS = re.compile(r"[：:，,、·・\-\+&\s]+")
REP_CJK_NOISE = re.compile(r"[!！?？~～`$%^*_+=\\\[\]{}<>《》【】\"']")
REP_CJK_TRAILING_PUNCT = re.compile(r"[!！?？~～。．…]+$")
REP_CJK_LEADING_PUNCT = re.compile(r"^[!！?？~～#＃@＠,，、:：·・\-\+&\s]+")
REP_CJK_QUESTION = re.compile(
    r"(什么|怎么|为什么|怎么办|怎么看|怎么回事|值不值|值不值得|好不好|吗$|呢$|么$|哪家|是否)"
)

ENGLISH_SHORT_KEEP = {
    "ai",
    "ar",
    "bgm",
    "cf",
    "cpu",
    "cv",
    "dj",
    "ed",
    "edm",
    "fps",
    "gpu",
    "hd",
    "ip",
    "lol",
    "mc",
    "mmd",
    "mmo",
    "mv",
    "nba",
    "op",
    "ost",
    "pc",
    "ps",
    "pv",
    "re",
    "rpg",
    "tv",
    "ufc",
    "ui",
    "vr",
    "vs",
    "wwe",
}
ENGLISH_STOPWORDS = {
    "a",
    "an",
    "and",
    "for",
    "in",
    "me",
    "my",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}
ENGLISH_CTA_PREFIXES = {"click", "follow", "subscribe", "use", "watch"}
ENGLISH_TEMPLATE_PARTS = {
    "blackboard",
    "filedetails",
    "http-equiv",
    "meta",
    "msource",
    "navhide",
    "postdesc",
    "redpack",
    "source",
    "streamsource",
    "utmsource",
}
ENGLISH_TEMPLATE_PHRASE_PARTS = {
    "channel",
    "channels",
    "follow",
    "following",
    "playlist",
    "playlists",
    "profile",
    "profiles",
    "subscribe",
    "subscriber",
    "subscribers",
    "watching",
}
ENGLISH_SINGLETON_STOPWORDS = {
    "account",
    "app",
    "channel",
    "details",
    "event",
    "get",
    "http",
    "https",
    "list",
    "page",
    "playlist",
    "profile",
    "sub",
    "top",
    "video",
    "videos",
    "watch",
}
FULL_DATA_WORKERS = 10
SHARD_PROGRESS_EVERY_DOCS = 50000
SHARD_PROGRESS_LOG_INTERVAL = 20.0
REGION_SHARDS = [
    ["cine_movie"],
    ["douga_anime"],
    ["tech_sports"],
    ["music_dance"],
    ["fashion_ent"],
    ["know_info"],
    ["daily_life"],
    ["other_life"],
    ["mobile_game"],
    ["other_game"],
]


def normalize_spaces(text: str) -> str:
    return REP_MULTISPACE.sub(" ", text).strip()


def contains_cjk(text: str) -> bool:
    return bool(REP_CJK.search(text))


def count_cjk(text: str) -> int:
    return sum(1 for char in text if REP_CJK.match(char))


def count_digits(text: str) -> int:
    return len(REP_DIGITS.findall(text))


def count_latin(text: str) -> int:
    return len(REP_LATIN.findall(text))


def count_ascii_token_chars(text: str) -> int:
    return len(REP_ASCII_TOKEN_CHARS.findall(text))


def count_ascii_mixed_chars(text: str) -> int:
    return len(REP_ASCII_MIXED.findall(text))


def calc_token_units(text: str) -> int:
    return count_cjk(text) * 3 + count_ascii_mixed_chars(text)


def looks_like_random_ascii(token: str) -> bool:
    if not REP_ALPHA.fullmatch(token):
        return False
    if len(token) <= 4:
        return False
    vowel_count = sum(char in "aeiou" for char in token)
    return vowel_count == 0


def looks_like_random_mixed_ascii(token: str) -> bool:
    if len(token) < 6:
        return False
    if not REP_ALNUM.fullmatch(token):
        return False
    letters = sum(char.isalpha() for char in token)
    digits = sum(char.isdigit() for char in token)
    if not letters or not digits:
        return False
    vowel_count = sum(char in "aeiou" for char in token.lower())
    return letters >= 4 and digits <= 3 and vowel_count <= 1


def normalize_common_token(text: str, lowercase: bool = True) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = normalize_spaces(text)
    if lowercase:
        text = text.lower()
    return text


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

    def normalize_token(self, token: str) -> str:
        token = normalize_common_token(token)
        token = token.strip("-_. ")
        return token

    def is_meaningful_part(self, part: str, phrase_mode: bool = False) -> bool:
        if not part:
            return False
        if part in ENGLISH_STOPWORDS:
            return phrase_mode
        if "." in part:
            return False

        subparts = [subpart for subpart in part.split("-") if subpart]
        if not subparts:
            return False

        for subpart in subparts:
            if not REP_ENG_ALLOWED.fullmatch(subpart):
                return False
            letters = sum(char.isalpha() for char in subpart)
            digits = sum(char.isdigit() for char in subpart)
            if letters == 0:
                return False
            if digits and len(subpart) < 3:
                return False
            if digits and letters < 2 and subpart not in ENGLISH_SHORT_KEEP:
                return False
            if digits and subpart[0].isdigit() and letters < 2:
                return False
            if digits and digits >= letters and len(subpart) < 5:
                return False
            if REP_ALNUM.fullmatch(subpart) and len(subpart) >= 7 and digits >= 2:
                vowel_count = sum(char in "aeiou" for char in subpart.lower())
                if letters >= 3 and vowel_count <= 1:
                    return False
            if REP_ALNUM.fullmatch(subpart) and len(subpart) >= 12 and digits:
                return False
            if looks_like_random_mixed_ascii(subpart):
                return False
            if letters and not digits:
                if len(subpart) == 1:
                    return False
                if (
                    len(subpart) == 2
                    and subpart not in ENGLISH_SHORT_KEEP
                    and not phrase_mode
                ):
                    return False
                if (
                    looks_like_random_ascii(subpart)
                    and subpart not in ENGLISH_SHORT_KEEP
                ):
                    return False
        return True

    def is_meaningful_token(self, token: str) -> bool:
        if not token or len(token) < 2 or len(token) > 48:
            return False
        if token in ENGLISH_SINGLETON_STOPWORDS:
            return False
        if token in ENGLISH_TEMPLATE_PARTS:
            return False
        if not REP_ENG_ALLOWED.fullmatch(token):
            return False
        if token[0] in "-." or token[-1] in "-.":
            return False
        if token.count(".") and not REP_VERSION.fullmatch(token):
            return False
        if REP_DOMAIN.fullmatch(token):
            return False

        parts = token.split(" ")
        if len(parts) > 6:
            return False
        if token.count(" ") > 5:
            return False
        if count_ascii_token_chars(token) > 24:
            return False
        if parts[0] in ENGLISH_CTA_PREFIXES and len(parts) > 1:
            return False
        if len(parts) > 1 and any(
            part in ENGLISH_TEMPLATE_PHRASE_PARTS for part in parts
        ):
            return False
        if any(part in ENGLISH_TEMPLATE_PARTS for part in parts):
            return False
        if len(parts) >= 4 and sum(part in ENGLISH_STOPWORDS for part in parts) >= 2:
            return False
        phrase_mode = len(parts) > 1
        return all(
            self.is_meaningful_part(part, phrase_mode=phrase_mode) for part in parts
        )

    def extract(self, text: str) -> list[str]:
        text = self.clear_dash(text)
        # self.set_signal()
        try:
            result = []
            seen = set()
            for token in REP_ENG.findall(text):
                token = self.normalize_token(token)
                if not self.is_meaningful_token(token):
                    continue
                if token in seen:
                    continue
                seen.add(token)
                result.append(token)
            # self.clear_signal()
            return result
        except Exception as e:
            # self.clear_signal()
            raise e


REP_PURE_ENG = re.compile(r"^[0-9a-zA-Z\_\-\.\ ]+$")


class ChineseWordsExtractor:
    def __init__(self):
        self.english_extractor = EnglishWordsExtractor()

    def normalize_token(self, token: str) -> str:
        token = normalize_common_token(token, lowercase=False)
        token = re.sub(r"[A-Z]+", lambda match: match.group(0).lower(), token)
        if contains_cjk(token):
            token = token.replace(",", "，").replace(":", "：")
            token = token.replace(" ", "")
        token = REP_CJK_TRAILING_PUNCT.sub("", token)
        token = REP_CJK_LEADING_PUNCT.sub("", token)
        token = normalize_spaces(token)
        return token

    def should_split_by_commas(self, token: str) -> bool:
        return sum(token.count(char) for char in [",", "，", "、"]) >= 2

    def is_meaningful_token(self, token: str) -> bool:
        if not token:
            return False
        if REP_PURE_ENG.fullmatch(token):
            return False
        if not contains_cjk(token):
            return False
        if " " in token:
            return False
        if not REP_CJK_ALLOWED.fullmatch(token):
            return False
        if REP_CJK_NOISE.search(token):
            return False

        token = normalize_spaces(token)
        cjk_len = count_cjk(token)
        latin_len = count_latin(token)
        digit_len = count_digits(token)
        has_connector = bool(REP_CJK_CONNECTORS.search(token))
        token_units = calc_token_units(token)

        if cjk_len > 8:
            return False
        if token_units > 24:
            return False
        if REP_CJK_QUESTION.search(token):
            return False

        if has_connector:
            segments = [
                segment.strip()
                for segment in REP_CJK_CONNECTORS.split(token)
                if segment.strip()
            ]
        else:
            segments = [token]
        if not segments:
            return False
        if len(segments) > 4:
            return False

        for segment in segments:
            segment_cjk_len = count_cjk(segment)
            segment_latin_len = count_latin(segment)
            segment_digit_len = count_digits(segment)
            if segment_cjk_len > 12:
                return False
            if (
                segment_latin_len
                and segment_digit_len
                and segment_latin_len < 2
                and segment_cjk_len < 2
            ):
                return False
            if segment_latin_len and not segment_cjk_len:
                if not self.english_extractor.is_meaningful_token(segment.lower()):
                    return False
            if segment_latin_len and segment_cjk_len:
                if segment_latin_len == 1 and segment_cjk_len < 3:
                    return False
                if segment[-1].isascii() and segment_latin_len < 2:
                    return False
            if segment_digit_len > 4 and segment_cjk_len < 2:
                return False
            if calc_token_units(segment) > 24:
                return False
        if latin_len and not has_connector and cjk_len < 2:
            return False
        return True

    def extract(self, text: str) -> list[str]:
        text = normalize_common_token(text, lowercase=False)
        res = []
        seen = set()
        for token in REP_CJK_SPLITTER.split(text):
            token = self.normalize_token(token)
            if not token:
                continue
            sub_tokens = [token]
            if self.should_split_by_commas(token):
                sub_tokens = [
                    self.normalize_token(part) for part in re.split(r"[，,、]+", token)
                ]
            for sub_token in sub_tokens:
                if not sub_token or not self.is_meaningful_token(sub_token):
                    continue
                if sub_token in seen:
                    continue
                seen.add(sub_token)
                res.append(sub_token)
        return res


class RecordType(TypedDict):
    word: str
    doc_freq: int
    term_freq: int


class ShardStatus(TypedDict, total=False):
    shard_idx: int
    state: str
    processed_docs: int
    total_docs: int
    skip_count: int
    max_count: int
    dump_path: str
    updated_at: float


class ShardPlan(TypedDict, total=False):
    shard_idx: int
    regions: list[str]
    skip_count: int
    max_count: int
    mongo_filter: dict
    strategy: str


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
        log_details: bool = True,
        progress_every_docs: int = SHARD_PROGRESS_EVERY_DOCS,
        progress_callback: Callable[[int], None] | None = None,
    ):
        self.generator = generator
        self.text_fields = text_fields
        self.sort_key = sort_key
        self.min_freq = min_freq
        self.dump_path = dump_path
        self.extractor = extractor
        self.log_details = log_details
        self.progress_every_docs = progress_every_docs
        self.progress_callback = progress_callback
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
                "doc_freq": 0,
                "term_freq": 0,
            }
        self.records[word]["doc_freq"] += 1

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
        if self.min_freq <= 1:
            return
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
        df = pl.DataFrame(list(self.records.values()))
        self.dump_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_csv(self.dump_path)
        if self.log_details:
            logger.note(f"> Dump records:")
            logger.line(df, indent=2)
            logger.okay(f"  * {self.dump_path}")

    def log_results(self):
        if not self.log_details:
            return
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
            if self.progress_callback and (doc_idx + 1) % self.progress_every_docs == 0:
                self.progress_callback(doc_idx + 1)
        if self.progress_callback:
            self.progress_callback(doc_idx + 1 if "doc_idx" in locals() else 0)
        print()
        self.filter_records()
        self.sort_records()
        self.log_results()
        self.dump_records()


def write_shard_status(
    status_path: Path,
    *,
    shard_idx: int,
    state: str,
    processed_docs: int,
    total_docs: int,
    skip_count: int,
    max_count: int,
    dump_path: Path | None = None,
):
    payload: ShardStatus = {
        "shard_idx": shard_idx,
        "state": state,
        "processed_docs": processed_docs,
        "total_docs": total_docs,
        "skip_count": skip_count,
        "max_count": max_count,
        "updated_at": time.time(),
    }
    if dump_path is not None:
        payload["dump_path"] = str(dump_path)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def load_shard_status(status_path: Path) -> ShardStatus | None:
    if not status_path.exists():
        return None
    try:
        return json.loads(status_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def format_shard_progress(statuses: list[ShardStatus], workers: int) -> list[str]:
    if not statuses:
        return [f"> Shard progress: 0/{workers} reported"]
    total_docs = sum(status.get("total_docs", 0) for status in statuses)
    processed_docs = sum(status.get("processed_docs", 0) for status in statuses)
    done_count = sum(status.get("state") == "done" for status in statuses)
    running_count = sum(status.get("state") == "running" for status in statuses)
    if total_docs:
        percent = processed_docs / total_docs * 100
        head = (
            "> Shard progress: "
            f"{processed_docs}/{total_docs} docs ({percent:.2f}%) | "
            f"done {done_count}/{workers} | running {running_count}"
        )
    else:
        head = (
            "> Shard progress: "
            f"{processed_docs} docs | total pending | "
            f"done {done_count}/{workers} | running {running_count}"
        )
    lines = [head]
    for status in sorted(statuses, key=lambda item: item.get("shard_idx", -1)):
        shard_total = status.get("total_docs", 0)
        shard_done = status.get("processed_docs", 0)
        if shard_total:
            shard_percent = shard_done / shard_total * 100
            progress_text = f"{shard_done}/{shard_total} ({shard_percent:.2f}%)"
        else:
            progress_text = f"{shard_done}/?"
        lines.append(
            f"  * shard {status.get('shard_idx', -1):02d}: {status.get('state', 'unknown')} "
            f"{progress_text}"
        )
    return lines


def build_skip_shard_plans(
    total_count: int,
    workers: int,
    base_skip_count: int = 0,
) -> list[ShardPlan]:
    if workers <= 1:
        return [
            {
                "shard_idx": 0,
                "skip_count": base_skip_count,
                "max_count": max(total_count - base_skip_count, 0),
                "strategy": "skip",
            }
        ]
    effective_total = max(total_count - base_skip_count, 0)
    shard_size = max(1, math.ceil(effective_total / workers))
    plans: list[ShardPlan] = []
    for worker_idx in range(workers):
        skip_count = base_skip_count + worker_idx * shard_size
        remain = max(effective_total - worker_idx * shard_size, 0)
        max_count = min(shard_size, remain)
        if max_count == 0:
            continue
        plans.append(
            {
                "shard_idx": worker_idx,
                "skip_count": skip_count,
                "max_count": max_count,
                "strategy": "skip",
            }
        )
    return plans


def merge_mongo_filters(filters: list[dict]) -> dict:
    filters = [mongo_filter for mongo_filter in filters if mongo_filter]
    if not filters:
        return {}
    if len(filters) == 1:
        return filters[0]
    return {"$or": filters}


def build_region_shard_plans(
    generator: MongoDocsGenerator,
    workers: int,
) -> list[ShardPlan]:
    shard_region_groups = REGION_SHARDS

    plans: list[ShardPlan] = []
    for shard_idx, regions in enumerate(shard_region_groups):
        filters = [REGION_MONGO_FILTERS[region] for region in regions]
        mongo_filter = merge_mongo_filters(filters)
        plans.append(
            {
                "shard_idx": shard_idx,
                "regions": regions,
                "mongo_filter": mongo_filter,
                "strategy": "region_filter",
            }
        )
    return plans


def merge_partial_records(
    partial_paths: list[Path],
    dump_path: Path,
    sort_key: str,
    min_freq: int,
):
    lazy_frames = [pl.scan_csv(path) for path in partial_paths]
    df = (
        pl.concat(lazy_frames)
        .group_by("word")
        .agg(
            pl.col("doc_freq").sum().alias("doc_freq"),
            pl.col("term_freq").sum().alias("term_freq"),
        )
        .filter(pl.col(sort_key) >= min_freq)
        .sort(sort_key, descending=True)
        .collect(streaming=True)
    )
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(dump_path)
    logger.note("> Merged shard records:")
    logger.line(df.head(10), indent=2)
    logger.okay(f"  * {dump_path}")


def build_generator(
    args: argparse.Namespace,
    *,
    skip_count: int = None,
    max_count: int = None,
    mongo_filter: dict | None = None,
    sort_index: str | None = "insert_at",
    sort_order: str | None = "asc",
    set_count: bool = True,
    set_bar: bool = True,
) -> MongoDocsGenerator:
    generator = MongoDocsGenerator()
    generator.init_cli_args(
        jkvs={
            **MONGO_ENVS,
            "mongo_collection": "videos",
            "include_fields": args.include_fields,
            "skip_count": skip_count if skip_count is not None else args.skip_count,
            "max_count": max_count if max_count is not None else args.max_count,
            "estimate_count": args.estimate_count,
            "batch_size": args.batch_size,
        }
    )
    generator.init_all_with_cli_args(set_cursor=False, set_count=False, set_bar=False)
    generator.init_mongo_cursor(
        collection="videos",
        include_fields=args.include_fields.split(","),
        sort_index=sort_index,
        sort_order=sort_order,
        extra_filters=mongo_filter,
        skip_count=skip_count if skip_count is not None else args.skip_count,
        max_count=max_count if max_count is not None else args.max_count,
        estimate_count=args.estimate_count,
        batch_size=args.batch_size,
    )
    if set_count:
        generator.init_mongo_count()
    if set_bar:
        generator.init_progress_bar()
    if not set_count:
        generator.total_count = 1
    if not set_bar:
        generator.doc_bar = False
    return generator


def build_recorder(
    args: argparse.Namespace,
    generator: MongoDocsGenerator,
    dump_path: Path,
    *,
    min_freq: int = None,
    log_details: bool = True,
    progress_every_docs: int = SHARD_PROGRESS_EVERY_DOCS,
    progress_callback: Callable[[int], None] | None = None,
) -> WordsRecorder:
    if args.chinese:
        extractor = ChineseWordsExtractor()
    else:
        extractor = EnglishWordsExtractor()
    return WordsRecorder(
        generator=generator,
        extractor=extractor,
        text_fields=args.include_fields.split(","),
        min_freq=args.min_freq if min_freq is None else min_freq,
        dump_path=dump_path,
        log_details=log_details,
        progress_every_docs=progress_every_docs,
        progress_callback=progress_callback,
    )


def calc_output_doc_count(args: argparse.Namespace, total_count: int) -> int:
    if args.max_count is not None:
        return args.max_count
    return total_count - (args.skip_count or 0)


def build_shard_plans(
    args: argparse.Namespace,
    total_count: int,
    counter_generator: MongoDocsGenerator,
) -> list[ShardPlan]:
    if args.max_count is None and not (args.skip_count or 0):
        plans = build_region_shard_plans(counter_generator, workers=args.workers)
        if plans:
            return plans
    return build_skip_shard_plans(
        total_count=total_count,
        workers=args.workers,
        base_skip_count=args.skip_count or 0,
    )


def run_shard_worker(
    args_dict: dict,
    shard_idx: int,
    skip_count: int,
    max_count: int | None,
    dump_path: str,
    status_path: str,
    mongo_filter: dict | None = None,
):
    args = argparse.Namespace(**args_dict)
    shard_total = max_count or 0
    processed_docs = 0

    def on_progress(current_processed_docs: int):
        nonlocal processed_docs
        processed_docs = current_processed_docs
        write_shard_status(
            shard_status_path,
            shard_idx=shard_idx,
            state="running",
            processed_docs=current_processed_docs,
            total_docs=shard_total,
            skip_count=skip_count,
            max_count=shard_total,
            dump_path=partial_path,
        )

    shard_status_path = Path(status_path)
    partial_path = Path(dump_path)
    write_shard_status(
        shard_status_path,
        shard_idx=shard_idx,
        state="running",
        processed_docs=0,
        total_docs=shard_total,
        skip_count=skip_count,
        max_count=shard_total,
        dump_path=partial_path,
    )
    generator = build_generator(
        args,
        skip_count=skip_count,
        max_count=max_count,
        mongo_filter=mongo_filter,
        sort_index=None if mongo_filter else "insert_at",
        sort_order=None if mongo_filter else "asc",
        set_count=False,
        set_bar=False,
    )
    recorder = build_recorder(
        args,
        generator,
        partial_path,
        min_freq=1,
        log_details=False,
        progress_callback=on_progress,
    )
    recorder.run()
    write_shard_status(
        shard_status_path,
        shard_idx=shard_idx,
        state="done",
        processed_docs=processed_docs,
        total_docs=shard_total,
        skip_count=skip_count,
        max_count=shard_total,
        dump_path=partial_path,
    )
    return dump_path


def run_parallel(args: argparse.Namespace):
    counter_generator = build_generator(args, set_count=True, set_bar=False)
    total_count = counter_generator.total_count
    output_doc_count = calc_output_doc_count(args, total_count)
    dump_path = get_dump_path(
        docs_count=output_doc_count, lang=("zh" if args.chinese else "en")
    )
    shard_dir = dump_path.parent / "shards"
    status_dir = shard_dir / f"{dump_path.stem}_status"
    shard_dir.mkdir(parents=True, exist_ok=True)
    status_dir.mkdir(parents=True, exist_ok=True)

    shard_plans = build_shard_plans(
        args,
        total_count=(args.skip_count or 0) + output_doc_count,
        counter_generator=counter_generator,
    )
    args_dict = vars(args).copy()
    partial_paths = []
    strategy = shard_plans[0].get("strategy", "skip") if shard_plans else "skip"
    logger.note(
        f"> Run parallel word extraction: {logstr.file(args.workers)} workers ({strategy})"
    )
    for plan in shard_plans:
        if plan.get("regions"):
            count_text = (
                f" [{plan['max_count']}]" if plan.get("max_count") is not None else ""
            )
            logger.mesg(
                f"  * shard {plan['shard_idx']:02d}: {','.join(plan['regions'])}{count_text}"
            )
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_map = {}
        for plan in shard_plans:
            worker_idx = plan["shard_idx"]
            skip_count = plan.get("skip_count", 0)
            max_count = plan.get("max_count")
            partial_path = shard_dir / f"{dump_path.stem}.part_{worker_idx:02d}.csv"
            status_path = status_dir / f"{dump_path.stem}.part_{worker_idx:02d}.json"
            partial_paths.append(partial_path)
            status_path.unlink(missing_ok=True)
            future = executor.submit(
                run_shard_worker,
                args_dict,
                worker_idx,
                skip_count,
                max_count,
                str(partial_path),
                str(status_path),
                plan.get("mongo_filter"),
            )
            future_map[future] = status_path

        last_progress_log_at = 0.0
        pending_futures = set(future_map)
        while pending_futures:
            done_futures, pending_futures = wait(
                pending_futures,
                timeout=1.0,
                return_when=FIRST_COMPLETED,
            )
            for future in done_futures:
                completed_path = future.result()
                logger.okay(f"  * shard done: {completed_path}")

            now = time.monotonic()
            if now - last_progress_log_at >= SHARD_PROGRESS_LOG_INTERVAL:
                statuses = []
                for status_path in future_map.values():
                    status = load_shard_status(status_path)
                    if status is not None:
                        statuses.append(status)
                for line in format_shard_progress(statuses, workers=len(shard_plans)):
                    logger.line(line)
                last_progress_log_at = now

    merge_partial_records(
        partial_paths=partial_paths,
        dump_path=dump_path,
        sort_key="doc_freq",
        min_freq=args.min_freq,
    )


def run_single(args: argparse.Namespace):
    output_doc_count = args.max_count
    generator = build_generator(args, set_count=True, set_bar=True)
    if output_doc_count is None:
        output_doc_count = calc_output_doc_count(args, generator.total_count)
    dump_path = get_dump_path(
        docs_count=output_doc_count, lang=("zh" if args.chinese else "en")
    )
    recorder = build_recorder(args, generator, dump_path)
    recorder.run()


class RecordArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("-mf", "--min-freq", type=int, default=3)
        self.add_argument("-en", "--english", action="store_true")
        self.add_argument("-zh", "--chinese", action="store_true")
        self.add_argument("-j", "--workers", type=int, default=FULL_DATA_WORKERS)
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
        args.include_fields = "tags"
    else:
        args.include_fields = "title,tags,desc"

    logger.okay(args)
    if args.workers > 1 and args.max_count != 0:
        run_parallel(args)
    else:
        run_single(args)


if __name__ == "__main__":
    arg_parser = MergedArgParser(RecordArgParser, MongoDocsGeneratorArgParser)
    args = arg_parser.parse_args()
    main(args)

    # Case: max-count docs
    # python -m models.word.eng -ec -en -mn 50000
    # python -m models.word.eng -ec -zh -mf 6 -mn 5000000

    # Case: all docs
    # python -m models.word.eng -ec -en -mf 6
    # python -m models.word.eng -ec -zh -mf 6
