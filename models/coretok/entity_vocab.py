import csv
import math
import pickle
import re

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from models.word.eng import EnglishWordsExtractor


ENTITY_SPLIT_RE = re.compile(r"[，,。.!！?？、;；:：()（）\[\]【】<>《》\-\|/\\\s]+")
GENERIC_TOKENS = {
    "这个",
    "那个",
    "直播间",
    "聊天",
    "视频",
    "教程",
    "礼物",
    "推荐",
    "解决",
    "安排",
    "省心",
    "真的",
    "一定",
}
ZH_POS_ALLOWLIST = {
    "名词",
    "人名",
    "地名",
    "其他专名",
    "机构团体",
    "简称略语",
    "习用语",
    "成语",
    "名动词",
    "外文字符",
}


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _compact_text(text: str) -> str:
    return _normalize_text(text).replace(" ", "")


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in (text or ""))


def _contains_ascii_alnum(text: str) -> bool:
    return any(char.isascii() and char.isalnum() for char in (text or ""))


def _load_csv_rows(path: Path):
    with path.open("r", encoding="utf-8") as file_obj:
        yield from csv.DictReader(file_obj)


@dataclass(frozen=True)
class EntityMatch:
    token: str
    score: float
    start: int
    end: int
    source: str


class CoreEntityVocab:
    def __init__(
        self,
        *,
        zh_scores: dict[str, float],
        en_scores: dict[str, float],
        zh_tokens_by_len: dict[int, set[str]],
        zh_lengths_by_first_char: dict[str, tuple[int, ...]],
    ):
        self.zh_scores = zh_scores
        self.en_scores = en_scores
        self.zh_tokens_by_len = zh_tokens_by_len
        self.zh_lengths_by_first_char = zh_lengths_by_first_char
        self.english_extractor = EnglishWordsExtractor()

    @classmethod
    def from_records(
        cls,
        *,
        zh_records: list[tuple[str, int, int]] | None = None,
        en_records: list[tuple[str, int, int]] | None = None,
    ) -> "CoreEntityVocab":
        zh_scores = {}
        en_scores = {}
        zh_tokens_by_len: dict[int, set[str]] = {}
        zh_lengths_by_first_char: dict[str, set[int]] = {}

        for token, doc_freq, term_freq in zh_records or []:
            normalized = _normalize_text(token)
            if not normalized:
                continue
            score = cls._score_from_freq(doc_freq, term_freq, source="zh")
            zh_scores[normalized] = score
            compact = _compact_text(normalized)
            token_len = len(compact)
            zh_tokens_by_len.setdefault(token_len, set()).add(compact)
            zh_lengths_by_first_char.setdefault(compact[:1], set()).add(token_len)

        for token, doc_freq, term_freq in en_records or []:
            normalized = _normalize_text(token)
            if not normalized:
                continue
            en_scores[normalized] = cls._score_from_freq(
                doc_freq, term_freq, source="en"
            )

        return cls(
            zh_scores=zh_scores,
            en_scores=en_scores,
            zh_tokens_by_len=zh_tokens_by_len,
            zh_lengths_by_first_char={
                key: tuple(sorted(lengths, reverse=True))
                for key, lengths in zh_lengths_by_first_char.items()
            },
        )

    @staticmethod
    def _score_from_freq(doc_freq: int, term_freq: int, *, source: str) -> float:
        base = math.log1p(max(doc_freq, 1)) * 0.7 + math.log1p(max(term_freq, 1)) * 0.3
        if source == "en":
            base += 0.3
        return round(base, 4)

    @classmethod
    def build_default(cls) -> "CoreEntityVocab":
        repo_root = Path(__file__).resolve().parents[2]
        zh_path = repo_root / "models" / "word" / "eng" / "zh_freq_770000000.csv"
        en_path = repo_root / "models" / "word" / "eng" / "en_freq_770000000.csv"
        cache_path = repo_root / "data" / "token_freqs" / "coretok_entity_vocab.pkl"

        if cache_path.exists() and cache_path.stat().st_mtime >= max(
            zh_path.stat().st_mtime, en_path.stat().st_mtime
        ):
            with cache_path.open("rb") as file_obj:
                payload = pickle.load(file_obj)
            return cls(
                zh_scores=payload["zh_scores"],
                en_scores=payload["en_scores"],
                zh_tokens_by_len=payload["zh_tokens_by_len"],
                zh_lengths_by_first_char=payload["zh_lengths_by_first_char"],
            )

        zh_scores = {}
        en_scores = {}
        zh_tokens_by_len: dict[int, set[str]] = {}
        zh_lengths_by_first_char: dict[str, set[int]] = {}

        for row in _load_csv_rows(zh_path):
            token = _normalize_text(row.get("word") or "")
            if not token or token in GENERIC_TOKENS:
                continue
            doc_freq = int(row.get("doc_freq") or 0)
            term_freq = int(row.get("term_freq") or 0)
            if doc_freq < 300:
                continue
            compact = _compact_text(token)
            if not compact or not _contains_cjk(compact):
                continue
            if len(compact) < 2 or len(compact) > 8:
                continue
            score = cls._score_from_freq(doc_freq, term_freq, source="zh")
            zh_scores[token] = score
            zh_tokens_by_len.setdefault(len(compact), set()).add(compact)
            zh_lengths_by_first_char.setdefault(compact[:1], set()).add(len(compact))

        for row in _load_csv_rows(en_path):
            token = _normalize_text(row.get("word") or "")
            if not token:
                continue
            doc_freq = int(row.get("doc_freq") or 0)
            term_freq = int(row.get("term_freq") or 0)
            if doc_freq < 300:
                continue
            if len(token) < 2 or len(token) > 40:
                continue
            if not _contains_ascii_alnum(token):
                continue
            en_scores[token] = cls._score_from_freq(doc_freq, term_freq, source="en")

        payload = {
            "zh_scores": zh_scores,
            "en_scores": en_scores,
            "zh_tokens_by_len": zh_tokens_by_len,
            "zh_lengths_by_first_char": {
                key: tuple(sorted(lengths, reverse=True))
                for key, lengths in zh_lengths_by_first_char.items()
            },
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as file_obj:
            pickle.dump(payload, file_obj)

        return cls(
            zh_scores=payload["zh_scores"],
            en_scores=payload["en_scores"],
            zh_tokens_by_len=payload["zh_tokens_by_len"],
            zh_lengths_by_first_char=payload["zh_lengths_by_first_char"],
        )

    def lookup_score(self, token: str) -> float:
        normalized = _normalize_text(token)
        return max(
            self.zh_scores.get(normalized, 0.0),
            self.en_scores.get(normalized, 0.0),
        )

    def extract_candidates(
        self,
        text: str,
        *,
        max_candidates: int = 12,
    ) -> list[str]:
        normalized = _normalize_text(text)
        if not normalized:
            return []

        matches: list[EntityMatch] = []
        seen = set()

        for token in self.english_extractor.extract(normalized):
            normalized_token = _normalize_text(token)
            if not normalized_token:
                continue
            score = self.en_scores.get(normalized_token, 0.0)
            if score <= 0.0 and len(normalized_token) < 3:
                continue
            start = normalized.find(normalized_token)
            match = EntityMatch(
                token=normalized_token,
                score=score or 0.2,
                start=max(start, 0),
                end=max(start, 0) + len(normalized_token),
                source="en",
            )
            key = (match.token, match.start, match.end)
            if key not in seen:
                seen.add(key)
                matches.append(match)

        for chunk in ENTITY_SPLIT_RE.split(normalized):
            compact = _compact_text(chunk)
            if not compact:
                continue
            offset = normalized.find(chunk)
            for index, char in enumerate(compact):
                lengths = self.zh_lengths_by_first_char.get(char)
                if not lengths:
                    continue
                for token_len in lengths:
                    end = index + token_len
                    if end > len(compact):
                        continue
                    candidate = compact[index:end]
                    token_set = self.zh_tokens_by_len.get(token_len) or set()
                    if candidate not in token_set:
                        continue
                    score = self.zh_scores.get(candidate, 0.0)
                    if score <= 0.0:
                        continue
                    match = EntityMatch(
                        token=candidate,
                        score=score,
                        start=max(offset, 0) + index,
                        end=max(offset, 0) + end,
                        source="zh",
                    )
                    key = (match.token, match.start, match.end)
                    if key not in seen:
                        seen.add(key)
                        matches.append(match)

        matches.sort(
            key=lambda item: (
                item.score,
                len(item.token),
                -item.start,
            ),
            reverse=True,
        )

        merged: list[str] = []
        for match in matches:
            if match.token in merged:
                continue
            merged.append(match.token)
            if len(merged) >= max_candidates:
                break
        return merged


@lru_cache(maxsize=1)
def get_default_core_entity_vocab() -> CoreEntityVocab:
    return CoreEntityVocab.build_default()
