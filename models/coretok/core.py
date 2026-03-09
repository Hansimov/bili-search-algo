import math
import re
import time
import zlib

from collections import Counter
from dataclasses import dataclass, field

try:
    import numpy as np
except ImportError:
    np = None


LATIN_CHAR_RE = re.compile(r"[a-z0-9]")
LATIN_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_\-\.]*")
CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")
CJK_SPAN_RE = re.compile(r"[\u4e00-\u9fff]+")
CORETEXT_SPLIT_RE = re.compile(r"[，,。.!！?？、;；:：()（）\[\]【】<>《》\-\|/\\\s]+")
SIGNATURE_WORD_COUNT = 4
SIGNATURE_BIT_COUNT = SIGNATURE_WORD_COUNT * 64
SIGNATURE_SHORTLIST_SIZE = 96
SIGNATURE_SHORTLIST_MIN_CANDIDATES = 128
POPCOUNT_TABLE = (
    np.array([bin(index).count("1") for index in range(256)], dtype=np.uint8)
    if np is not None
    else None
)


def normalize_core_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _compact_normalized_text(text: str) -> str:
    return (text or "").replace(" ", "")


def _compact_core_text(text: str) -> str:
    return _compact_normalized_text(normalize_core_text(text))


def _is_degenerate_text(text: str) -> bool:
    normalized = normalize_core_text(text)
    if not normalized:
        return True
    unique_chars = {char for char in normalized if not char.isspace()}
    return len(unique_chars) <= 1


def count_mixed_units(text: str) -> int:
    normalized = normalize_core_text(text)
    if not normalized:
        return 0

    cjk_chars = len(CJK_CHAR_RE.findall(normalized))
    latin_chars = len(LATIN_CHAR_RE.findall(normalized))
    return cjk_chars + (math.ceil(latin_chars / 3) if latin_chars else 0)


def count_cjk_chars(text: str) -> int:
    return len(CJK_CHAR_RE.findall(normalize_core_text(text)))


def count_latin_chars(text: str) -> int:
    return len(LATIN_CHAR_RE.findall(normalize_core_text(text)))


@dataclass
class CoreCorpusStats:
    total_docs: int = 0
    candidate_doc_freqs: Counter = field(default_factory=Counter)
    stop_candidates: set[str] = field(default_factory=set)
    stop_quantile: float = 0.98
    min_docs_for_stop: int = 4
    stop_coverage_floor: float = 0.005
    stop_coverage_threshold: float = 0.0
    max_stop_candidates: int = 64

    def fit(
        self,
        texts: list[str] | None = None,
        *,
        for_stage1: bool,
        text_counter: Counter | None = None,
    ) -> "CoreCorpusStats":
        self.total_docs = 0
        self.candidate_doc_freqs = Counter()
        self.stop_candidates = set()
        self.stop_coverage_threshold = 0.0

        if text_counter is None:
            text_counter = Counter(
                text for text in (texts or []) if normalize_core_text(text)
            )

        for text, freq in text_counter.items():
            normalized = normalize_core_text(text)
            if _is_degenerate_text(normalized):
                continue
            candidates = set(
                extract_core_candidates(
                    normalized,
                    for_stage1=for_stage1,
                    corpus_stats=None,
                )
            )
            if not candidates:
                continue
            self.total_docs += int(freq)
            for candidate in candidates:
                self.candidate_doc_freqs[candidate] += int(freq)

        short_candidates = sorted(
            [
                (
                    candidate,
                    freq / max(self.total_docs, 1),
                    int(freq),
                )
                for candidate, freq in self.candidate_doc_freqs.items()
                if 2 <= count_mixed_units(candidate) <= 4
            ],
            key=lambda item: (item[1], item[2], len(item[0])),
            reverse=True,
        )
        if short_candidates:
            threshold_index = min(
                max(int(len(short_candidates) * self.stop_quantile), 0),
                len(short_candidates) - 1,
            )
            stop_limit = min(
                max(int(math.sqrt(max(self.total_docs, 1)) / 8), 8),
                self.max_stop_candidates,
                len(short_candidates),
            )
            self.stop_coverage_threshold = max(
                self.stop_coverage_floor,
                short_candidates[threshold_index][1],
                short_candidates[stop_limit - 1][1],
            )

        for candidate, coverage, freq in short_candidates[: self.max_stop_candidates]:
            if (
                freq >= self.min_docs_for_stop
                and coverage >= self.stop_coverage_threshold
            ):
                self.stop_candidates.add(candidate)

        for candidate, freq in self.candidate_doc_freqs.items():
            if candidate in self.stop_candidates:
                continue
            coverage = self.coverage(candidate)
            if (
                2 <= count_mixed_units(candidate) <= 4
                and freq >= self.min_docs_for_stop
                and coverage >= self.stop_coverage_threshold
            ):
                self.stop_candidates.add(candidate)
        return self

    def coverage(self, candidate: str) -> float:
        normalized = normalize_core_text(candidate)
        if not normalized or self.total_docs <= 0:
            return 0.0
        return self.candidate_doc_freqs.get(normalized, 0) / max(self.total_docs, 1)

    def is_stop_candidate(self, candidate: str) -> bool:
        return normalize_core_text(candidate) in self.stop_candidates

    def to_dict(self) -> dict:
        return {
            "total_docs": int(self.total_docs),
            "candidate_doc_freqs": {
                token: int(freq) for token, freq in self.candidate_doc_freqs.items()
            },
            "stop_candidates": sorted(self.stop_candidates),
            "stop_quantile": float(self.stop_quantile),
            "min_docs_for_stop": int(self.min_docs_for_stop),
            "stop_coverage_floor": float(self.stop_coverage_floor),
            "stop_coverage_threshold": float(self.stop_coverage_threshold),
            "max_stop_candidates": int(self.max_stop_candidates),
        }

    @classmethod
    def from_dict(cls, payload: dict | None) -> "CoreCorpusStats":
        payload = payload or {}
        return cls(
            total_docs=int(payload.get("total_docs") or 0),
            candidate_doc_freqs=Counter(
                {
                    str(token): int(freq)
                    for token, freq in (
                        payload.get("candidate_doc_freqs") or {}
                    ).items()
                }
            ),
            stop_candidates=set(payload.get("stop_candidates") or []),
            stop_quantile=float(payload.get("stop_quantile") or 0.98),
            min_docs_for_stop=int(payload.get("min_docs_for_stop") or 4),
            stop_coverage_floor=float(payload.get("stop_coverage_floor") or 0.005),
            stop_coverage_threshold=float(
                payload.get("stop_coverage_threshold") or 0.0
            ),
            max_stop_candidates=int(payload.get("max_stop_candidates") or 64),
        )


def is_low_info_text(text: str, corpus_stats: CoreCorpusStats | None = None) -> bool:
    normalized = normalize_core_text(text)
    if _is_degenerate_text(normalized):
        return True
    if corpus_stats is not None and corpus_stats.is_stop_candidate(normalized):
        return True
    return False


def is_valid_stage1_tag(
    tag: str,
    corpus_stats: CoreCorpusStats | None = None,
) -> bool:
    normalized = normalize_core_text(tag)
    if not normalized or is_low_info_text(normalized, corpus_stats=corpus_stats):
        return False
    if count_mixed_units(normalized) > 8:
        return False
    if count_cjk_chars(normalized) > 8:
        return False
    if count_latin_chars(normalized) > 24:
        return False
    return bool(CJK_CHAR_RE.search(normalized) or LATIN_CHAR_RE.search(normalized))


def suggest_token_budget(text: str) -> int:
    units = count_mixed_units(text)
    if units <= 0:
        return 0
    if units <= 3:
        return 1
    if units <= 4:
        return 1
    if units <= 6:
        return 2
    return 3


def build_candidate_plan(
    text: str,
    *,
    for_stage1: bool,
    corpus_stats: CoreCorpusStats | None = None,
) -> dict:
    normalized = normalize_core_text(text)
    if not normalized:
        return {"text": "", "budget": 0, "candidates": []}

    if for_stage1:
        budget = suggest_token_budget(normalized)
    else:
        budget = max(1, min(6, math.ceil(count_mixed_units(normalized) / 3)))

    return {
        "text": normalized,
        "budget": budget,
        "candidates": extract_core_candidates(
            normalized,
            for_stage1=for_stage1,
            corpus_stats=corpus_stats,
        ),
    }


def _char_ngrams_from_compact(compact: str) -> set[str]:
    if len(compact) <= 2:
        return {compact} if compact else set()
    grams = {compact[index : index + 2] for index in range(len(compact) - 1)}
    grams.add(compact)
    return grams


def _char_ngrams(text: str) -> set[str]:
    return _char_ngrams_from_compact(_compact_core_text(text))


def _surface_overlap_score_with_features(
    left: str,
    right: str,
    *,
    compact_left: str | None = None,
    compact_right: str | None = None,
    left_grams: set[str] | frozenset[str] | None = None,
    right_grams: set[str] | frozenset[str] | None = None,
) -> float:
    if left == right:
        return 1.0

    compact_left = (
        compact_left if compact_left is not None else _compact_core_text(left)
    )
    compact_right = (
        compact_right if compact_right is not None else _compact_core_text(right)
    )
    if compact_left and compact_right:
        shorter = min(len(compact_left), len(compact_right))
        longer = max(len(compact_left), len(compact_right))
        containment = 0.0
        if compact_left in compact_right or compact_right in compact_left:
            containment = shorter / max(longer, 1)
    else:
        containment = 0.0

    left_grams = (
        left_grams
        if left_grams is not None
        else _char_ngrams_from_compact(compact_left)
    )
    right_grams = (
        right_grams
        if right_grams is not None
        else _char_ngrams_from_compact(compact_right)
    )
    if not left_grams or not right_grams:
        return containment

    inter = len(left_grams & right_grams)
    union = len(left_grams | right_grams)
    jaccard = inter / union if union else 0.0
    return max(jaccard, containment)


def _signature_words(text: str) -> tuple[int, ...]:
    words = [0] * SIGNATURE_WORD_COUNT
    for gram in _char_ngrams(text):
        if not gram:
            continue
        bucket = zlib.crc32(gram.encode("utf-8")) % SIGNATURE_BIT_COUNT
        word_index = bucket // 64
        bit_index = bucket % 64
        words[word_index] |= 1 << bit_index
    return tuple(words)


def _surface_overlap_score(left: str, right: str) -> float:
    return _surface_overlap_score_with_features(left, right)


def _balanced_cjk_parts(span: str) -> list[str]:
    compact = span.strip()
    if len(compact) <= 4:
        return [compact]
    if len(compact) <= 6:
        pivot = len(compact) // 2
        return [compact[:pivot], compact[pivot:]]

    part_size = math.ceil(len(compact) / 3)
    return [
        compact[index : index + part_size]
        for index in range(0, len(compact), part_size)
        if compact[index : index + part_size]
    ]


def extract_core_candidates(
    text: str,
    *,
    for_stage1: bool,
    corpus_stats: CoreCorpusStats | None = None,
) -> list[str]:
    normalized = normalize_core_text(text)
    if not normalized:
        return []

    candidates = []
    seen = set()

    def add(value: str):
        candidate = normalize_core_text(value)
        if (
            not candidate
            or candidate in seen
            or is_low_info_text(candidate, corpus_stats=corpus_stats)
        ):
            return
        seen.add(candidate)
        candidates.append(candidate)

    add(normalized)

    for token in LATIN_TOKEN_RE.findall(normalized):
        if len(token) >= 2:
            add(token)

    cjk_source = [normalized] if for_stage1 else CORETEXT_SPLIT_RE.split(normalized)
    for chunk in cjk_source:
        for span in CJK_SPAN_RE.findall(chunk):
            if len(span) < 2:
                continue
            if len(span) <= 8:
                add(span)
            for part in _balanced_cjk_parts(span):
                if 2 <= len(part) <= 8:
                    add(part)

    return candidates


@dataclass
class CoreTokenLexicon:
    token_to_id: dict[str, int] = field(default_factory=dict)
    id_to_token: dict[int, str] = field(default_factory=dict)
    token_freqs: Counter = field(default_factory=Counter)
    token_sources: dict[int, str] = field(default_factory=dict)
    token_units: dict[int, int] = field(default_factory=dict)
    next_token_id: int = 1
    gram_to_ids: dict[str, set[int]] = field(default_factory=dict)
    first_char_to_ids: dict[str, set[int]] = field(default_factory=dict)
    match_cache: dict[str, tuple[int | None, float]] = field(default_factory=dict)
    token_signature_words: dict[int, tuple[int, ...]] = field(default_factory=dict)
    token_compact_texts: dict[int, str] = field(default_factory=dict)
    token_char_grams: dict[int, frozenset[str]] = field(default_factory=dict)
    np_token_units: object | None = field(default=None, repr=False)
    np_token_signatures: object | None = field(default=None, repr=False)
    np_capacity: int = 0

    def _ensure_numpy_capacity(self, token_id: int):
        if np is None:
            return
        required_size = int(token_id) + 1
        if (
            self.np_token_units is not None
            and self.np_token_signatures is not None
            and self.np_capacity >= required_size
        ):
            return

        capacity = max(self.np_capacity or 32, 32)
        while capacity < required_size:
            capacity *= 2

        new_units = np.zeros(capacity, dtype=np.int16)
        new_signatures = np.zeros(
            (capacity, SIGNATURE_WORD_COUNT),
            dtype=np.uint64,
        )
        if self.np_token_units is not None and self.np_token_signatures is not None:
            new_units[: self.np_capacity] = self.np_token_units[: self.np_capacity]
            new_signatures[: self.np_capacity] = self.np_token_signatures[
                : self.np_capacity
            ]

        self.np_token_units = new_units
        self.np_token_signatures = new_signatures
        self.np_capacity = capacity

    def _index_token(self, token_id: int, token: str):
        token_compact = _compact_normalized_text(token)
        token_grams = frozenset(_char_ngrams_from_compact(token_compact))
        token_units = count_mixed_units(token)
        token_signature_words = _signature_words(token)
        for gram in token_grams:
            if not gram:
                continue
            self.gram_to_ids.setdefault(gram, set()).add(token_id)
        first_char = token[:1]
        if first_char:
            self.first_char_to_ids.setdefault(first_char, set()).add(token_id)
        self.token_units[token_id] = token_units
        self.token_signature_words[token_id] = token_signature_words
        self.token_compact_texts[token_id] = token_compact
        self.token_char_grams[token_id] = token_grams
        self._ensure_numpy_capacity(token_id)
        if self.np_token_units is not None and self.np_token_signatures is not None:
            self.np_token_units[token_id] = token_units
            self.np_token_signatures[token_id] = token_signature_words

    def _shortlist_candidate_ids(
        self,
        normalized: str,
        candidate_ids: set[int],
        normalized_units: int,
    ) -> list[int]:
        if (
            np is None
            or POPCOUNT_TABLE is None
            or len(candidate_ids) < SIGNATURE_SHORTLIST_MIN_CANDIDATES
        ):
            return list(candidate_ids)

        ids_array = np.fromiter(candidate_ids, dtype=np.int32)
        if self.np_token_units is not None:
            unit_array = self.np_token_units[ids_array]
        else:
            unit_array = np.fromiter(
                (self.token_units.get(int(token_id), 0) for token_id in ids_array),
                dtype=np.int16,
                count=ids_array.size,
            )
        unit_mask = np.abs(unit_array - normalized_units) <= 4
        filtered_ids = ids_array[unit_mask]
        if filtered_ids.size == 0:
            filtered_ids = ids_array
        if filtered_ids.size <= SIGNATURE_SHORTLIST_SIZE:
            return filtered_ids.astype(int).tolist()

        if self.np_token_signatures is not None:
            signature_matrix = self.np_token_signatures[filtered_ids]
        else:
            signature_matrix = np.asarray(
                [
                    self.token_signature_words[int(token_id)]
                    for token_id in filtered_ids
                ],
                dtype=np.uint64,
            )
        query_signature = np.asarray(_signature_words(normalized), dtype=np.uint64)
        and_bytes = np.bitwise_and(signature_matrix, query_signature).view(np.uint8)
        or_bytes = np.bitwise_or(signature_matrix, query_signature).view(np.uint8)
        intersections = POPCOUNT_TABLE[and_bytes].sum(axis=1, dtype=np.uint16)
        unions = POPCOUNT_TABLE[or_bytes].sum(axis=1, dtype=np.uint16)
        approx_scores = intersections / np.maximum(unions, 1)

        shortlist_size = min(SIGNATURE_SHORTLIST_SIZE, filtered_ids.size)
        shortlist_indices = np.argpartition(approx_scores, -shortlist_size)[
            -shortlist_size:
        ]
        shortlist_ids = filtered_ids[shortlist_indices]
        return shortlist_ids.astype(int).tolist()

    def add_token(self, token: str, source: str) -> int:
        normalized = normalize_core_text(token)
        if not normalized:
            raise ValueError("Cannot add empty token")
        existing = self.token_to_id.get(normalized)
        if existing is not None:
            self.token_freqs[existing] += 1
            if self.token_sources.get(existing) == "text" and source == "tag":
                self.token_sources[existing] = "tag"
            return existing

        token_id = self.next_token_id
        self.next_token_id += 1
        self.token_to_id[normalized] = token_id
        self.id_to_token[token_id] = normalized
        self.token_sources[token_id] = source
        self.token_freqs[token_id] += 1
        self._index_token(token_id, normalized)
        self.match_cache.clear()
        return token_id

    def touch_token(self, token_id: int, delta: int = 1) -> int:
        if delta <= 0:
            return token_id
        self.token_freqs[token_id] += delta
        return token_id

    def get_token_id(self, token: str) -> int | None:
        return self.token_to_id.get(normalize_core_text(token))

    def decode(self, token_ids: list[int]) -> list[str]:
        return [
            self.id_to_token[token_id]
            for token_id in token_ids
            if token_id in self.id_to_token
        ]

    def find_best_match(self, token: str) -> tuple[int | None, float]:
        normalized = normalize_core_text(token)
        exact_id = self.token_to_id.get(normalized)
        if exact_id is not None:
            return exact_id, 1.0

        cached = self.match_cache.get(normalized)
        if cached is not None:
            return cached

        best_token_id = None
        best_score = 0.0
        gram_postings = []
        for gram in _char_ngrams(normalized):
            ids = self.gram_to_ids.get(gram)
            if ids:
                gram_postings.append((len(ids), ids))

        candidate_ids = set()
        for _, ids in sorted(gram_postings, key=lambda item: item[0])[:3]:
            candidate_ids.update(ids)
        if not candidate_ids:
            first_char = normalized[:1]
            candidate_ids = set(self.first_char_to_ids.get(first_char) or ())
        if not candidate_ids:
            candidate_ids = set(self.id_to_token.keys())

        normalized_units = count_mixed_units(normalized)
        normalized_compact = _compact_normalized_text(normalized)
        normalized_grams = _char_ngrams_from_compact(normalized_compact)
        shortlisted_candidate_ids = self._shortlist_candidate_ids(
            normalized,
            candidate_ids,
            normalized_units,
        )
        if shortlisted_candidate_ids:
            candidate_ids = shortlisted_candidate_ids
        else:
            candidate_ids = {
                token_id
                for token_id in candidate_ids
                if abs(self.token_units.get(token_id, 0) - normalized_units) <= 4
            } or candidate_ids

        for token_id in candidate_ids:
            existing = self.id_to_token.get(token_id)
            if not existing:
                continue
            score = _surface_overlap_score_with_features(
                normalized,
                existing,
                compact_left=normalized_compact,
                compact_right=self.token_compact_texts.get(token_id),
                left_grams=normalized_grams,
                right_grams=self.token_char_grams.get(token_id),
            )
            if score > best_score:
                best_token_id = token_id
                best_score = score
        self.match_cache[normalized] = (best_token_id, best_score)
        return best_token_id, best_score

    def novelty_probability(self, token: str) -> float:
        _, best_score = self.find_best_match(token)
        return round(1.0 - best_score, 4)

    def to_dict(self) -> dict:
        return {
            "token_to_id": dict(self.token_to_id),
            "id_to_token": {
                str(token_id): token for token_id, token in self.id_to_token.items()
            },
            "token_freqs": {
                str(token_id): int(freq) for token_id, freq in self.token_freqs.items()
            },
            "token_sources": {
                str(token_id): source for token_id, source in self.token_sources.items()
            },
            "token_units": {
                str(token_id): int(units)
                for token_id, units in self.token_units.items()
            },
            "next_token_id": int(self.next_token_id),
        }

    @classmethod
    def from_dict(cls, payload: dict | None) -> "CoreTokenLexicon":
        payload = payload or {}
        token_to_id = {
            str(token): int(token_id)
            for token, token_id in (payload.get("token_to_id") or {}).items()
        }
        id_to_token = {
            int(token_id): str(token)
            for token_id, token in (payload.get("id_to_token") or {}).items()
        }
        token_freqs = Counter(
            {
                int(token_id): int(freq)
                for token_id, freq in (payload.get("token_freqs") or {}).items()
            }
        )
        token_sources = {
            int(token_id): str(source)
            for token_id, source in (payload.get("token_sources") or {}).items()
        }
        token_units = {
            int(token_id): int(units)
            for token_id, units in (payload.get("token_units") or {}).items()
        }
        next_token_id = int(
            payload.get("next_token_id") or (max(id_to_token.keys(), default=0) + 1)
        )
        lexicon = cls(
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            token_freqs=token_freqs,
            token_sources=token_sources,
            token_units=token_units,
            next_token_id=next_token_id,
        )
        for token_id, token in id_to_token.items():
            lexicon._index_token(token_id, token)
        return lexicon


class _BaseCoreTokenizer:
    def __init__(
        self,
        *,
        lexicon: CoreTokenLexicon | None = None,
        corpus_stats: CoreCorpusStats | None = None,
        novelty_threshold: float,
        reuse_threshold: float,
        source: str,
    ):
        self.lexicon = lexicon or CoreTokenLexicon()
        self.corpus_stats = corpus_stats
        self.novelty_threshold = novelty_threshold
        self.reuse_threshold = reuse_threshold
        self.source = source
        self._candidate_plan_cache: dict[tuple[str, bool], dict] = {}
        self.last_fit_stats: dict = {}

    def _score_candidate(self, candidate: str, budget: int) -> tuple[float, int, str]:
        whole_bonus = 4.0 if count_mixed_units(candidate) <= 8 else 0.0
        freq_bonus = 0.0
        existing_id = self.lexicon.get_token_id(candidate)
        if existing_id is not None:
            freq_bonus = min(float(self.lexicon.token_freqs.get(existing_id, 0)), 8.0)
        length_bonus = min(count_mixed_units(candidate), budget * 2)
        return (whole_bonus + freq_bonus + length_bonus, len(candidate), candidate)

    def _get_candidate_plan(
        self,
        text: str,
        *,
        for_stage1: bool,
        prepared_plan: dict | None = None,
    ) -> dict:
        normalized = normalize_core_text(text)
        cache_key = (normalized, for_stage1)
        if prepared_plan is not None:
            plan = {
                "text": normalize_core_text(prepared_plan.get("text") or normalized),
                "budget": int(prepared_plan.get("budget") or 0),
                "candidates": list(prepared_plan.get("candidates") or []),
            }
            self._candidate_plan_cache[cache_key] = plan
            return plan

        plan = self._candidate_plan_cache.get(cache_key)
        if plan is None:
            plan = build_candidate_plan(
                normalized,
                for_stage1=for_stage1,
                corpus_stats=self.corpus_stats,
            )
            self._candidate_plan_cache[cache_key] = plan
        return plan

    def _choose_candidates(
        self,
        text: str,
        *,
        budget: int,
        for_stage1: bool,
        prepared_plan: dict | None = None,
    ) -> list[str]:
        plan = self._get_candidate_plan(
            text,
            for_stage1=for_stage1,
            prepared_plan=prepared_plan,
        )
        candidates = plan["candidates"]
        ranked = sorted(
            candidates,
            key=lambda candidate: self._score_candidate(candidate, budget),
            reverse=True,
        )
        return ranked[:budget]

    def _allow_approximate_reuse(
        self,
        candidate: str,
        existing_token: str,
        *,
        source_text: str | None,
        best_score: float,
    ) -> bool:
        if best_score >= 0.98:
            return True
        if not source_text:
            return False
        compact_source = _compact_core_text(source_text)
        compact_existing = _compact_core_text(existing_token)
        compact_candidate = _compact_core_text(candidate)
        if not compact_source or not compact_existing:
            return False
        if compact_existing in compact_source:
            return True
        if compact_candidate and compact_candidate in compact_existing:
            return False
        return False

    def _materialize_token(
        self,
        candidate: str,
        *,
        allow_new_tokens: bool,
        source_text: str | None = None,
    ) -> int | None:
        existing_id = self.lexicon.get_token_id(candidate)
        if existing_id is not None:
            return self.lexicon.touch_token(existing_id)

        best_token_id, best_score = self.lexicon.find_best_match(candidate)
        can_reuse_best = False
        if best_token_id is not None and best_score >= self.reuse_threshold:
            existing_token = self.lexicon.id_to_token.get(best_token_id) or ""
            can_reuse_best = self._allow_approximate_reuse(
                candidate,
                existing_token,
                source_text=source_text,
                best_score=best_score,
            )
            if can_reuse_best:
                return self.lexicon.touch_token(best_token_id)

        if not allow_new_tokens:
            return None

        novelty_probability = round(1.0 - best_score, 4)
        if (
            best_token_id is not None
            and can_reuse_best
            and novelty_probability < self.novelty_threshold
        ):
            return best_token_id
        return self.lexicon.add_token(candidate, source=self.source)

    def decode(self, token_ids: list[int]) -> list[str]:
        return self.lexicon.decode(token_ids)


class CoreTagTokenizer(_BaseCoreTokenizer):
    def __init__(
        self,
        *,
        lexicon: CoreTokenLexicon | None = None,
        corpus_stats: CoreCorpusStats | None = None,
    ):
        super().__init__(
            lexicon=lexicon,
            corpus_stats=corpus_stats,
            novelty_threshold=0.55,
            reuse_threshold=0.5,
            source="tag",
        )

    def encode(
        self,
        tag: str,
        *,
        allow_new_tokens: bool = False,
        candidate_plan: dict | None = None,
    ) -> list[int]:
        if not is_valid_stage1_tag(tag, corpus_stats=self.corpus_stats):
            return []
        budget = suggest_token_budget(tag)
        token_ids = []
        for candidate in self._choose_candidates(
            tag,
            budget=budget,
            for_stage1=True,
            prepared_plan=candidate_plan,
        ):
            token_id = self._materialize_token(
                candidate,
                allow_new_tokens=allow_new_tokens,
                source_text=tag,
            )
            if token_id is not None and token_id not in token_ids:
                token_ids.append(token_id)
            if len(token_ids) >= budget:
                break
        return token_ids

    def fit(
        self,
        tags: list[str] | None = None,
        epochs: int = 3,
        tag_counter: Counter | None = None,
        frequency_items: list[tuple[str, int]] | None = None,
        candidate_plans: dict[str, dict] | None = None,
    ) -> list[list[int]]:
        encoded = []
        if frequency_items is None:
            tag_counter = tag_counter or Counter(
                tag for tag in (tags or []) if normalize_core_text(tag)
            )
            frequency_items = tag_counter.most_common()
        for _ in range(max(epochs, 1)):
            for tag, freq in frequency_items:
                token_ids = self.encode(
                    tag,
                    allow_new_tokens=True,
                    candidate_plan=(candidate_plans or {}).get(
                        normalize_core_text(tag)
                    ),
                )
                if token_ids:
                    encoded.extend([token_ids] * int(freq))
                    extra_repeats = max(int(freq) - 1, 0)
                    for token_id in token_ids:
                        self.lexicon.touch_token(token_id, delta=extra_repeats)
        return encoded


class CoreTexTokenizer(_BaseCoreTokenizer):
    def __init__(
        self,
        *,
        lexicon: CoreTokenLexicon | None = None,
        corpus_stats: CoreCorpusStats | None = None,
    ):
        super().__init__(
            lexicon=lexicon,
            corpus_stats=corpus_stats,
            novelty_threshold=0.82,
            reuse_threshold=0.6,
            source="text",
        )
        self.base_match_token_ids = (
            set((lexicon.id_to_token or {}).keys()) if lexicon else set()
        )
        self.base_gram_to_ids: dict[str, set[int]] = {}
        self.base_first_char_to_ids: dict[str, set[int]] = {}
        self.base_match_cache: dict[str, tuple[int | None, float]] = {}
        for token_id in self.base_match_token_ids:
            token = self.lexicon.id_to_token.get(token_id)
            if not token:
                continue
            for gram in _char_ngrams(token):
                if gram:
                    self.base_gram_to_ids.setdefault(gram, set()).add(token_id)
            first_char = token[:1]
            if first_char:
                self.base_first_char_to_ids.setdefault(first_char, set()).add(token_id)

    def _budget_for_text(self, text: str) -> int:
        units = count_mixed_units(text)
        if units <= 0:
            return 0
        return max(1, min(6, math.ceil(units / 3)))

    def _resolve_min_new_token_freq(
        self,
        frequency_items: list[tuple[str, int]],
        min_new_token_freq: int | None,
    ) -> int:
        if min_new_token_freq is not None:
            return max(int(min_new_token_freq), 1)
        return 1

    def _rank_text_candidates(
        self,
        text: str,
        *,
        candidate_plan: dict | None = None,
    ) -> list[str]:
        normalized = normalize_core_text(text)
        if not normalized:
            return []
        budget = self._budget_for_text(normalized)
        if budget <= 0:
            return []
        return [
            candidate
            for candidate in self._choose_candidates(
                normalized,
                budget=budget,
                for_stage1=False,
                prepared_plan=candidate_plan,
            )
            if count_mixed_units(candidate) <= 8
        ]

    def _find_base_match(self, token: str) -> tuple[int | None, float]:
        normalized = normalize_core_text(token)
        if not normalized:
            return None, 0.0

        cached = self.base_match_cache.get(normalized)
        if cached is not None:
            return cached

        exact_id = self.lexicon.get_token_id(normalized)
        if exact_id is not None and exact_id in self.base_match_token_ids:
            self.base_match_cache[normalized] = (exact_id, 1.0)
            return exact_id, 1.0

        best_token_id = None
        best_score = 0.0
        gram_postings = []
        for gram in _char_ngrams(normalized):
            ids = self.base_gram_to_ids.get(gram)
            if ids:
                gram_postings.append((len(ids), ids))

        candidate_ids = set()
        for _, ids in sorted(gram_postings, key=lambda item: item[0])[:3]:
            candidate_ids.update(ids)
        if not candidate_ids:
            candidate_ids = set(self.base_first_char_to_ids.get(normalized[:1]) or ())
        if not candidate_ids:
            candidate_ids = set(self.base_match_token_ids)

        normalized_units = count_mixed_units(normalized)
        normalized_compact = _compact_normalized_text(normalized)
        normalized_grams = _char_ngrams_from_compact(normalized_compact)
        candidate_ids = {
            token_id
            for token_id in candidate_ids
            if abs(self.lexicon.token_units.get(token_id, 0) - normalized_units) <= 4
        } or candidate_ids

        for token_id in candidate_ids:
            existing = self.lexicon.id_to_token.get(token_id)
            if not existing:
                continue
            score = _surface_overlap_score_with_features(
                normalized,
                existing,
                compact_left=normalized_compact,
                compact_right=self.lexicon.token_compact_texts.get(token_id),
                left_grams=normalized_grams,
                right_grams=self.lexicon.token_char_grams.get(token_id),
            )
            if score > best_score:
                best_token_id = token_id
                best_score = score

        self.base_match_cache[normalized] = (best_token_id, best_score)
        return best_token_id, best_score

    def _materialize_text_candidate(
        self,
        candidate: str,
        *,
        allow_new_tokens: bool,
        candidate_total_freq: int,
        min_new_token_freq: int,
    ) -> tuple[int | None, str]:
        existing_id = self.lexicon.get_token_id(candidate)
        if existing_id is not None:
            return self.lexicon.touch_token(existing_id), "exact"

        best_token_id, best_score = self._find_base_match(candidate)
        if best_token_id is not None and best_score >= self.reuse_threshold:
            return self.lexicon.touch_token(best_token_id), "reuse"

        if (
            best_token_id is not None
            and round(1.0 - best_score, 4) < self.novelty_threshold
        ):
            return self.lexicon.touch_token(best_token_id), "fallback_reuse"

        if not allow_new_tokens:
            return None, "skipped"
        if candidate_total_freq < min_new_token_freq:
            return None, "blocked_new"
        return self.lexicon.add_token(candidate, source=self.source), "new"

    def encode(
        self,
        text: str,
        *,
        allow_new_tokens: bool = False,
        candidate_plan: dict | None = None,
    ) -> list[int]:
        normalized = normalize_core_text(text)
        if not normalized:
            return []
        budget = self._budget_for_text(normalized)
        token_ids = []
        for candidate in self._choose_candidates(
            normalized,
            budget=budget,
            for_stage1=False,
            prepared_plan=candidate_plan,
        ):
            if count_mixed_units(candidate) > 8:
                continue
            token_id = self._materialize_token(
                candidate,
                allow_new_tokens=allow_new_tokens,
                source_text=normalized,
            )
            if token_id is not None and token_id not in token_ids:
                token_ids.append(token_id)
            if len(token_ids) >= budget:
                break
        return token_ids

    def fit(
        self,
        texts: list[str] | None = None,
        epochs: int = 1,
        text_counter: Counter | None = None,
        frequency_items: list[tuple[str, int]] | None = None,
        candidate_plans: dict[str, dict] | None = None,
        min_new_token_freq: int | None = None,
    ) -> list[list[int]]:
        encoded = []
        fit_started = time.perf_counter()
        if frequency_items is None:
            text_counter = text_counter or Counter(
                text for text in (texts or []) if normalize_core_text(text)
            )
            frequency_items = text_counter.most_common()
        resolved_min_new_token_freq = self._resolve_min_new_token_freq(
            frequency_items,
            min_new_token_freq,
        )

        rank_seconds = 0.0
        materialize_seconds = 0.0
        emit_seconds = 0.0
        unique_candidate_count = 0
        active_text_count = 0
        decision_counts = Counter()
        materialize_cache_hits = 0
        materialize_cache_misses = 0

        if resolved_min_new_token_freq <= 1:
            stage_started = time.perf_counter()
            unique_candidates = set()
            ranked_text_plans = []
            for text, freq in frequency_items:
                normalized_text = normalize_core_text(text)
                ranked_candidates = self._rank_text_candidates(
                    normalized_text,
                    candidate_plan=(candidate_plans or {}).get(normalized_text),
                )
                if not ranked_candidates:
                    continue
                active_text_count += 1
                ranked_text_plans.append(
                    (normalized_text, int(freq), ranked_candidates)
                )
                unique_candidates.update(ranked_candidates)
            rank_seconds += time.perf_counter() - stage_started
            unique_candidate_count = len(unique_candidates)

            for _ in range(max(epochs, 1)):
                materialize_cache: dict[str, int | None] = {}
                for text, freq, ranked_candidates in ranked_text_plans:
                    materialize_started = time.perf_counter()
                    token_ids = []
                    for candidate in ranked_candidates:
                        if candidate in materialize_cache:
                            materialize_cache_hits += 1
                            token_id = materialize_cache[candidate]
                            if token_id is not None:
                                self.lexicon.touch_token(token_id)
                        else:
                            materialize_cache_misses += 1
                            token_id = self._materialize_token(
                                candidate,
                                allow_new_tokens=True,
                                source_text=text,
                            )
                            materialize_cache[candidate] = token_id
                        if token_id is not None and token_id not in token_ids:
                            token_ids.append(token_id)
                    materialize_seconds += time.perf_counter() - materialize_started
                    emit_started = time.perf_counter()
                    if token_ids:
                        encoded.extend([token_ids] * int(freq))
                        extra_repeats = max(int(freq) - 1, 0)
                        for token_id in token_ids:
                            self.lexicon.touch_token(token_id, delta=extra_repeats)
                    emit_seconds += time.perf_counter() - emit_started
        else:
            for _ in range(max(epochs, 1)):
                stage_started = time.perf_counter()
                candidate_counter = Counter()
                ranked_text_plans = []
                for text, freq in frequency_items:
                    ranked_candidates = self._rank_text_candidates(
                        text,
                        candidate_plan=(candidate_plans or {}).get(
                            normalize_core_text(text)
                        ),
                    )
                    if not ranked_candidates:
                        continue
                    ranked_text_plans.append((ranked_candidates, int(freq)))
                    for candidate in ranked_candidates:
                        candidate_counter[candidate] += int(freq)
                rank_seconds += time.perf_counter() - stage_started

                active_text_count = len(ranked_text_plans)
                unique_candidate_count = max(
                    unique_candidate_count, len(candidate_counter)
                )

                for ranked_candidates, freq in ranked_text_plans:
                    materialize_started = time.perf_counter()
                    token_ids = []
                    for candidate in ranked_candidates:
                        token_id, decision = self._materialize_text_candidate(
                            candidate,
                            allow_new_tokens=True,
                            candidate_total_freq=int(
                                candidate_counter.get(candidate, 1)
                            ),
                            min_new_token_freq=resolved_min_new_token_freq,
                        )
                        decision_counts[decision] += 1
                        if token_id is not None and token_id not in token_ids:
                            token_ids.append(token_id)
                    materialize_seconds += time.perf_counter() - materialize_started

                    emit_started = time.perf_counter()
                    if token_ids:
                        encoded.extend([token_ids] * int(freq))
                        extra_repeats = max(int(freq) - 1, 0)
                        for token_id in token_ids:
                            self.lexicon.touch_token(token_id, delta=extra_repeats)
                    emit_seconds += time.perf_counter() - emit_started

        self.last_fit_stats = {
            "stage2_rank_seconds": round(rank_seconds, 4),
            "stage2_materialize_seconds": round(materialize_seconds, 4),
            "stage2_emit_seconds": round(emit_seconds, 4),
            "stage2_unique_text_count": len(frequency_items),
            "stage2_active_text_count": active_text_count,
            "stage2_unique_candidate_count": unique_candidate_count,
            "stage2_min_new_token_freq": resolved_min_new_token_freq,
            "stage2_exact_candidate_count": int(decision_counts.get("exact", 0)),
            "stage2_reuse_candidate_count": int(decision_counts.get("reuse", 0)),
            "stage2_fallback_reuse_count": int(
                decision_counts.get("fallback_reuse", 0)
            ),
            "stage2_new_candidate_count": int(decision_counts.get("new", 0)),
            "stage2_blocked_new_candidate_count": int(
                decision_counts.get("blocked_new", 0)
            ),
            "stage2_skipped_candidate_count": int(decision_counts.get("skipped", 0)),
            "stage2_materialize_cache_hits": int(materialize_cache_hits),
            "stage2_materialize_cache_misses": int(materialize_cache_misses),
            "stage2_fit_seconds": round(time.perf_counter() - fit_started, 4),
        }
        return encoded


@dataclass
class CoreImpEvaluator:
    tag_doc_count: int = 0
    text_doc_count: int = 0
    tag_df: Counter = field(default_factory=Counter)
    text_df: Counter = field(default_factory=Counter)

    def fit(
        self,
        tag_token_sequences: list[list[int]],
        text_token_sequences: list[list[int]],
    ) -> "CoreImpEvaluator":
        self.tag_doc_count = len(tag_token_sequences)
        self.text_doc_count = len(text_token_sequences)
        for token_ids in tag_token_sequences:
            for token_id in set(token_ids):
                self.tag_df[token_id] += 1
        for token_ids in text_token_sequences:
            for token_id in set(token_ids):
                self.text_df[token_id] += 1
        return self

    def score_token(self, token_id: int) -> float:
        total_docs = self.tag_doc_count + self.text_doc_count
        total_df = self.tag_df.get(token_id, 0) + self.text_df.get(token_id, 0)
        if total_docs <= 0 or total_df <= 0:
            return 0.0

        rarity = math.log((total_docs + 1) / (total_df + 1)) + 1.0
        tag_strength = self.tag_df.get(token_id, 0) / max(self.tag_doc_count, 1)
        text_strength = self.text_df.get(token_id, 0) / max(self.text_doc_count, 1)
        source_bias = 1.25 if tag_strength >= text_strength else 0.95
        return round(rarity * source_bias, 4)

    def score_sequence(self, token_ids: list[int]) -> list[tuple[int, float]]:
        raw = [(token_id, self.score_token(token_id)) for token_id in token_ids]
        total = sum(score for _, score in raw) or 1.0
        return [(token_id, round(score / total, 4)) for token_id, score in raw]

    def to_dict(self) -> dict:
        return {
            "tag_doc_count": int(self.tag_doc_count),
            "text_doc_count": int(self.text_doc_count),
            "tag_df": {
                str(token_id): int(freq) for token_id, freq in self.tag_df.items()
            },
            "text_df": {
                str(token_id): int(freq) for token_id, freq in self.text_df.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: dict | None) -> "CoreImpEvaluator":
        payload = payload or {}
        return cls(
            tag_doc_count=int(payload.get("tag_doc_count") or 0),
            text_doc_count=int(payload.get("text_doc_count") or 0),
            tag_df=Counter(
                {
                    int(token_id): int(freq)
                    for token_id, freq in (payload.get("tag_df") or {}).items()
                }
            ),
            text_df=Counter(
                {
                    int(token_id): int(freq)
                    for token_id, freq in (payload.get("text_df") or {}).items()
                }
            ),
        )
