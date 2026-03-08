import math
import re

from collections import Counter
from dataclasses import dataclass, field


LATIN_CHAR_RE = re.compile(r"[a-z0-9]")
LATIN_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_\-\.]*")
CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")
CJK_SPAN_RE = re.compile(r"[\u4e00-\u9fff]+")
CORETEXT_SPLIT_RE = re.compile(r"[，,。.!！?？、;；:：()（）\[\]【】<>《》\-\|/\\\s]+")


def normalize_core_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


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

    def fit(self, texts: list[str], *, for_stage1: bool) -> "CoreCorpusStats":
        self.total_docs = 0
        self.candidate_doc_freqs = Counter()
        self.stop_candidates = set()
        self.stop_coverage_threshold = 0.0

        for text in texts:
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
            self.total_docs += 1
            self.candidate_doc_freqs.update(candidates)

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


def _char_ngrams(text: str) -> set[str]:
    compact = re.sub(r"\s+", "", normalize_core_text(text))
    if len(compact) <= 2:
        return {compact} if compact else set()
    grams = {compact[index : index + 2] for index in range(len(compact) - 1)}
    grams.add(compact)
    return grams


def _surface_overlap_score(left: str, right: str) -> float:
    if left == right:
        return 1.0
    compact_left = re.sub(r"\s+", "", normalize_core_text(left))
    compact_right = re.sub(r"\s+", "", normalize_core_text(right))
    if compact_left and compact_right:
        shorter = min(len(compact_left), len(compact_right))
        longer = max(len(compact_left), len(compact_right))
        containment = 0.0
        if compact_left in compact_right or compact_right in compact_left:
            containment = shorter / max(longer, 1)
    else:
        containment = 0.0
    left_grams = _char_ngrams(left)
    right_grams = _char_ngrams(right)
    if not left_grams or not right_grams:
        return containment
    inter = len(left_grams & right_grams)
    union = len(left_grams | right_grams)
    jaccard = inter / union if union else 0.0
    return max(jaccard, containment)


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
    next_token_id: int = 1
    gram_to_ids: dict[str, set[int]] = field(default_factory=dict)
    match_cache: dict[str, tuple[int | None, float]] = field(default_factory=dict)

    def _index_token(self, token_id: int, token: str):
        for gram in _char_ngrams(token):
            if not gram:
                continue
            self.gram_to_ids.setdefault(gram, set()).add(token_id)

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

    def touch_token(self, token_id: int) -> int:
        self.token_freqs[token_id] += 1
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
            candidate_ids = {
                token_id
                for token_id, existing in self.id_to_token.items()
                if existing.startswith(first_char)
            }
        if not candidate_ids:
            candidate_ids = set(self.id_to_token.keys())

        normalized_units = count_mixed_units(normalized)
        candidate_ids = {
            token_id
            for token_id in candidate_ids
            if abs(
                count_mixed_units(self.id_to_token.get(token_id, "")) - normalized_units
            )
            <= 4
        } or candidate_ids

        for token_id in candidate_ids:
            existing = self.id_to_token.get(token_id)
            if not existing:
                continue
            score = _surface_overlap_score(normalized, existing)
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
        next_token_id = int(
            payload.get("next_token_id") or (max(id_to_token.keys(), default=0) + 1)
        )
        lexicon = cls(
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            token_freqs=token_freqs,
            token_sources=token_sources,
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

    def _score_candidate(self, candidate: str, budget: int) -> tuple[float, int, str]:
        whole_bonus = 4.0 if count_mixed_units(candidate) <= 8 else 0.0
        freq_bonus = 0.0
        existing_id = self.lexicon.get_token_id(candidate)
        if existing_id is not None:
            freq_bonus = min(float(self.lexicon.token_freqs.get(existing_id, 0)), 8.0)
        length_bonus = min(count_mixed_units(candidate), budget * 2)
        return (whole_bonus + freq_bonus + length_bonus, len(candidate), candidate)

    def _choose_candidates(
        self, text: str, *, budget: int, for_stage1: bool
    ) -> list[str]:
        candidates = extract_core_candidates(
            text,
            for_stage1=for_stage1,
            corpus_stats=self.corpus_stats,
        )
        ranked = sorted(
            candidates,
            key=lambda candidate: self._score_candidate(candidate, budget),
            reverse=True,
        )
        return ranked[:budget]

    def _materialize_token(
        self, candidate: str, *, allow_new_tokens: bool
    ) -> int | None:
        existing_id = self.lexicon.get_token_id(candidate)
        if existing_id is not None:
            return self.lexicon.touch_token(existing_id)

        best_token_id, best_score = self.lexicon.find_best_match(candidate)
        if best_token_id is not None and best_score >= self.reuse_threshold:
            return self.lexicon.touch_token(best_token_id)

        if not allow_new_tokens:
            return None

        if self.lexicon.novelty_probability(candidate) < self.novelty_threshold:
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

    def encode(self, tag: str, *, allow_new_tokens: bool = False) -> list[int]:
        if not is_valid_stage1_tag(tag, corpus_stats=self.corpus_stats):
            return []
        budget = suggest_token_budget(tag)
        token_ids = []
        for candidate in self._choose_candidates(tag, budget=budget, for_stage1=True):
            token_id = self._materialize_token(
                candidate, allow_new_tokens=allow_new_tokens
            )
            if token_id is not None and token_id not in token_ids:
                token_ids.append(token_id)
            if len(token_ids) >= budget:
                break
        return token_ids

    def fit(self, tags: list[str], epochs: int = 3) -> list[list[int]]:
        encoded = []
        tag_counter = Counter(tag for tag in tags if normalize_core_text(tag))
        for _ in range(max(epochs, 1)):
            for tag, freq in tag_counter.most_common():
                token_ids = self.encode(tag, allow_new_tokens=True)
                if token_ids:
                    encoded.extend([token_ids] * int(freq))
                    for _ in range(max(int(freq) - 1, 0)):
                        for token_id in token_ids:
                            self.lexicon.touch_token(token_id)
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

    def _budget_for_text(self, text: str) -> int:
        units = count_mixed_units(text)
        if units <= 0:
            return 0
        return max(1, min(6, math.ceil(units / 3)))

    def encode(self, text: str, *, allow_new_tokens: bool = False) -> list[int]:
        normalized = normalize_core_text(text)
        if not normalized:
            return []
        budget = self._budget_for_text(normalized)
        token_ids = []
        for candidate in self._choose_candidates(
            normalized, budget=budget, for_stage1=False
        ):
            if count_mixed_units(candidate) > 8:
                continue
            token_id = self._materialize_token(
                candidate, allow_new_tokens=allow_new_tokens
            )
            if token_id is not None and token_id not in token_ids:
                token_ids.append(token_id)
            if len(token_ids) >= budget:
                break
        return token_ids

    def fit(self, texts: list[str], epochs: int = 1) -> list[list[int]]:
        encoded = []
        text_counter = Counter(text for text in texts if normalize_core_text(text))
        for _ in range(max(epochs, 1)):
            for text, freq in text_counter.most_common():
                token_ids = self.encode(text, allow_new_tokens=True)
                if token_ids:
                    encoded.extend([token_ids] * int(freq))
                    for _ in range(max(int(freq) - 1, 0)):
                        for token_id in token_ids:
                            self.lexicon.touch_token(token_id)
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
