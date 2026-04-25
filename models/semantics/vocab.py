from __future__ import annotations

import re

from pathlib import Path

from models.sentencepiece.vocab_filters import is_malformed_token

try:
    import ahocorasick
except ImportError:  # pragma: no cover - fallback for minimal environments
    ahocorasick = None


DEFAULT_VOCAB_PATH = (
    Path(__file__).resolve().parents[1] / "sentencepiece" / "vocabs.txt"
)
WORD_SPLIT_RE = re.compile(r"[\s,，。！？!?、|/:：()（）\[\]【】《》<>\"'`~#]+")
ASCII_RE = re.compile(r"^[a-z0-9._+#/\-]+$")
CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def collapse_text(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


def normalize_surface(value: object) -> str:
    text = collapse_text(value).replace("▂", " ").strip()
    return " ".join(text.lower().split())


def cjk_char_count(text: str) -> int:
    return sum(1 for char in text if CJK_RE.match(char))


class BaseVocab:
    def __init__(
        self,
        path: Path | str = DEFAULT_VOCAB_PATH,
        *,
        max_term_length: int = 24,
        max_cjk_chars: int = 12,
        max_vocab_size: int = 800000,
    ):
        self.path = Path(path)
        self.max_term_length = max_term_length
        self.max_cjk_chars = max_cjk_chars
        self.max_vocab_size = max(0, max_vocab_size)
        self.terms = self._load_terms(self.path)
        self.compact_terms = {term.replace(" ", ""): term for term in self.terms}
        self.compact_lengths = {term: len(term.replace(" ", "")) for term in self.terms}
        self.max_scan_length = min(
            max((len(term.replace(" ", "")) for term in self.terms), default=1),
            self.max_term_length,
        )
        self.matcher = self._build_matcher()

    def _load_terms(self, path: Path) -> frozenset[str]:
        if not path.exists():
            return frozenset()
        terms: set[str] = set()
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                raw_term = line.split("\t", 1)[0].strip()
                term = normalize_surface(raw_term)
                if self.accepts(term):
                    terms.add(term)
                    if self.max_vocab_size and len(terms) >= self.max_vocab_size:
                        break
        return frozenset(terms)

    def _build_matcher(self):
        if ahocorasick is None or not self.compact_terms:
            return None
        matcher = ahocorasick.Automaton()
        for compact, term in self.compact_terms.items():
            matcher.add_word(compact, term)
        matcher.make_automaton()
        return matcher

    def accepts(self, term: str) -> bool:
        normalized = normalize_surface(term)
        if not normalized:
            return False
        if len(normalized) > self.max_term_length:
            return False
        if cjk_char_count(normalized) > self.max_cjk_chars:
            return False
        if normalized.isdigit():
            return False
        if is_malformed_token(normalized):
            return False
        if ASCII_RE.fullmatch(normalized) and len(normalized) <= 1:
            return False
        return True

    def contains(self, term: str) -> bool:
        return normalize_surface(term) in self.terms

    def normalize_if_valid(self, term: object, *, allow_oov: bool = False) -> str:
        normalized = normalize_surface(term)
        if not self.accepts(normalized):
            return ""
        if allow_oov or not self.terms or normalized in self.terms:
            return normalized
        return ""

    def split_terms(self, text: object, *, allow_oov: bool = False) -> list[str]:
        terms: list[str] = []
        for raw_part in WORD_SPLIT_RE.split(str(text or "")):
            term = self.normalize_if_valid(raw_part, allow_oov=allow_oov)
            if term:
                terms.append(term)
        return terms

    def iter_vocab_matches(self, text: object) -> list[str]:
        normalized = normalize_surface(text).replace(" ", "")
        if not normalized or not self.terms:
            return []
        if self.matcher is not None:
            return self._iter_automaton_matches(normalized)
        matches: list[str] = []
        seen: set[str] = set()
        text_length = len(normalized)
        for start in range(text_length):
            max_end = min(text_length, start + self.max_scan_length)
            for end in range(max_end, start + 1, -1):
                candidate = normalized[start:end]
                matched = self.compact_terms.get(candidate)
                if matched and matched not in seen:
                    seen.add(matched)
                    matches.append(matched)
                    break
        return matches

    def _iter_automaton_matches(self, compact_text: str) -> list[str]:
        best_by_start: dict[int, str] = {}
        for end_index, term in self.matcher.iter(compact_text):
            term_length = self.compact_lengths.get(term, len(term))
            start_index = end_index - term_length + 1
            current = best_by_start.get(start_index)
            if current is None or term_length > self.compact_lengths.get(
                current, len(current)
            ):
                best_by_start[start_index] = term
        matches: list[str] = []
        seen: set[str] = set()
        for start_index in sorted(best_by_start):
            term = best_by_start[start_index]
            if term not in seen:
                seen.add(term)
                matches.append(term)
        return matches
