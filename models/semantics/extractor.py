from __future__ import annotations

from dataclasses import dataclass

from models.semantics.graph import ExtractedDoc, TermRecord, TermRole
from models.semantics.preprocess import (
    content_hash_of,
    extract_owner_terms,
    extract_tag_terms,
    extract_title_terms,
    stable_doc_key,
)
from models.semantics.vocab import BaseVocab


@dataclass(slots=True)
class ExtractionConfig:
    title_term_limit: int = 8
    tag_term_limit: int = 8
    owner_term_limit: int = 1
    max_terms_per_doc: int = 12
    max_edges_per_doc: int = 48
    negative_samples_per_doc: int = 4


class DocExtractor:
    def __init__(self, vocab: BaseVocab, config: ExtractionConfig | None = None):
        self.vocab = vocab
        self.config = config or ExtractionConfig()

    def extract(self, doc: dict) -> ExtractedDoc | None:
        doc_key = stable_doc_key(doc)
        if not doc_key or doc_key in {"aid:", "bvid:"}:
            return None
        title_terms = extract_title_terms(self.vocab, doc, self.config.title_term_limit)
        tag_terms = extract_tag_terms(self.vocab, doc, self.config.tag_term_limit)
        owner_terms = extract_owner_terms(self.vocab, doc, self.config.owner_term_limit)
        if not title_terms and not tag_terms:
            return None
        terms = self._build_term_records(title_terms, tag_terms)
        if not terms:
            return None
        return ExtractedDoc(
            doc_key=doc_key,
            content_hash=content_hash_of(doc),
            terms=terms,
            owner_terms=owner_terms,
        )

    def _build_term_records(
        self, title_terms: tuple[str, ...], tag_terms: tuple[str, ...]
    ) -> tuple[TermRecord, ...]:
        merged: dict[str, TermRecord] = {}

        def add(term: str, role: TermRole, score: float) -> None:
            current = merged.get(term)
            if current is None:
                merged[term] = TermRecord(term, int(role), score)
                return
            merged[term] = TermRecord(
                term,
                current.roles | int(role),
                max(current.score, score),
            )

        for term in tag_terms:
            add(term, TermRole.TAG, 1.0)
        for term in title_terms:
            add(term, TermRole.TITLE, 0.72)

        ranked = sorted(
            merged.values(),
            key=lambda item: (-item.score, -len(item.surface), item.surface),
        )
        return tuple(ranked[: self.config.max_terms_per_doc])
