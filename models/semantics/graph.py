from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from enum import IntFlag


class NodeKind(str, Enum):
    VOCAB = "vocab"
    TAG = "tag"
    OWNER = "owner"


class RelKind(str, Enum):
    DOC_COOCCURRENCE = "doc_cooccurrence"
    OWNER_TAG = "owner_tag"
    NEGATIVE_SAMPLE = "negative_sample"


class TermRole(IntFlag):
    TITLE = 1
    TAG = 2
    OWNER = 4


@dataclass(frozen=True, slots=True)
class TermNode:
    surface: str
    kind: NodeKind


@dataclass(frozen=True, slots=True)
class EdgeKey:
    source: str
    target: str
    relation: RelKind


@dataclass(frozen=True, slots=True)
class TermRecord:
    surface: str
    roles: int
    score: float

    @property
    def is_title(self) -> bool:
        return bool(self.roles & TermRole.TITLE)

    @property
    def is_tag(self) -> bool:
        return bool(self.roles & TermRole.TAG)

    @property
    def is_owner(self) -> bool:
        return bool(self.roles & TermRole.OWNER)


@dataclass(frozen=True, slots=True)
class ExtractedDoc:
    doc_key: str
    content_hash: str
    terms: tuple[TermRecord, ...]
    owner_terms: tuple[str, ...]

    @property
    def semantic_terms(self) -> tuple[str, ...]:
        return tuple(term.surface for term in self.terms)
