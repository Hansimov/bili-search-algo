from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from models.semantics.storage import encode_normalized_term


@dataclass(frozen=True, slots=True)
class CandidateScore:
    term: str
    score: float


class FastTextV2CandidateScorer:
    """Lightweight candidate reranker for pre-generated semantic candidates.

    FastText v2 is intentionally not used as a global nearest-neighbor index: on
    broad noisy vocabularies, raw fastText cosine tends to produce hubs. The
    useful path is to score a small candidate list from the semantic graph,
    correction generator, or TEI/LSH retrieval.
    """

    def __init__(self, model, *, center: bool = True) -> None:
        self.model = model
        self.kv = model.wv if hasattr(model, "wv") else model
        self.center = bool(center)
        self._mean = None
        if self.center:
            vectors = np.asarray(self.kv.vectors, dtype=np.float32)
            self._mean = vectors.mean(axis=0)

    @classmethod
    def load(cls, path: Path | str, *, center: bool = True):
        try:
            from gensim.models import FastText, KeyedVectors
        except Exception as exc:
            raise RuntimeError("gensim is required to load fasttext_v2 models") from exc

        path = Path(path)
        if path.suffix == ".kv":
            model = KeyedVectors.load(str(path), mmap="r")
        else:
            model = FastText.load(str(path))
        return cls(model, center=center)

    def vector(self, term: str) -> np.ndarray:
        token = encode_normalized_term(str(term or "").strip())
        vector = np.asarray(self.kv[token], dtype=np.float32)
        if self._mean is not None:
            vector = vector - self._mean
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-8:
            return vector
        return vector / norm

    def similarity(self, source: str, target: str) -> float:
        left = self.vector(source)
        right = self.vector(target)
        return float(np.dot(left, right))

    def rank(self, source: str, candidates: Iterable[str]) -> list[CandidateScore]:
        scores = [
            CandidateScore(str(candidate), self.similarity(source, str(candidate)))
            for candidate in candidates
            if str(candidate).strip()
        ]
        return sorted(scores, key=lambda item: (-item.score, item.term))
