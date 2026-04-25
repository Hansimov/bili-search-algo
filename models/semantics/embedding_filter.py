from __future__ import annotations

import json
import os
import threading
import unicodedata
import warnings

from dataclasses import dataclass
from typing import Callable


SimilarityFn = Callable[[str, list[str]], list[float]]


def _import_tei_clients():
    warnings.filterwarnings("ignore", message='Field "model_.*protected namespace')
    from tfmx import TEIClients

    return TEIClients


@dataclass(slots=True)
class EmbeddingFilterStats:
    enabled: bool
    sources_seen: int = 0
    sources_filtered: int = 0
    targets_seen: int = 0
    targets_kept: int = 0
    targets_removed: int = 0
    skipped_reason: str = ""

    def to_dict(self) -> dict[str, int | bool | str]:
        return {
            "enabled": self.enabled,
            "sources_seen": self.sources_seen,
            "sources_filtered": self.sources_filtered,
            "targets_seen": self.targets_seen,
            "targets_kept": self.targets_kept,
            "targets_removed": self.targets_removed,
            "skipped_reason": self.skipped_reason,
        }


def parse_endpoint_values(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if value is None:
        value = os.getenv("SEMANTICS_TEI_ENDPOINTS") or os.getenv("TEI_CLIENTS_ENDPOINTS")
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            payload = json.loads(text)
            if isinstance(payload, list):
                return [str(item).strip() for item in payload if str(item).strip()]
        except json.JSONDecodeError:
            pass
    return [part.strip() for part in text.split(",") if part.strip()]


def has_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in str(text or ""))


def has_ascii_letter(text: str) -> bool:
    return any(char.isascii() and char.isalpha() for char in str(text or ""))


def normalized_text(text: str) -> str:
    return unicodedata.normalize("NFKC", str(text or "")).casefold().strip()


def min_similarity_for_pair(
    source: str,
    target: str,
    *,
    min_score: float,
    cjk_min_score: float,
    mixed_script_min_score: float,
) -> float:
    if has_cjk(source) and has_cjk(target):
        return max(min_score, cjk_min_score)
    if has_cjk(source) != has_cjk(target) and (
        has_ascii_letter(source) or has_ascii_letter(target)
    ):
        return max(min_score, mixed_script_min_score)
    return min_score


def filter_mapping_by_similarity(
    mapping: dict[str, dict[str, float]],
    similarity_fn: SimilarityFn,
    *,
    min_score: float = 0.52,
    cjk_min_score: float = 0.58,
    mixed_script_min_score: float = 0.62,
    max_sources: int = 0,
    max_targets_per_source: int = 0,
    reweight: bool = True,
) -> tuple[dict[str, dict[str, float]], EmbeddingFilterStats]:
    stats = EmbeddingFilterStats(enabled=True)
    filtered: dict[str, dict[str, float]] = {}

    ranked_sources = sorted(
        mapping,
        key=lambda source: (
            -max((float(weight) for weight in mapping[source].values()), default=0.0),
            -len(mapping[source]),
            normalized_text(source),
        ),
    )
    for source_index, source in enumerate(ranked_sources):
        if max_sources > 0 and source_index >= max_sources:
            filtered[source] = dict(mapping[source])
            continue
        targets = sorted(mapping[source].items(), key=lambda item: (-item[1], item[0]))
        if max_targets_per_source > 0:
            targets = targets[:max_targets_per_source]
        if not targets:
            continue

        stats.sources_seen += 1
        target_texts = [target for target, _weight in targets]
        scores = similarity_fn(source, target_texts)
        if len(scores) != len(target_texts):
            stats.skipped_reason = "similarity_score_count_mismatch"
            filtered[source] = dict(mapping[source])
            continue

        kept: dict[str, float] = {}
        for (target, weight), score in zip(targets, scores):
            stats.targets_seen += 1
            threshold = min_similarity_for_pair(
                source,
                target,
                min_score=min_score,
                cjk_min_score=cjk_min_score,
                mixed_script_min_score=mixed_script_min_score,
            )
            if score < threshold:
                stats.targets_removed += 1
                continue
            next_weight = float(weight)
            if reweight:
                next_weight = float(weight) * (0.65 + 0.35 * min(max(score, 0.0), 1.0))
            kept[target] = round(max(0.0, min(0.98, next_weight)), 4)
            stats.targets_kept += 1

        if kept:
            filtered[source] = kept
            stats.sources_filtered += 1

    return filtered, stats


class TeiSimilarityScorer:
    def __init__(
        self,
        endpoints: str | list[str] | None = None,
        *,
        init_timeout: float = 15.0,
    ):
        self.endpoints = parse_endpoint_values(endpoints)
        self.init_timeout = init_timeout
        self._clients = None
        self._initialized = False

    def is_available(self) -> bool:
        return bool(self.endpoints) and self._ensure_initialized()

    def _ensure_initialized(self) -> bool:
        if self._initialized:
            return self._clients is not None
        if not self.endpoints:
            self._initialized = True
            return False

        holder: dict[str, object] = {}

        def _init():
            try:
                TEIClients = _import_tei_clients()
                holder["clients"] = TEIClients(endpoints=self.endpoints)
            except Exception as exc:
                holder["error"] = exc

        thread = threading.Thread(target=_init, daemon=True)
        thread.start()
        thread.join(timeout=self.init_timeout)
        self._initialized = True
        if thread.is_alive() or holder.get("error") is not None:
            return False
        self._clients = holder.get("clients")
        return self._clients is not None

    def similarities(self, source: str, targets: list[str]) -> list[float]:
        if not targets:
            return []
        if not self._ensure_initialized() or self._clients is None:
            return [0.0 for _target in targets]
        rankings = self._clients.rerank([source], targets)
        scores: list[float] = []
        for ranking in rankings or []:
            try:
                _rank, score = ranking
                scores.append(float(score or 0.0))
            except (TypeError, ValueError):
                scores.append(0.0)
        return scores


class TeiEmbeddingSimilarityScorer:
    def __init__(
        self,
        endpoints: str | list[str] | None = None,
        *,
        init_timeout: float = 15.0,
        batch_size: int = 512,
    ):
        self.endpoints = parse_endpoint_values(endpoints)
        self.init_timeout = init_timeout
        self.batch_size = batch_size
        self._clients = None
        self._initialized = False
        self._vectors: dict[str, object] = {}

    def is_available(self) -> bool:
        return bool(self.endpoints) and self._ensure_initialized()

    def _ensure_initialized(self) -> bool:
        if self._initialized:
            return self._clients is not None
        if not self.endpoints:
            self._initialized = True
            return False

        holder: dict[str, object] = {}

        def _init():
            try:
                TEIClients = _import_tei_clients()
                holder["clients"] = TEIClients(endpoints=self.endpoints)
            except Exception as exc:
                holder["error"] = exc

        thread = threading.Thread(target=_init, daemon=True)
        thread.start()
        thread.join(timeout=self.init_timeout)
        self._initialized = True
        if thread.is_alive() or holder.get("error") is not None:
            return False
        self._clients = holder.get("clients")
        return self._clients is not None

    def preload_terms(self, terms: list[str] | set[str] | tuple[str, ...]) -> None:
        if not self._ensure_initialized() or self._clients is None:
            return
        missing = sorted({normalized_text(term): str(term) for term in terms}.values())
        missing = [term for term in missing if term and term not in self._vectors]
        if not missing:
            return
        import numpy as np

        for offset in range(0, len(missing), self.batch_size):
            chunk = missing[offset : offset + self.batch_size]
            vectors = self._clients.embed(chunk, normalize=True, truncate=True)
            for term, vector in zip(chunk, vectors):
                self._vectors[term] = np.asarray(vector, dtype=np.float32)

    def similarities(self, source: str, targets: list[str]) -> list[float]:
        if not targets:
            return []
        self.preload_terms([source, *targets])
        source_vector = self._vectors.get(source)
        if source_vector is None:
            return [0.0 for _target in targets]
        scores: list[float] = []
        for target in targets:
            target_vector = self._vectors.get(target)
            if target_vector is None:
                scores.append(0.0)
                continue
            try:
                scores.append(float(source_vector.dot(target_vector)))
            except AttributeError:
                scores.append(
                    float(sum(a * b for a, b in zip(source_vector, target_vector)))
                )
        return scores


def hash_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    try:
        left_bytes = bytes.fromhex(left)
        right_bytes = bytes.fromhex(right)
    except ValueError:
        return 0.0
    if not left_bytes or not right_bytes:
        return 0.0
    dist = sum((a ^ b).bit_count() for a, b in zip(left_bytes, right_bytes))
    bit_count = min(len(left_bytes), len(right_bytes)) * 8
    if bit_count <= 0:
        return 0.0
    return 1.0 - (dist / bit_count)


class TeiLshSimilarityScorer:
    def __init__(
        self,
        endpoints: str | list[str] | None = None,
        *,
        init_timeout: float = 15.0,
        bitn: int = 2048,
    ):
        self.endpoints = parse_endpoint_values(endpoints)
        self.init_timeout = init_timeout
        self.bitn = bitn
        self._clients = None
        self._initialized = False
        self._hashes: dict[str, str] = {}

    def is_available(self) -> bool:
        return bool(self.endpoints) and self._ensure_initialized()

    def _ensure_initialized(self) -> bool:
        if self._initialized:
            return self._clients is not None
        if not self.endpoints:
            self._initialized = True
            return False

        holder: dict[str, object] = {}

        def _init():
            try:
                TEIClients = _import_tei_clients()
                holder["clients"] = TEIClients(endpoints=self.endpoints)
            except Exception as exc:
                holder["error"] = exc

        thread = threading.Thread(target=_init, daemon=True)
        thread.start()
        thread.join(timeout=self.init_timeout)
        self._initialized = True
        if thread.is_alive() or holder.get("error") is not None:
            return False
        self._clients = holder.get("clients")
        return self._clients is not None

    def preload_terms(self, terms: list[str] | set[str] | tuple[str, ...]) -> None:
        if not self._ensure_initialized() or self._clients is None:
            return
        missing = sorted(
            {
                str(term)
                for term in terms
                if str(term) and str(term) not in self._hashes
            }
        )
        if not missing:
            return
        hashes = self._clients.lsh(
            missing,
            bitn=self.bitn,
            normalize=True,
            truncate=True,
        )
        for term, hash_value in zip(missing, hashes):
            self._hashes[term] = str(hash_value or "")

    def similarities(self, source: str, targets: list[str]) -> list[float]:
        if not targets:
            return []
        self.preload_terms([source, *targets])
        source_hash = self._hashes.get(source, "")
        return [
            hash_similarity(source_hash, self._hashes.get(target, ""))
            for target in targets
        ]


__all__ = [
    "EmbeddingFilterStats",
    "TeiEmbeddingSimilarityScorer",
    "TeiLshSimilarityScorer",
    "TeiSimilarityScorer",
    "filter_mapping_by_similarity",
    "hash_similarity",
    "min_similarity_for_pair",
    "parse_endpoint_values",
]
