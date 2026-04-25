from __future__ import annotations

import json
import math
import sqlite3
import time
import unicodedata

from collections import defaultdict
from collections.abc import Iterable
from heapq import heappush, heapreplace
from itertools import combinations
from pathlib import Path

from models.semantics.graph import ExtractedDoc, TermRecord, TermRole
from models.semantics.embedding_filter import (
    TeiEmbeddingSimilarityScorer,
    TeiLshSimilarityScorer,
    TeiSimilarityScorer,
    filter_mapping_by_similarity,
    min_similarity_for_pair,
)
from models.semantics.vocab import collapse_text, normalize_surface


TSV_SPACE_MASK = "▂"
TITLE_ROLE = int(TermRole.TITLE)
TAG_ROLE = int(TermRole.TAG)
OWNER_ROLE = int(TermRole.OWNER)
MERGED_OUTPUT_FILES = {
    ".merge.sqlite",
    "nodes.tsv",
    "negative_samples.tsv",
    "rewrite.tsv",
    "synonym.tsv",
    "near_synonym.tsv",
    "doc_cooccurrence.tsv",
    "meta.json",
    "edges.tsv",
}


def group_dir(version_root: Path, group_id: int) -> Path:
    return version_root / f"group_{group_id:02d}"


def encode_term(value: object) -> str:
    return normalize_surface(value).replace(" ", TSV_SPACE_MASK)


def encode_normalized_term(value: str) -> str:
    return value.replace(" ", TSV_SPACE_MASK)


def decode_term(value: object) -> str:
    return normalize_surface(str(value or "").replace(TSV_SPACE_MASK, " "))


def decode_normalized_term(value: str) -> str:
    return value.replace(TSV_SPACE_MASK, " ")


def encode_raw_cell(value: object) -> str:
    return collapse_text(value).replace("\t", " ").replace(" ", TSV_SPACE_MASK)


def decode_raw_cell(value: object) -> str:
    return collapse_text(str(value or "").replace(TSV_SPACE_MASK, " "))


def format_weight(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".")


def write_doc_term_segment(
    version_root: Path,
    group_id: int,
    docs: Iterable[ExtractedDoc],
) -> dict[str, int]:
    target_dir = group_dir(version_root, group_id) / "segments"
    target_dir.mkdir(parents=True, exist_ok=True)
    stamp = f"{int(time.time() * 1000)}"
    docs_path = target_dir / f"docs.seg.{stamp}.tsv"
    doc_rows = 0
    term_rows = 0

    with docs_path.open("w", encoding="utf-8") as handle:
        for doc in docs:
            if not doc.terms:
                continue
            cells = [encode_raw_cell(doc.doc_key), doc.content_hash]
            for term in doc.terms:
                cells.extend(
                    [
                        encode_term(term.surface),
                        str(int(term.roles)),
                        format_weight(term.score),
                    ]
                )
                term_rows += 1
            handle.write("\t".join(cells) + "\n")
            doc_rows += 1

    return {"doc_rows": doc_rows, "doc_terms": term_rows}


DocTermRecord = tuple[str, int, float]
TermProfile = tuple[int, int, int]


def parse_doc_identity_row(line: str) -> tuple[str, str] | None:
    first_tab = line.find("\t")
    if first_tab < 0:
        return None
    second_tab = line.find("\t", first_tab + 1)
    if second_tab < 0:
        return None
    return (
        decode_raw_cell(line[:first_tab]),
        line[first_tab + 1 : second_tab],
    )


def parse_doc_term_row(line: str) -> tuple[str, str, tuple[DocTermRecord, ...]] | None:
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 5:
        return None
    doc_key = decode_raw_cell(parts[0])
    content_hash = parts[1]
    records: list[DocTermRecord] = []
    for index in range(2, len(parts) - 2, 3):
        surface = decode_normalized_term(parts[index])
        if not surface:
            continue
        try:
            roles = int(parts[index + 1])
            score = float(parts[index + 2])
        except ValueError:
            continue
        records.append((surface, roles, score))
    return doc_key, content_hash, tuple(records)


def _clean_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in MERGED_OUTPUT_FILES:
        path = output_dir / name
        if path.exists() and path.is_file():
            path.unlink()


def _connect_merge_db(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(str(path))
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-200000")
    connection.execute(
        "CREATE TABLE term_stats("
        "surface TEXT PRIMARY KEY, "
        "df INTEGER NOT NULL, "
        "title_df INTEGER NOT NULL, "
        "tag_df INTEGER NOT NULL, "
        "owner_df INTEGER NOT NULL)"
    )
    connection.execute(
        "CREATE TABLE current_docs("
        "doc_key TEXT PRIMARY KEY, "
        "content_hash TEXT NOT NULL, "
        "segment_ord INTEGER NOT NULL)"
    )
    return connection


def _iter_doc_segment_paths(version_root: Path):
    yield from sorted(version_root.glob("group_*/segments/docs.seg.*.tsv"))


def _collect_current_docs(
    version_root: Path, connection: sqlite3.Connection
) -> dict[str, int]:
    stats = {"segment_rows_seen": 0, "current_docs": 0}
    batch: list[tuple[str, str, int]] = []
    segment_ord = 0
    for path in _iter_doc_segment_paths(version_root):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                parsed = parse_doc_identity_row(line)
                if parsed is None:
                    continue
                doc_key, content_hash = parsed
                batch.append((doc_key, content_hash, segment_ord))
                segment_ord += 1
                stats["segment_rows_seen"] += 1
                if len(batch) >= 100000:
                    _flush_current_docs(connection, batch)
                    batch.clear()
    _flush_current_docs(connection, batch)
    row = connection.execute("SELECT COUNT(*) FROM current_docs").fetchone()
    stats["current_docs"] = int(row[0] if row else 0)
    return stats


def _flush_current_docs(
    connection: sqlite3.Connection, rows: list[tuple[str, str, int]]
) -> None:
    if not rows:
        return
    connection.executemany(
        "INSERT INTO current_docs(doc_key, content_hash, segment_ord) VALUES (?, ?, ?) "
        "ON CONFLICT(doc_key) DO UPDATE SET "
        "content_hash = excluded.content_hash, "
        "segment_ord = excluded.segment_ord "
        "WHERE excluded.segment_ord > current_docs.segment_ord",
        rows,
    )
    connection.commit()


def _is_current_doc(
    connection: sqlite3.Connection, doc_key: str, content_hash: str
) -> bool:
    row = connection.execute(
        "SELECT content_hash FROM current_docs WHERE doc_key = ?",
        (doc_key,),
    ).fetchone()
    return bool(row and row[0] == content_hash)


def _flush_term_stats(
    connection: sqlite3.Connection, rows: dict[str, list[int]]
) -> None:
    if not rows:
        return
    connection.executemany(
        "INSERT INTO term_stats(surface, df, title_df, tag_df, owner_df) VALUES (?, ?, ?, ?, ?) "
        "ON CONFLICT(surface) DO UPDATE SET "
        "df = df + excluded.df, "
        "title_df = title_df + excluded.title_df, "
        "tag_df = tag_df + excluded.tag_df, "
        "owner_df = owner_df + excluded.owner_df",
        [
            (surface, counts[0], counts[1], counts[2], counts[3])
            for surface, counts in rows.items()
        ],
    )
    connection.commit()


def _collect_term_stats(
    version_root: Path,
    connection: sqlite3.Connection,
    *,
    filter_current_docs: bool,
) -> dict[str, int]:
    stats = {"segment_files": 0, "doc_rows": 0, "term_rows": 0}
    for path in _iter_doc_segment_paths(version_root):
        local: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                parsed = parse_doc_term_row(line)
                if parsed is None:
                    continue
                doc_key, content_hash, records = parsed
                if filter_current_docs and not _is_current_doc(
                    connection, doc_key, content_hash
                ):
                    continue
                if not records:
                    continue
                stats["doc_rows"] += 1
                unique: dict[str, int] = {}
                for surface, roles, _score in records:
                    unique[surface] = unique.get(surface, 0) | roles
                for surface, roles in unique.items():
                    counts = local[surface]
                    counts[0] += 1
                    if roles & TITLE_ROLE:
                        counts[1] += 1
                    if roles & TAG_ROLE:
                        counts[2] += 1
                    if roles & OWNER_ROLE:
                        counts[3] += 1
                    stats["term_rows"] += 1
        _flush_term_stats(connection, local)
        stats["segment_files"] += 1
    return stats


def _allowed_terms(
    connection: sqlite3.Connection,
    *,
    doc_count: int,
    min_df: int,
    max_df_ratio: float,
) -> dict[str, tuple[int, float]]:
    if doc_count <= 0:
        return {}
    max_df = doc_count
    if 0 < max_df_ratio < 1:
        max_df = max(min_df, int(doc_count * max_df_ratio))
    allowed: dict[str, tuple[int, float]] = {}
    for surface, df in connection.execute(
        "SELECT surface, df FROM term_stats WHERE df >= ? AND df <= ?",
        (min_df, max_df),
    ):
        idf = math.log((doc_count + 1.0) / (int(df) + 0.5))
        allowed[str(surface)] = (int(df), max(idf, 0.01))
    return allowed


def _rank_records(
    records: tuple[DocTermRecord, ...],
    allowed: dict[str, tuple[int, float]],
    max_terms_per_doc: int,
) -> list[tuple[str, float]]:
    ranked: list[tuple[str, float, float]] = []
    seen: set[str] = set()
    for surface, roles, score in records:
        if surface in seen or surface not in allowed:
            continue
        seen.add(surface)
        _df, idf = allowed[surface]
        role_bonus = 1.0
        if roles & TAG_ROLE:
            role_bonus += 0.18
        if roles & TITLE_ROLE:
            role_bonus += 0.06
        ranked.append((surface, score * role_bonus * idf, score))
    ranked.sort(key=lambda item: (-item[1], -item[2], item[0]))
    return [
        (surface, rank_score)
        for surface, rank_score, _score in ranked[:max_terms_per_doc]
    ]


def _pack_pair(left_id: int, right_id: int) -> int:
    if left_id > right_id:
        left_id, right_id = right_id, left_id
    return (left_id << 32) | right_id


def _unpack_pair(key: int) -> tuple[int, int]:
    return key >> 32, key & 0xFFFFFFFF


def _collect_edges(
    version_root: Path,
    *,
    allowed: dict[str, tuple[int, float]],
    filter_current_docs: bool,
    current_connection: sqlite3.Connection,
    max_terms_per_doc: int,
    max_pairs_per_doc: int,
    negative_samples_per_doc: int,
) -> tuple[dict[str, int], dict[int, list[float]], dict[int, int], list[str]]:
    stats = {"edge_pairs_seen": 0, "edge_pairs_kept": 0, "negative_rows_seen": 0}
    term_ids = {surface: index for index, surface in enumerate(allowed)}
    terms_by_id = list(allowed)
    edge_rows: dict[int, list[float]] = {}
    negative_rows: dict[int, int] = defaultdict(int)
    previous_term_ids: list[int] = []

    for path in _iter_doc_segment_paths(version_root):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                parsed = parse_doc_term_row(line)
                if parsed is None:
                    continue
                doc_key, content_hash, records = parsed
                if filter_current_docs and not _is_current_doc(
                    current_connection, doc_key, content_hash
                ):
                    continue
                ranked = _rank_records(records, allowed, max_terms_per_doc)
                if len(ranked) < 2:
                    previous_term_ids = [
                        term_ids[surface] for surface, _score in ranked
                    ]
                    continue
                pair_candidates = []
                for (source, source_score), (target, target_score) in combinations(
                    ranked, 2
                ):
                    source_id = term_ids[source]
                    target_id = term_ids[target]
                    if source_id == target_id:
                        continue
                    strength = math.sqrt(
                        max(source_score, 0.001) * max(target_score, 0.001)
                    )
                    pair_candidates.append(
                        (_pack_pair(source_id, target_id), strength)
                    )
                pair_candidates.sort(key=lambda item: (-item[1], item[0]))
                for pair_key, strength in pair_candidates[:max_pairs_per_doc]:
                    values = edge_rows.setdefault(pair_key, [0.0, 0.0])
                    values[0] += 1
                    values[1] += strength
                    stats["edge_pairs_seen"] += 1

                if negative_samples_per_doc > 0 and previous_term_ids:
                    positive_ids = {term_ids[surface] for surface, _score in ranked}
                    samples = [
                        term_id
                        for term_id in previous_term_ids
                        if term_id not in positive_ids
                    ]
                    for source_id in sorted(positive_ids)[:2]:
                        for target in samples[:negative_samples_per_doc]:
                            negative_rows[_pack_pair(source_id, target)] += 1
                            stats["negative_rows_seen"] += 1
                previous_term_ids = [term_ids[surface] for surface, _score in ranked]

    stats["edge_pairs_kept"] = len(edge_rows)
    return stats, edge_rows, dict(negative_rows), terms_by_id


def _add_rule(
    mapping: dict[str, dict[str, float]], source: str, target: str, weight: float
) -> None:
    source = normalize_surface(source)
    target = normalize_surface(target)
    if not source or not target or source == target:
        return
    expansions = mapping.setdefault(source, {})
    current = expansions.get(target)
    if current is None or current < weight:
        expansions[target] = weight


def _merge_mapping(
    target: dict[str, dict[str, float]], source: dict[str, dict[str, float]]
) -> dict[str, dict[str, float]]:
    for surface, expansions in source.items():
        for expansion, weight in expansions.items():
            _add_rule(target, surface, expansion, weight)
    return target


def _edge_weight(
    cooc_count: int,
    strength: float,
    source_df: int,
    target_df: int,
    doc_count: int,
) -> float:
    if cooc_count <= 0 or doc_count <= 0:
        return 0.0
    lift = (cooc_count * doc_count) / max(source_df * target_df, 1)
    lift_score = min(0.34, max(0.0, math.log(lift)) / 8.0)
    support_score = min(0.26, math.log1p(cooc_count) / 14.0)
    role_score = min(0.18, (strength / max(cooc_count, 1)) / 8.0)
    return round(min(0.96, 0.22 + lift_score + support_score + role_score), 4)


def _load_derived_near_synonym_mapping(
    edge_rows: dict[int, list[float]],
    *,
    doc_count: int,
    allowed: dict[str, tuple[int, float]],
    terms_by_id: list[str],
    min_cooc: int,
    top_k: int,
    min_score: float,
) -> dict[str, dict[str, float]]:
    mapping: dict[str, dict[str, float]] = defaultdict(dict)
    min_dynamic_score = max(0.42, min_score + 0.10)
    min_dynamic_cooc = max(3, min_cooc)
    candidates: dict[str, list[tuple[str, float, int]]] = defaultdict(list)
    for pair_key, values in edge_rows.items():
        cooc = int(values[0])
        if cooc < min_dynamic_cooc:
            continue
        source_id, target_id = _unpack_pair(pair_key)
        source = terms_by_id[source_id]
        target = terms_by_id[target_id]
        if not _can_promote_near_synonym(source, target):
            continue
        source_df = allowed.get(source, (0, 0.0))[0]
        target_df = allowed.get(target, (0, 0.0))[0]
        weight = _edge_weight(
            cooc, float(values[1]), source_df, target_df, doc_count
        )
        if weight < min_dynamic_score:
            continue
        candidates[source].append((target, weight, cooc))
        candidates[target].append((source, weight, cooc))

    for source, targets in candidates.items():
        ranked = sorted(targets, key=lambda item: (-item[1], -item[2], item[0]))[:top_k]
        for target, weight, _cooc in ranked:
            _add_rule(mapping, source, target, min(0.88, weight))
    return mapping


def _can_promote_near_synonym(source: str, target: str) -> bool:
    if source == target:
        return False
    source_compact = _semantic_variant_key(source)
    target_compact = _semantic_variant_key(target)
    source_len = len(source_compact)
    target_len = len(target_compact)
    if source_len < 2 or target_len < 2:
        return False
    if source_len > 16 or target_len > 16:
        return False
    if abs(source_len - target_len) > 8:
        return False
    if _is_semantic_fragment(source_compact) or _is_semantic_fragment(target_compact):
        return False
    if source_compact in target_compact or target_compact in source_compact:
        if min(source_len, target_len) <= 2:
            return False
        if _digit_signature(source_compact) != _digit_signature(target_compact):
            return False
        return abs(source_len - target_len) <= 4
    if _has_ascii_letter(source_compact) and _has_ascii_letter(target_compact):
        return _limited_edit_distance(source_compact, target_compact, 2) <= 2
    return False


def _semantic_variant_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(char for char in normalized if not char.isspace())


def _has_ascii_letter(value: str) -> bool:
    return any(char.isascii() and char.isalpha() for char in value)


def _has_cjk_text(value: str) -> bool:
    return any(_is_cjk(char) for char in value)


def _has_digit(value: str) -> bool:
    return any(char.isdigit() for char in value)


def _digit_signature(value: str) -> tuple[str, ...]:
    sequences: list[str] = []
    current: list[str] = []
    for char in value:
        if char.isdigit():
            current.append(char)
        elif current:
            sequences.append("".join(current))
            current.clear()
    if current:
        sequences.append("".join(current))
    return tuple(sequences)


def _is_cjk(char: str) -> bool:
    return "\u4e00" <= char <= "\u9fff"


def _is_semantic_fragment(value: str) -> bool:
    if not value:
        return True
    alnum_or_cjk = [char for char in value if char.isalnum() or _is_cjk(char)]
    if len(alnum_or_cjk) < 2:
        return True
    if all(char.isdigit() for char in alnum_or_cjk):
        return True
    if len(alnum_or_cjk) <= 3 and sum(char.isdigit() for char in alnum_or_cjk) >= 2:
        return True
    return False


def _limited_edit_distance(left: str, right: str, limit: int) -> int:
    if abs(len(left) - len(right)) > limit:
        return limit + 1
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        row_min = current[0]
        for right_index, right_char in enumerate(right, start=1):
            cost = 0 if left_char == right_char else 1
            value = min(
                previous[right_index] + 1,
                current[right_index - 1] + 1,
                previous[right_index - 1] + cost,
            )
            current.append(value)
            row_min = min(row_min, value)
        if row_min > limit:
            return limit + 1
        previous = current
    return previous[-1]


def _lex_rank(value: str) -> tuple[int, ...]:
    return tuple([-ord(char) for char in value] + [0])


def _candidate_rank(
    weight: float, cooc: int, target: str
) -> tuple[float, int, tuple[int, ...]]:
    return (float(weight), int(cooc), _lex_rank(target))


def _push_top_candidate(
    heap: list[tuple[tuple[float, int, tuple[int, ...]], str, float, int]],
    *,
    target: str,
    weight: float,
    cooc: int,
    top_k: int,
) -> None:
    if top_k <= 0:
        return
    item = (_candidate_rank(weight, cooc, target), target, weight, cooc)
    if len(heap) < top_k:
        heappush(heap, item)
    elif item[0] > heap[0][0]:
        heapreplace(heap, item)


def _build_doc_cooccurrence_mapping(
    edge_rows: dict[int, list[float]],
    *,
    doc_count: int,
    allowed: dict[str, tuple[int, float]],
    terms_by_id: list[str],
    min_cooc: int,
    top_k: int,
    min_score: float,
) -> dict[str, dict[str, float]]:
    heaps_by_source: dict[
        str, list[tuple[tuple[float, int, tuple[int, ...]], str, float, int]]
    ] = defaultdict(list)
    for pair_key, values in edge_rows.items():
        cooc = int(values[0])
        if cooc < min_cooc:
            continue
        source_id, target_id = _unpack_pair(pair_key)
        source = terms_by_id[source_id]
        target = terms_by_id[target_id]
        source_df = allowed.get(source, (0, 0.0))[0]
        target_df = allowed.get(target, (0, 0.0))[0]
        weight = _edge_weight(
            cooc, float(values[1]), source_df, target_df, doc_count
        )
        if weight < min_score:
            continue
        _push_top_candidate(
            heaps_by_source[source],
            target=target,
            weight=weight,
            cooc=cooc,
            top_k=top_k,
        )
        _push_top_candidate(
            heaps_by_source[target],
            target=source,
            weight=weight,
            cooc=cooc,
            top_k=top_k,
        )

    mapping: dict[str, dict[str, float]] = defaultdict(dict)
    for source in sorted(heaps_by_source):
        ranked = sorted(
            heaps_by_source[source],
            key=lambda item: (-item[2], -item[3], item[1]),
        )
        for _rank, target, weight, _cooc in ranked:
            _add_rule(mapping, source, target, weight)
    return mapping


def _write_compact_mapping(path: Path, mapping: dict[str, dict[str, float]]) -> int:
    row_count = 0
    with path.open("w", encoding="utf-8") as handle:
        for source in sorted(mapping):
            targets = sorted(
                mapping[source].items(), key=lambda item: (-item[1], item[0])
            )
            if not targets:
                continue
            cells = [encode_normalized_term(source)]
            for target, weight in targets:
                cells.extend([encode_normalized_term(target), format_weight(weight)])
            handle.write("\t".join(cells) + "\n")
            row_count += 1
    return row_count


def _bridge_profile_ok(profile: TermProfile | None) -> bool:
    if profile is None:
        return True
    _df, title_df, tag_df = profile
    if tag_df < 8:
        return False
    if tag_df >= title_df:
        return True
    return (float(tag_df) / max(float(title_df), 1.0)) >= 0.25


def _bridge_profile_score(profile: TermProfile | None) -> float:
    if profile is None:
        return 1.0
    df, title_df, tag_df = profile
    tag_ratio = float(tag_df) / max(float(title_df), 1.0)
    tag_bias = min(4.0, tag_ratio)
    return (1.0 + tag_bias) * (math.log1p(tag_df) / max(math.log1p(df), 1.0))


def _load_term_profiles(
    connection: sqlite3.Connection, allowed: set[str]
) -> dict[str, TermProfile]:
    profiles: dict[str, TermProfile] = {}
    for surface, df, title_df, tag_df in connection.execute(
        "SELECT surface, df, title_df, tag_df FROM term_stats"
    ):
        if surface not in allowed:
            continue
        profiles[str(surface)] = (int(df), int(title_df), int(tag_df))
    return profiles


def _can_consider_semantic_bridge(
    source: str,
    target: str,
    term_profiles: dict[str, TermProfile] | None = None,
) -> bool:
    if source == target:
        return False
    source_key = _semantic_variant_key(source)
    target_key = _semantic_variant_key(target)
    if len(source_key) < 2 or len(target_key) < 2:
        return False
    if len(source_key) > 24 or len(target_key) > 24:
        return False
    if _is_semantic_fragment(source_key) or _is_semantic_fragment(target_key):
        return False
    if _has_digit(source_key) or _has_digit(target_key):
        return False
    if term_profiles is not None and (
        not _bridge_profile_ok(term_profiles.get(source))
        or not _bridge_profile_ok(term_profiles.get(target))
    ):
        return False

    source_has_cjk = _has_cjk_text(source_key)
    target_has_cjk = _has_cjk_text(target_key)
    source_has_ascii = _has_ascii_letter(source_key)
    target_has_ascii = _has_ascii_letter(target_key)
    if source_has_cjk != target_has_cjk and (source_has_ascii or target_has_ascii):
        return True
    return source_has_cjk and target_has_cjk


def _promote_embedding_semantic_bridges(
    synonym_mapping: dict[str, dict[str, float]],
    doc_cooccurrence_mapping: dict[str, dict[str, float]],
    similarity_fn,
    *,
    min_weight: float,
    min_score: float,
    cjk_min_score: float,
    mixed_script_min_score: float,
    max_sources: int,
    max_targets_per_source: int,
    term_profiles: dict[str, TermProfile] | None = None,
) -> dict[str, int | float]:
    stats: dict[str, int | float] = {
        "sources_seen": 0,
        "sources_scored": 0,
        "targets_seen": 0,
        "targets_scored": 0,
        "targets_promoted": 0,
        "targets_rejected_by_similarity": 0,
        "min_weight": round(float(min_weight), 4),
    }
    ranked_sources = sorted(
        doc_cooccurrence_mapping,
        key=lambda source: (
            -_bridge_profile_score(
                term_profiles.get(source) if term_profiles is not None else None
            ),
            -max(
                (
                    float(weight)
                    for target, weight in doc_cooccurrence_mapping[source].items()
                    if _can_consider_semantic_bridge(source, target, term_profiles)
                ),
                default=0.0,
            ),
            -len(doc_cooccurrence_mapping[source]),
            normalize_surface(source),
        ),
    )
    source_candidates: list[tuple[str, list[tuple[str, float]]]] = []
    for source in ranked_sources:
        if max_sources > 0 and len(source_candidates) >= max_sources:
            break
        candidates = [
            (target, float(weight))
            for target, weight in doc_cooccurrence_mapping[source].items()
            if float(weight) >= min_weight
            and _can_consider_semantic_bridge(source, target, term_profiles)
        ]
        stats["targets_seen"] = int(stats["targets_seen"]) + len(candidates)
        if not candidates:
            continue
        candidates.sort(key=lambda item: (-item[1], item[0]))
        if max_targets_per_source > 0:
            candidates = candidates[:max_targets_per_source]
        source_candidates.append((source, candidates))

    preload_terms = getattr(similarity_fn, "preload_terms", None)
    if callable(preload_terms):
        terms: set[str] = set()
        for source, candidates in source_candidates:
            terms.add(source)
            terms.update(target for target, _weight in candidates)
        preload_terms(terms)

    for source, candidates in source_candidates:

        stats["sources_seen"] = int(stats["sources_seen"]) + 1
        targets = [target for target, _weight in candidates]
        similarities = getattr(similarity_fn, "similarities", None)
        scores = (
            similarities(source, targets)
            if callable(similarities)
            else similarity_fn(source, targets)
        )
        if len(scores) != len(targets):
            continue
        stats["sources_scored"] = int(stats["sources_scored"]) + 1
        stats["targets_scored"] = int(stats["targets_scored"]) + len(targets)

        for (target, weight), score in zip(candidates, scores):
            threshold = min_similarity_for_pair(
                source,
                target,
                min_score=min_score,
                cjk_min_score=cjk_min_score,
                mixed_script_min_score=mixed_script_min_score,
            )
            if float(score) < threshold:
                stats["targets_rejected_by_similarity"] = (
                    int(stats["targets_rejected_by_similarity"]) + 1
                )
                continue
            promoted_weight = round(
                min(0.94, max(0.0, float(weight)) * (0.7 + 0.3 * float(score))),
                4,
            )
            _add_rule(synonym_mapping, source, target, promoted_weight)
            stats["targets_promoted"] = int(stats["targets_promoted"]) + 1
    return stats


def _write_nodes(path: Path, connection: sqlite3.Connection, allowed: set[str]) -> int:
    row_count = 0
    with path.open("w", encoding="utf-8") as handle:
        for surface, df, title_df, tag_df, owner_df in connection.execute(
            "SELECT surface, df, title_df, tag_df, owner_df FROM term_stats ORDER BY df DESC, surface"
        ):
            if surface not in allowed:
                continue
            kind = "tag" if int(tag_df) >= int(title_df) else "vocab"
            handle.write(
                "\t".join(
                    [
                        encode_normalized_term(str(surface)),
                        kind,
                        str(int(df)),
                        str(int(title_df)),
                        str(int(tag_df)),
                        str(int(owner_df)),
                    ]
                )
                + "\n"
            )
            row_count += 1
    return row_count


def _write_negative_samples(
    path: Path, negative_rows: dict[int, int], terms_by_id: list[str]
) -> int:
    row_count = 0
    with path.open("w", encoding="utf-8") as handle:
        for pair_key, count in sorted(
            negative_rows.items(),
            key=lambda item: (
                -item[1],
                terms_by_id[_unpack_pair(item[0])[0]],
                terms_by_id[_unpack_pair(item[0])[1]],
            ),
        ):
            source_id, target_id = _unpack_pair(pair_key)
            handle.write(
                "\t".join(
                    [
                        encode_normalized_term(terms_by_id[source_id]),
                        encode_normalized_term(terms_by_id[target_id]),
                        str(int(count)),
                    ]
                )
                + "\n"
            )
            row_count += 1
    return row_count


def merge_groups(
    version_root: Path | str,
    *,
    output_dir: Path | str | None = None,
    min_df: int = 3,
    min_cooc: int = 2,
    top_k: int = 32,
    max_df_ratio: float = 0.08,
    max_terms_per_doc: int = 12,
    max_pairs_per_doc: int = 48,
    negative_samples_per_doc: int = 0,
    min_score: float = 0.28,
    keep_merge_db: bool = False,
    embedding_filter_enabled: bool = False,
    embedding_endpoints: str | list[str] | None = None,
    embedding_min_score: float = 0.52,
    embedding_cjk_min_score: float = 0.58,
    embedding_mixed_script_min_score: float = 0.62,
    embedding_max_sources: int = 20000,
    embedding_max_targets_per_source: int = 24,
    embedding_filter_near_synonym: bool = True,
    embedding_filter_doc_cooccurrence: bool = True,
    embedding_bridge_promotion_enabled: bool = False,
    embedding_bridge_min_weight: float = 0.72,
    embedding_bridge_max_sources: int = 1200,
    embedding_bridge_max_targets_per_source: int = 8,
    embedding_bridge_scorer: str = "embed",
    embedding_bridge_lsh_bits: int = 2048,
) -> dict[str, int | float | str]:
    version_root = Path(version_root)
    output_dir = Path(output_dir) if output_dir else version_root / "merged"
    _clean_output_dir(output_dir)
    merge_db_path = output_dir / ".merge.sqlite"
    connection = _connect_merge_db(merge_db_path)
    success = False
    try:
        current_doc_stats = _collect_current_docs(version_root, connection)
        filter_current_docs = (
            current_doc_stats["segment_rows_seen"] != current_doc_stats["current_docs"]
        )
        read_stats = _collect_term_stats(
            version_root,
            connection,
            filter_current_docs=filter_current_docs,
        )
        doc_count = read_stats["doc_rows"]
        allowed = _allowed_terms(
            connection,
            doc_count=doc_count,
            min_df=min_df,
            max_df_ratio=max_df_ratio,
        )
        term_profiles = _load_term_profiles(connection, set(allowed))
        edge_stats, edge_rows, negative_rows, terms_by_id = _collect_edges(
            version_root,
            allowed=allowed,
            filter_current_docs=filter_current_docs,
            current_connection=connection,
            max_terms_per_doc=max_terms_per_doc,
            max_pairs_per_doc=max_pairs_per_doc,
            negative_samples_per_doc=negative_samples_per_doc,
        )
        rewrite_mapping: dict[str, dict[str, float]] = defaultdict(dict)
        synonym_mapping: dict[str, dict[str, float]] = defaultdict(dict)
        near_synonym_mapping: dict[str, dict[str, float]] = defaultdict(dict)
        embedding_filter_stats: dict[str, dict] = {}
        _merge_mapping(
            near_synonym_mapping,
            _load_derived_near_synonym_mapping(
                edge_rows,
                doc_count=doc_count,
                allowed=allowed,
                terms_by_id=terms_by_id,
                min_cooc=min_cooc,
                top_k=max(4, min(top_k, 12)),
                min_score=min_score,
            ),
        )
        doc_cooccurrence_mapping = _build_doc_cooccurrence_mapping(
            edge_rows,
            doc_count=doc_count,
            allowed=allowed,
            terms_by_id=terms_by_id,
            min_cooc=min_cooc,
            top_k=top_k,
            min_score=min_score,
        )
        if embedding_filter_enabled:
            needs_rerank_filter = (
                embedding_filter_near_synonym or embedding_filter_doc_cooccurrence
            )
            scorer = (
                TeiSimilarityScorer(embedding_endpoints)
                if needs_rerank_filter
                else None
            )
            if scorer is not None and scorer.is_available():
                if embedding_filter_near_synonym:
                    near_synonym_mapping, stats = filter_mapping_by_similarity(
                        near_synonym_mapping,
                        scorer.similarities,
                        min_score=embedding_min_score,
                        cjk_min_score=embedding_cjk_min_score,
                        mixed_script_min_score=embedding_mixed_script_min_score,
                        max_sources=embedding_max_sources,
                        max_targets_per_source=embedding_max_targets_per_source,
                        reweight=True,
                    )
                    embedding_filter_stats["near_synonym"] = stats.to_dict()
                else:
                    embedding_filter_stats["near_synonym"] = {
                        "enabled": False,
                        "reason": "disabled_by_merge_option",
                    }
                if embedding_filter_doc_cooccurrence:
                    doc_cooccurrence_mapping, stats = filter_mapping_by_similarity(
                        doc_cooccurrence_mapping,
                        scorer.similarities,
                        min_score=embedding_min_score,
                        cjk_min_score=embedding_cjk_min_score,
                        mixed_script_min_score=embedding_mixed_script_min_score,
                        max_sources=embedding_max_sources,
                        max_targets_per_source=embedding_max_targets_per_source,
                        reweight=True,
                    )
                    embedding_filter_stats["doc_cooccurrence"] = stats.to_dict()
                else:
                    embedding_filter_stats["doc_cooccurrence"] = {
                        "enabled": False,
                        "reason": "disabled_by_merge_option",
                    }
            elif needs_rerank_filter:
                embedding_filter_stats["skipped"] = {
                    "enabled": True,
                    "reason": "tei_unavailable",
                }
            if embedding_bridge_promotion_enabled:
                if embedding_bridge_scorer == "lsh":
                    bridge_scorer = TeiLshSimilarityScorer(
                        embedding_endpoints,
                        bitn=embedding_bridge_lsh_bits,
                    )
                    bridge_min_score = max(embedding_min_score, 0.68)
                    bridge_cjk_min_score = max(embedding_cjk_min_score, 0.72)
                    bridge_mixed_script_min_score = max(
                        embedding_mixed_script_min_score,
                        0.70,
                    )
                else:
                    bridge_scorer = TeiEmbeddingSimilarityScorer(embedding_endpoints)
                    bridge_min_score = embedding_min_score
                    bridge_cjk_min_score = embedding_cjk_min_score
                    bridge_mixed_script_min_score = embedding_mixed_script_min_score
                if bridge_scorer.is_available():
                    embedding_filter_stats["semantic_bridge"] = (
                        _promote_embedding_semantic_bridges(
                            synonym_mapping,
                            doc_cooccurrence_mapping,
                            bridge_scorer,
                            min_weight=embedding_bridge_min_weight,
                            min_score=bridge_min_score,
                            cjk_min_score=bridge_cjk_min_score,
                            mixed_script_min_score=bridge_mixed_script_min_score,
                            max_sources=embedding_bridge_max_sources,
                            max_targets_per_source=(
                                embedding_bridge_max_targets_per_source
                            ),
                            term_profiles=term_profiles,
                        )
                    )
                    embedding_filter_stats["semantic_bridge"][
                        "scorer"
                    ] = embedding_bridge_scorer
                    embedding_filter_stats["semantic_bridge"][
                        "lsh_bits"
                    ] = (
                        embedding_bridge_lsh_bits
                        if embedding_bridge_scorer == "lsh"
                        else 0
                    )
                    embedding_filter_stats["semantic_bridge"]["min_score"] = (
                        bridge_min_score
                    )
                    embedding_filter_stats["semantic_bridge"]["cjk_min_score"] = (
                        bridge_cjk_min_score
                    )
                    embedding_filter_stats["semantic_bridge"][
                        "mixed_script_min_score"
                    ] = bridge_mixed_script_min_score
                else:
                    embedding_filter_stats["semantic_bridge"] = {
                        "enabled": True,
                        "reason": "tei_unavailable",
                        "scorer": embedding_bridge_scorer,
                    }
        relation_rows = {
            "node_rows": _write_nodes(
                output_dir / "nodes.tsv", connection, set(allowed)
            ),
            "negative_rows": _write_negative_samples(
                output_dir / "negative_samples.tsv", negative_rows, terms_by_id
            ),
            "rewrite_rows": _write_compact_mapping(
                output_dir / "rewrite.tsv", rewrite_mapping
            ),
            "synonym_rows": _write_compact_mapping(
                output_dir / "synonym.tsv", synonym_mapping
            ),
            "near_synonym_rows": _write_compact_mapping(
                output_dir / "near_synonym.tsv", near_synonym_mapping
            ),
            "doc_cooccurrence_rows": _write_compact_mapping(
                output_dir / "doc_cooccurrence.tsv",
                doc_cooccurrence_mapping,
            ),
        }
        total_terms_row = connection.execute(
            "SELECT COUNT(*) FROM term_stats"
        ).fetchone()
        meta = {
            "output_dir": str(output_dir),
            "doc_count": doc_count,
            "total_terms": int(total_terms_row[0] if total_terms_row else 0),
            "allowed_terms": len(allowed),
            "min_df": min_df,
            "min_cooc": min_cooc,
            "max_df_ratio": max_df_ratio,
            "max_terms_per_doc": max_terms_per_doc,
            "max_pairs_per_doc": max_pairs_per_doc,
            "min_score": min_score,
            "merge_db_kept": keep_merge_db,
            "embedding_filter_enabled": embedding_filter_enabled,
            "embedding_filter_near_synonym": embedding_filter_near_synonym,
            "embedding_filter_doc_cooccurrence": embedding_filter_doc_cooccurrence,
            "embedding_bridge_promotion_enabled": embedding_bridge_promotion_enabled,
            "embedding_bridge_min_weight": embedding_bridge_min_weight,
            "embedding_bridge_scorer": embedding_bridge_scorer,
            "embedding_bridge_lsh_bits": embedding_bridge_lsh_bits,
            "embedding_filter": embedding_filter_stats,
            "current_doc_filter_enabled": filter_current_docs,
            **current_doc_stats,
            **read_stats,
            **edge_stats,
            **relation_rows,
        }
        (output_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        success = True
        return meta
    finally:
        connection.close()
        if success and not keep_merge_db and merge_db_path.exists():
            merge_db_path.unlink()
