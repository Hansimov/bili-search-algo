from __future__ import annotations

import json
import math
import sqlite3
import time

from collections import defaultdict
from collections.abc import Iterable
from heapq import heappush, heapreplace
from itertools import combinations
from pathlib import Path

from models.semantics.graph import ExtractedDoc, TermRecord, TermRole
from models.semantics.vocab import collapse_text, normalize_surface


TSV_SPACE_MASK = "▂"
DEFAULT_REWRITE_RULES: dict[str, tuple[str, ...]] = {
    "专访": ("采访", "访谈"),
    "访谈": ("采访", "专访"),
    "评测": ("测评",),
    "测评": ("评测",),
    "康夫 ui": ("comfyui",),
}
DEFAULT_SYNONYM_GROUPS: tuple[tuple[str, ...], ...] = (
    ("采访", "访谈", "专访"),
    ("教程", "教学", "讲解"),
    ("评测", "测评"),
)
DEFAULT_NEAR_SYNONYM_GROUPS: tuple[tuple[float, tuple[str, ...]], ...] = (
    (0.82, ("开箱", "上手", "体验")),
    (0.78, ("解析", "解读", "盘点")),
    (0.74, ("整活", "搞笑", "沙雕")),
)
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


def decode_term(value: object) -> str:
    return normalize_surface(str(value or "").replace(TSV_SPACE_MASK, " "))


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


def parse_doc_term_row(line: str) -> tuple[str, str, tuple[TermRecord, ...]] | None:
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 5:
        return None
    doc_key = decode_raw_cell(parts[0])
    content_hash = parts[1]
    records: list[TermRecord] = []
    for index in range(2, len(parts) - 2, 3):
        surface = decode_term(parts[index])
        if not surface:
            continue
        try:
            roles = int(parts[index + 1])
            score = float(parts[index + 2])
        except ValueError:
            continue
        records.append(TermRecord(surface, roles, score))
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
        "CREATE TABLE doc_edges("
        "source TEXT NOT NULL, "
        "target TEXT NOT NULL, "
        "cooc INTEGER NOT NULL, "
        "strength REAL NOT NULL, "
        "PRIMARY KEY(source, target))"
    )
    connection.execute(
        "CREATE TABLE negatives("
        "source TEXT NOT NULL, "
        "target TEXT NOT NULL, "
        "count INTEGER NOT NULL, "
        "PRIMARY KEY(source, target))"
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
                parsed = parse_doc_term_row(line)
                if parsed is None:
                    continue
                doc_key, content_hash, _records = parsed
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
                for record in records:
                    unique[record.surface] = (
                        unique.get(record.surface, 0) | record.roles
                    )
                for surface, roles in unique.items():
                    counts = local[surface]
                    counts[0] += 1
                    if roles & TermRole.TITLE:
                        counts[1] += 1
                    if roles & TermRole.TAG:
                        counts[2] += 1
                    if roles & TermRole.OWNER:
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
    records: tuple[TermRecord, ...],
    allowed: dict[str, tuple[int, float]],
    max_terms_per_doc: int,
) -> list[tuple[TermRecord, float]]:
    ranked: list[tuple[TermRecord, float]] = []
    seen: set[str] = set()
    for record in records:
        if record.surface in seen or record.surface not in allowed:
            continue
        seen.add(record.surface)
        _df, idf = allowed[record.surface]
        role_bonus = 1.0
        if record.roles & TermRole.TAG:
            role_bonus += 0.18
        if record.roles & TermRole.TITLE:
            role_bonus += 0.06
        ranked.append((record, record.score * role_bonus * idf))
    ranked.sort(key=lambda item: (-item[1], -item[0].score, item[0].surface))
    return ranked[:max_terms_per_doc]


def _flush_edges(
    connection: sqlite3.Connection, rows: dict[tuple[str, str], list[float]]
) -> None:
    if not rows:
        return
    connection.executemany(
        "INSERT INTO doc_edges(source, target, cooc, strength) VALUES (?, ?, ?, ?) "
        "ON CONFLICT(source, target) DO UPDATE SET "
        "cooc = cooc + excluded.cooc, strength = strength + excluded.strength",
        [
            (source, target, int(values[0]), float(values[1]))
            for (source, target), values in rows.items()
        ],
    )
    connection.commit()


def _flush_negatives(
    connection: sqlite3.Connection, rows: dict[tuple[str, str], int]
) -> None:
    if not rows:
        return
    connection.executemany(
        "INSERT INTO negatives(source, target, count) VALUES (?, ?, ?) "
        "ON CONFLICT(source, target) DO UPDATE SET count = count + excluded.count",
        [(source, target, count) for (source, target), count in rows.items()],
    )
    connection.commit()


def _collect_edges(
    version_root: Path,
    connection: sqlite3.Connection,
    *,
    allowed: dict[str, tuple[int, float]],
    filter_current_docs: bool,
    max_terms_per_doc: int,
    max_pairs_per_doc: int,
    negative_samples_per_doc: int,
) -> dict[str, int]:
    stats = {"edge_pairs_seen": 0, "edge_pairs_kept": 0, "negative_rows_seen": 0}
    edge_rows: dict[tuple[str, str], list[float]] = {}
    negative_rows: dict[tuple[str, str], int] = defaultdict(int)
    previous_terms: list[str] = []

    for path in _iter_doc_segment_paths(version_root):
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
                ranked = _rank_records(records, allowed, max_terms_per_doc)
                if len(ranked) < 2:
                    previous_terms = [record.surface for record, _score in ranked]
                    continue
                pair_candidates = []
                for (source_record, source_score), (
                    target_record,
                    target_score,
                ) in combinations(ranked, 2):
                    source = source_record.surface
                    target = target_record.surface
                    if source == target:
                        continue
                    if source > target:
                        source, target = target, source
                    strength = math.sqrt(
                        max(source_score, 0.001) * max(target_score, 0.001)
                    )
                    pair_candidates.append((source, target, strength))
                pair_candidates.sort(key=lambda item: (-item[2], item[0], item[1]))
                for source, target, strength in pair_candidates[:max_pairs_per_doc]:
                    values = edge_rows.setdefault((source, target), [0.0, 0.0])
                    values[0] += 1
                    values[1] += strength
                    stats["edge_pairs_seen"] += 1

                if negative_samples_per_doc > 0 and previous_terms:
                    positives = {record.surface for record, _score in ranked}
                    samples = [term for term in previous_terms if term not in positives]
                    for source in sorted(positives)[:2]:
                        for target in samples[:negative_samples_per_doc]:
                            negative_rows[(source, target)] += 1
                            stats["negative_rows_seen"] += 1
                previous_terms = [record.surface for record, _score in ranked]

                if len(edge_rows) >= 100000:
                    _flush_edges(connection, edge_rows)
                    edge_rows.clear()
                if len(negative_rows) >= 50000:
                    _flush_negatives(connection, negative_rows)
                    negative_rows.clear()
    _flush_edges(connection, edge_rows)
    _flush_negatives(connection, negative_rows)
    row = connection.execute("SELECT COUNT(*) FROM doc_edges").fetchone()
    stats["edge_pairs_kept"] = int(row[0] if row else 0)
    return stats


def _add_rule(
    mapping: dict[str, dict[str, float]], source: str, target: str, weight: float
) -> None:
    source = normalize_surface(source)
    target = normalize_surface(target)
    if not source or not target or source == target:
        return
    current = mapping[source].get(target)
    if current is None or current < weight:
        mapping[source][target] = weight


def _seed_rewrite_mapping() -> dict[str, dict[str, float]]:
    mapping: dict[str, dict[str, float]] = defaultdict(dict)
    for source, targets in DEFAULT_REWRITE_RULES.items():
        for target in targets:
            _add_rule(mapping, source, target, 1.0)
    return mapping


def _seed_group_mapping(
    groups: tuple[tuple[str, ...], ...], weight: float
) -> dict[str, dict[str, float]]:
    mapping: dict[str, dict[str, float]] = defaultdict(dict)
    for terms in groups:
        for source in terms:
            for target in terms:
                _add_rule(mapping, source, target, weight)
    return mapping


def _seed_weighted_group_mapping(
    groups: tuple[tuple[float, tuple[str, ...]], ...]
) -> dict[str, dict[str, float]]:
    mapping: dict[str, dict[str, float]] = defaultdict(dict)
    for weight, terms in groups:
        for source in terms:
            for target in terms:
                _add_rule(mapping, source, target, weight)
    return mapping


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


def _load_doc_cooccurrence_mapping(
    connection: sqlite3.Connection,
    *,
    doc_count: int,
    allowed: dict[str, tuple[int, float]],
    min_cooc: int,
    top_k: int,
    min_score: float,
) -> dict[str, dict[str, float]]:
    candidates: dict[str, list[tuple[str, float, int]]] = defaultdict(list)
    for source, target, cooc, strength in connection.execute(
        "SELECT source, target, cooc, strength FROM doc_edges WHERE cooc >= ?",
        (min_cooc,),
    ):
        source_df = allowed.get(source, (0, 0.0))[0]
        target_df = allowed.get(target, (0, 0.0))[0]
        weight = _edge_weight(
            int(cooc), float(strength), source_df, target_df, doc_count
        )
        if weight < min_score:
            continue
        candidates[source].append((target, weight, int(cooc)))
        candidates[target].append((source, weight, int(cooc)))

    mapping: dict[str, dict[str, float]] = defaultdict(dict)
    for source, targets in candidates.items():
        ranked = sorted(targets, key=lambda item: (-item[1], -item[2], item[0]))[:top_k]
        for target, weight, _count in ranked:
            _add_rule(mapping, source, target, weight)
    return mapping


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


def _flush_doc_cooccurrence_candidates(
    connection: sqlite3.Connection,
    source: str | None,
    heap: list[tuple[tuple[float, int, tuple[int, ...]], str, float, int]],
) -> int:
    if source is None or not heap:
        return 0
    rows = [
        (source, target, float(weight), int(cooc))
        for _rank, target, weight, cooc in heap
    ]
    connection.executemany(
        "INSERT INTO doc_cooc_candidates(source, target, weight, cooc) VALUES (?, ?, ?, ?) "
        "ON CONFLICT(source, target) DO UPDATE SET "
        "weight = max(weight, excluded.weight), cooc = max(cooc, excluded.cooc)",
        rows,
    )
    return len(rows)


def _collect_doc_cooccurrence_candidates(
    connection: sqlite3.Connection,
    *,
    doc_count: int,
    allowed: dict[str, tuple[int, float]],
    min_cooc: int,
    top_k: int,
    min_score: float,
    reverse: bool = False,
) -> int:
    if reverse:
        query = (
            "SELECT target, source, cooc, strength FROM doc_edges "
            "WHERE cooc >= ? ORDER BY target, source"
        )
    else:
        query = (
            "SELECT source, target, cooc, strength FROM doc_edges "
            "WHERE cooc >= ? ORDER BY source, target"
        )

    candidate_rows = 0
    uncommitted_rows = 0
    current_source: str | None = None
    heap: list[tuple[tuple[float, int, tuple[int, ...]], str, float, int]] = []

    for source, target, cooc, strength in connection.execute(query, (min_cooc,)):
        source = str(source)
        target = str(target)
        if source != current_source:
            flushed_rows = _flush_doc_cooccurrence_candidates(
                connection, current_source, heap
            )
            candidate_rows += flushed_rows
            uncommitted_rows += flushed_rows
            if uncommitted_rows >= 100000:
                connection.commit()
                uncommitted_rows = 0
            current_source = source
            heap = []

        source_df = allowed.get(source, (0, 0.0))[0]
        target_df = allowed.get(target, (0, 0.0))[0]
        weight = _edge_weight(
            int(cooc), float(strength), source_df, target_df, doc_count
        )
        if weight < min_score:
            continue
        _push_top_candidate(
            heap,
            target=target,
            weight=weight,
            cooc=int(cooc),
            top_k=top_k,
        )

    flushed_rows = _flush_doc_cooccurrence_candidates(connection, current_source, heap)
    candidate_rows += flushed_rows
    if uncommitted_rows + flushed_rows > 0:
        connection.commit()
    return candidate_rows


def _write_doc_cooccurrence_mapping(
    path: Path,
    connection: sqlite3.Connection,
    *,
    doc_count: int,
    allowed: dict[str, tuple[int, float]],
    min_cooc: int,
    top_k: int,
    min_score: float,
) -> int:
    connection.execute("DROP TABLE IF EXISTS doc_cooc_candidates")
    connection.execute(
        "CREATE TABLE doc_cooc_candidates("
        "source TEXT NOT NULL, "
        "target TEXT NOT NULL, "
        "weight REAL NOT NULL, "
        "cooc INTEGER NOT NULL, "
        "PRIMARY KEY(source, target))"
    )
    connection.commit()

    _collect_doc_cooccurrence_candidates(
        connection,
        doc_count=doc_count,
        allowed=allowed,
        min_cooc=min_cooc,
        top_k=top_k,
        min_score=min_score,
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_doc_edges_target_source ON doc_edges(target, source)"
    )
    connection.commit()
    _collect_doc_cooccurrence_candidates(
        connection,
        doc_count=doc_count,
        allowed=allowed,
        min_cooc=min_cooc,
        top_k=top_k,
        min_score=min_score,
        reverse=True,
    )

    row_count = 0
    current_source: str | None = None
    targets: list[tuple[str, float, int]] = []

    def flush_source() -> None:
        nonlocal row_count, targets
        if current_source is None or not targets:
            return
        ranked = sorted(targets, key=lambda item: (-item[1], -item[2], item[0]))[:top_k]
        cells = [encode_term(current_source)]
        for target, weight, _cooc in ranked:
            cells.extend([encode_term(target), format_weight(weight)])
        handle.write("\t".join(cells) + "\n")
        row_count += 1
        targets = []

    with path.open("w", encoding="utf-8") as handle:
        for source, target, weight, cooc in connection.execute(
            "SELECT source, target, weight, cooc FROM doc_cooc_candidates ORDER BY source, target"
        ):
            source = str(source)
            if source != current_source:
                flush_source()
                current_source = source
            targets.append((str(target), float(weight), int(cooc)))
        flush_source()
    return row_count


def _write_compact_mapping(path: Path, mapping: dict[str, dict[str, float]]) -> int:
    row_count = 0
    with path.open("w", encoding="utf-8") as handle:
        for source in sorted(mapping):
            targets = sorted(
                mapping[source].items(), key=lambda item: (-item[1], item[0])
            )
            if not targets:
                continue
            cells = [encode_term(source)]
            for target, weight in targets:
                cells.extend([encode_term(target), format_weight(weight)])
            handle.write("\t".join(cells) + "\n")
            row_count += 1
    return row_count


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
                        encode_term(surface),
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


def _write_negative_samples(path: Path, connection: sqlite3.Connection) -> int:
    row_count = 0
    with path.open("w", encoding="utf-8") as handle:
        for source, target, count in connection.execute(
            "SELECT source, target, count FROM negatives ORDER BY count DESC, source, target"
        ):
            handle.write(
                "\t".join([encode_term(source), encode_term(target), str(int(count))])
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
    negative_samples_per_doc: int = 4,
    min_score: float = 0.28,
    keep_merge_db: bool = False,
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
        edge_stats = _collect_edges(
            version_root,
            connection,
            allowed=allowed,
            filter_current_docs=filter_current_docs,
            max_terms_per_doc=max_terms_per_doc,
            max_pairs_per_doc=max_pairs_per_doc,
            negative_samples_per_doc=negative_samples_per_doc,
        )
        relation_rows = {
            "node_rows": _write_nodes(
                output_dir / "nodes.tsv", connection, set(allowed)
            ),
            "negative_rows": _write_negative_samples(
                output_dir / "negative_samples.tsv", connection
            ),
            "rewrite_rows": _write_compact_mapping(
                output_dir / "rewrite.tsv", _seed_rewrite_mapping()
            ),
            "synonym_rows": _write_compact_mapping(
                output_dir / "synonym.tsv",
                _seed_group_mapping(DEFAULT_SYNONYM_GROUPS, 0.92),
            ),
            "near_synonym_rows": _write_compact_mapping(
                output_dir / "near_synonym.tsv",
                _seed_weighted_group_mapping(DEFAULT_NEAR_SYNONYM_GROUPS),
            ),
            "doc_cooccurrence_rows": _write_doc_cooccurrence_mapping(
                output_dir / "doc_cooccurrence.tsv",
                connection,
                doc_count=doc_count,
                allowed=allowed,
                min_cooc=min_cooc,
                top_k=top_k,
                min_score=min_score,
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
