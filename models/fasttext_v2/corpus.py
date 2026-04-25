from __future__ import annotations

import json
import re

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from models.semantics.graph import TermRole
from models.semantics.storage import (
    encode_normalized_term,
    parse_doc_term_row,
)


@dataclass(frozen=True, slots=True)
class FastTextV2VocabEntry:
    token: str
    kind: str
    df: int
    title_df: int
    tag_df: int
    owner_df: int

    @property
    def tag_ratio(self) -> float:
        return float(self.tag_df) / max(float(self.df), 1.0)


_MODEL_CODE_RE = re.compile(
    r"^(?=.*[a-zA-Z])(?=.*\d)[a-zA-Z0-9][a-zA-Z0-9._+-]{1,31}$"
)
_NOISY_PUNCT_RE = re.compile(r"[!?！？]{1,}")


@dataclass(frozen=True, slots=True)
class FastTextV2CorpusConfig:
    min_token_score: float = 0.01
    min_terms_per_doc: int = 2
    max_terms_per_doc: int = 16
    max_docs: int = 0
    tag_repeat: int = 2
    title_repeat: int = 1
    add_role_tokens: bool = True
    add_doc_anchor: bool = True
    include_doc_sentences: bool = True
    vocab: Mapping[str, FastTextV2VocabEntry] | None = None
    min_vocab_df: int = 5
    max_vocab_df: int = 30000
    min_tag_df: int = 1
    min_tag_ratio: float = 0.2
    min_model_code_df: int = 3
    max_event_tag_df: int = 1000
    include_relation_sentences: bool = True
    relation_kinds: Sequence[str] = ("synonym", "near_synonym")
    min_relation_weight: float = 0.68
    max_relation_targets: int = 12
    relation_repeat: int = 2


def _safe_training_token(term: str) -> str:
    return encode_normalized_term(str(term or "").strip())


def _looks_like_model_code(token: str) -> bool:
    return bool(_MODEL_CODE_RE.match(token))


def _entry_allowed(entry: FastTextV2VocabEntry, config: FastTextV2CorpusConfig) -> bool:
    if entry.df < config.min_vocab_df:
        return False
    if config.max_vocab_df > 0 and entry.df > config.max_vocab_df:
        return False
    if _NOISY_PUNCT_RE.search(entry.token):
        return False
    if (
        config.max_event_tag_df > 0
        and "·" in entry.token
        and entry.df >= config.max_event_tag_df
    ):
        return False
    if entry.kind == "tag" and entry.tag_df >= config.min_tag_df:
        return True
    if entry.tag_df >= config.min_tag_df and entry.tag_ratio >= config.min_tag_ratio:
        return True
    if _looks_like_model_code(entry.token) and entry.df >= config.min_model_code_df:
        return True
    return False


def _token_allowed(token: str, config: FastTextV2CorpusConfig) -> bool:
    if not token:
        return False
    if config.vocab is None:
        return True
    entry = config.vocab.get(token)
    if entry is None:
        return False
    return _entry_allowed(entry, config)


def _role_repeats(roles: int, config: FastTextV2CorpusConfig) -> int:
    repeats = 0
    if roles & int(TermRole.TAG):
        repeats += max(0, config.tag_repeat)
    if roles & int(TermRole.TITLE):
        repeats += max(0, config.title_repeat)
    return max(1, repeats)


def _role_marker(roles: int) -> str:
    if roles & int(TermRole.TAG):
        return "__tag__"
    if roles & int(TermRole.TITLE):
        return "__title__"
    return "__term__"


def sentence_from_doc_terms(
    records: Iterable[tuple[str, int, float]],
    config: FastTextV2CorpusConfig | None = None,
) -> list[str]:
    config = config or FastTextV2CorpusConfig()
    ranked = sorted(
        (
            (str(surface), int(roles), float(score))
            for surface, roles, score in records
            if str(surface).strip() and float(score) >= config.min_token_score
        ),
        key=lambda item: (-item[2], -_role_repeats(item[1], config), item[0]),
    )[: config.max_terms_per_doc]
    if len(ranked) < config.min_terms_per_doc:
        return []

    tokens: list[str] = []
    if config.add_doc_anchor:
        tokens.append("__video__")
    for surface, roles, _score in ranked:
        token = _safe_training_token(surface)
        if not _token_allowed(token, config):
            continue
        if config.add_role_tokens:
            tokens.append(_role_marker(roles))
        tokens.extend([token] * _role_repeats(roles, config))
    payload_size = len(tokens) - (1 if config.add_doc_anchor else 0)
    return tokens if payload_size >= config.min_terms_per_doc else []


def iter_doc_segment_paths(version_root: Path | str):
    yield from sorted(Path(version_root).glob("group_*/segments/docs.seg.*.tsv"))


def iter_semantic_segment_sentences(
    version_root: Path | str,
    config: FastTextV2CorpusConfig | None = None,
):
    config = config or FastTextV2CorpusConfig()
    emitted = 0
    for path in iter_doc_segment_paths(version_root):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                parsed = parse_doc_term_row(line)
                if parsed is None:
                    continue
                _doc_key, _content_hash, records = parsed
                sentence = sentence_from_doc_terms(records, config)
                if not sentence:
                    continue
                yield sentence
                emitted += 1
                if config.max_docs > 0 and emitted >= config.max_docs:
                    return


def _iter_relation_rows(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            source = _safe_training_token(parts[0])
            targets: list[tuple[str, float]] = []
            for index in range(1, len(parts) - 1, 2):
                target = _safe_training_token(parts[index])
                try:
                    weight = float(parts[index + 1])
                except ValueError:
                    continue
                targets.append((target, weight))
            yield source, targets


def iter_relation_sentences(
    merged_dir: Path | str,
    config: FastTextV2CorpusConfig | None = None,
):
    config = config or FastTextV2CorpusConfig()
    if not config.include_relation_sentences:
        return
    merged_dir = Path(merged_dir)
    for kind in config.relation_kinds:
        path = merged_dir / f"{kind}.tsv"
        if not path.exists():
            continue
        marker = f"__{kind}__"
        for source, targets in _iter_relation_rows(path):
            if not _token_allowed(source, config):
                continue
            kept = [
                target
                for target, weight in targets
                if weight >= config.min_relation_weight
                and target != source
                and _token_allowed(target, config)
            ][: config.max_relation_targets]
            if not kept:
                continue
            sentence = [marker, source]
            for target in kept:
                sentence.extend([target] * max(1, config.relation_repeat))
            yield sentence


def load_nodes_vocab(
    merged_dir: Path | str,
    *,
    min_df: int = 1,
    max_vocab: int = 0,
) -> list[FastTextV2VocabEntry]:
    path = Path(merged_dir) / "nodes.tsv"
    entries: list[FastTextV2VocabEntry] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            try:
                entry = FastTextV2VocabEntry(
                    token=_safe_training_token(parts[0]),
                    kind=parts[1],
                    df=int(parts[2]),
                    title_df=int(parts[3]),
                    tag_df=int(parts[4]),
                    owner_df=int(parts[5]),
                )
            except ValueError:
                continue
            if entry.df < min_df:
                continue
            entries.append(entry)
            if max_vocab > 0 and len(entries) >= max_vocab:
                break
    return entries


def load_nodes_vocab_map(
    merged_dir: Path | str,
    *,
    min_df: int = 1,
    max_vocab: int = 0,
) -> dict[str, FastTextV2VocabEntry]:
    return {
        entry.token: entry
        for entry in load_nodes_vocab(merged_dir, min_df=min_df, max_vocab=max_vocab)
    }


def write_training_corpus(
    version_root: Path | str,
    output_path: Path | str,
    config: FastTextV2CorpusConfig | None = None,
    *,
    merged_dir: Path | str | None = None,
) -> dict[str, int | str]:
    config = config or FastTextV2CorpusConfig()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    docs = 0
    tokens = 0
    with output_path.open("w", encoding="utf-8") as handle:
        if config.include_doc_sentences:
            for sentence in iter_semantic_segment_sentences(version_root, config):
                handle.write(" ".join(sentence) + "\n")
                docs += 1
                tokens += len(sentence)
        relation_docs = 0
        relation_tokens = 0
        if merged_dir is not None:
            for sentence in iter_relation_sentences(merged_dir, config):
                handle.write(" ".join(sentence) + "\n")
                docs += 1
                tokens += len(sentence)
                relation_docs += 1
                relation_tokens += len(sentence)
    meta = {
        "version_root": str(version_root),
        "merged_dir": str(merged_dir) if merged_dir is not None else "",
        "output_path": str(output_path),
        "docs": docs,
        "tokens": tokens,
        "avg_tokens_per_doc": round(tokens / max(docs, 1), 4),
        "relation_docs": relation_docs,
        "relation_tokens": relation_tokens,
        "vocab_size": len(config.vocab) if config.vocab is not None else 0,
        "config": {
            "min_token_score": config.min_token_score,
            "min_terms_per_doc": config.min_terms_per_doc,
            "max_terms_per_doc": config.max_terms_per_doc,
            "max_docs": config.max_docs,
            "tag_repeat": config.tag_repeat,
            "title_repeat": config.title_repeat,
            "add_role_tokens": config.add_role_tokens,
            "add_doc_anchor": config.add_doc_anchor,
            "include_doc_sentences": config.include_doc_sentences,
            "min_vocab_df": config.min_vocab_df,
            "max_vocab_df": config.max_vocab_df,
            "min_tag_df": config.min_tag_df,
            "min_tag_ratio": config.min_tag_ratio,
            "min_model_code_df": config.min_model_code_df,
            "max_event_tag_df": config.max_event_tag_df,
            "include_relation_sentences": config.include_relation_sentences,
            "relation_kinds": list(config.relation_kinds),
            "min_relation_weight": config.min_relation_weight,
            "max_relation_targets": config.max_relation_targets,
            "relation_repeat": config.relation_repeat,
        },
    }
    output_path.with_suffix(output_path.suffix + ".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return meta
