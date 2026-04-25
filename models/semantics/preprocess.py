from __future__ import annotations

import hashlib
import json

from typing import Iterable

from models.semantics.vocab import BaseVocab, collapse_text, normalize_surface


def stable_doc_key(doc: dict) -> str:
    bvid = collapse_text(doc.get("bvid"))
    if bvid:
        return f"bvid:{bvid}"
    aid = doc.get("aid") or doc.get("id") or doc.get("_id")
    return f"aid:{collapse_text(aid)}"


def content_hash_of(doc: dict) -> str:
    owner = doc.get("owner") if isinstance(doc.get("owner"), dict) else {}
    payload = {
        "title": collapse_text(doc.get("title")),
        "desc": collapse_text(doc.get("desc")),
        "tags": normalize_tag_list(doc.get("tags")),
        "rtags": normalize_tag_list(doc.get("rtags")),
        "owner": collapse_text(owner.get("name")),
        "tid": doc.get("tid"),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.blake2b(encoded, digest_size=12).hexdigest()


def group_id_of(doc_key: str, num_groups: int) -> int:
    digest = hashlib.blake2b(str(doc_key).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % max(1, num_groups)


def normalize_tag_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items: Iterable[object] = value.replace("#", ",").split(",")
    elif isinstance(value, Iterable):
        raw_items = value
    else:
        raw_items = [value]
    tags: list[str] = []
    for raw_item in raw_items:
        tag = normalize_surface(raw_item)
        if tag:
            tags.append(tag)
    return tags


def dedupe_keep_order(values: Iterable[str], limit: int) -> tuple[str, ...]:
    seen: set[str] = set()
    items: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        items.append(value)
        if len(items) >= limit:
            break
    return tuple(items)


def extract_title_terms(vocab: BaseVocab, doc: dict, limit: int) -> tuple[str, ...]:
    title = collapse_text(doc.get("title"))
    desc = collapse_text(doc.get("desc"))
    candidates = [
        *vocab.iter_vocab_matches(title),
        *vocab.split_terms(title, allow_oov=False),
        *vocab.iter_vocab_matches(desc[:160]),
    ]
    return dedupe_keep_order(candidates, limit)


def extract_tag_terms(vocab: BaseVocab, doc: dict, limit: int) -> tuple[str, ...]:
    candidates: list[str] = []
    for tag in [
        *normalize_tag_list(doc.get("tags")),
        *normalize_tag_list(doc.get("rtags")),
    ]:
        normalized = vocab.normalize_if_valid(tag, allow_oov=True)
        if normalized:
            candidates.append(normalized)
    return dedupe_keep_order(candidates, limit)


def extract_owner_terms(vocab: BaseVocab, doc: dict, limit: int) -> tuple[str, ...]:
    owner = doc.get("owner") if isinstance(doc.get("owner"), dict) else {}
    owner_name = vocab.normalize_if_valid(owner.get("name"), allow_oov=True)
    return dedupe_keep_order([owner_name], limit)
