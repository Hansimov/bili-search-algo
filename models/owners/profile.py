import argparse
import hashlib
import json
import math
import re
import time

from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Optional

from pymongo import ReplaceOne
from sedb import MongoOperator
from tclogger import dict_to_str, logger, str_to_ts

from configs.envs import DATA_ROOT, MONGO_ENVS


OWNER_PROFILE_ROOT = DATA_ROOT / "owners"
OWNER_PROFILE_SNAPSHOTS_PATH = OWNER_PROFILE_ROOT / "owner_profile_snapshots.jsonl"
ALGO_MONGO_DBNAME = "bili_algo"
DEFAULT_FEATURE_BUCKETS = 4096
PROFILE_SAMPLE_TITLES_LIMIT = 6
PROFILE_TOP_TAGS_LIMIT = 16
PROFILE_TOPIC_PHRASES_LIMIT = 12
PROFILE_DESC_SAMPLES_LIMIT = 3
DEFAULT_PROFILE_TOP_K = 96
OWNER_PROFILE_VERSION = "v2slim1"
OWNER_SEMANTIC_PROFILE_VERSION = "v3idf2"
DEFAULT_SEMANTIC_TOP_TERMS = 24
DEFAULT_VECTOR_BUCKETS = 512
DEFAULT_VECTOR_LIMIT = 48
DEFAULT_SEMANTIC_MIN_DF = 2
DEFAULT_SEMANTIC_MIN_IDF = 0.35
COMMON_LOW_INFO_TERMS = {
    "日常",
    "生活",
    "音乐",
    "视频",
    "记录",
    "分享",
    "内容",
    "合集",
    "搞笑",
}

DEFAULT_PROFILE_FIELD_WEIGHTS = {
    "owner_name": 3.0,
    "tags": 4.0,
    "title": 2.0,
    "desc": 0.5,
}

INFLUENCE_MAX_VIEW = 1e10
INFLUENCE_MAX_VIDEOS = 10000
INFLUENCE_MAX_LIKE = 1e8
INFLUENCE_MAX_COIN = 1e7

INFLUENCE_WEIGHTS = {
    "view": 0.40,
    "scale": 0.20,
    "like": 0.25,
    "coin": 0.15,
}

QUALITY_RANGES = {
    "favorite_rate": {"low": 0.001, "high": 0.03},
    "coin_rate": {"low": 0.0005, "high": 0.015},
    "like_rate": {"low": 0.01, "high": 0.08},
}
QUALITY_WEIGHTS = {
    "favorite_rate": 0.35,
    "coin_rate": 0.25,
    "like_rate": 0.20,
    "stat_quality": 0.20,
}
QUALITY_CONFIDENCE_MIN_VIDEOS = 20
QUALITY_CONFIDENCE_FLOOR = 0.3

ACTIVITY_WEIGHTS = {
    "recency": 0.45,
    "frequency": 0.25,
    "persistence": 0.15,
    "volume": 0.15,
}
ACTIVITY_RECENCY_TAU = 60
ACTIVITY_PERSISTENCE_DAYS = 180
ACTIVITY_MIN_VIDEOS = 5

VIDEO_PROJECTION = {
    "_id": 0,
    "bvid": 1,
    "title": 1,
    "desc": 1,
    "tags": 1,
    "pubdate": 1,
    "tid": 1,
    "ptid": 1,
    "duration": 1,
    "pic": 1,
    "stat.view": 1,
    "stat.like": 1,
    "stat.coin": 1,
    "stat.favorite": 1,
    "stat.danmaku": 1,
    "stat.reply": 1,
    "stat.share": 1,
    "stat_score": 1,
    "owner.mid": 1,
    "owner.name": 1,
}

LATIN_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_\-\.]{1,}")
CJK_SPAN_RE = re.compile(r"[\u4e00-\u9fff]+")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def iter_subword_units(text: str) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []

    units: list[str] = []
    for token in LATIN_TOKEN_RE.findall(text):
        units.append(token)
        for width in (3, 4, 5):
            if len(token) < width:
                continue
            for idx in range(len(token) - width + 1):
                units.append(token[idx : idx + width])

    for span in CJK_SPAN_RE.findall(text):
        span = span.strip()
        if not span:
            continue
        if len(span) <= 4:
            units.append(span)
        for width in (2, 3):
            if len(span) < width:
                continue
            for idx in range(len(span) - width + 1):
                units.append(span[idx : idx + width])
    return units


def iter_surface_units(text: str) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []

    units: list[str] = []
    seen = set()

    for token in LATIN_TOKEN_RE.findall(text):
        if len(token) < 2 or token in seen:
            continue
        seen.add(token)
        units.append(token)

    for span in CJK_SPAN_RE.findall(text):
        span = span.strip()
        if len(span) < 2:
            continue
        if len(span) <= 12:
            if span not in seen:
                seen.add(span)
                units.append(span)
            continue
        for part in re.split(r"[的了和与及并在把将向给对就还也都很再又]", span):
            part = part.strip()
            if len(part) < 2 or len(part) > 12 or part in seen:
                continue
            seen.add(part)
            units.append(part)

    return units


def hash_feature_unit(unit: str, bucket_count: int = DEFAULT_FEATURE_BUCKETS) -> str:
    digest = hashlib.blake2b(unit.encode("utf-8"), digest_size=8).hexdigest()
    return f"b{int(digest, 16) % bucket_count}"


def hash_bucket_id(unit: str, bucket_count: int = DEFAULT_VECTOR_BUCKETS) -> int:
    digest = hashlib.blake2b(unit.encode("utf-8"), digest_size=8).hexdigest()
    return int(digest, 16) % bucket_count


def update_weighted_feature_counter(
    counter: Counter,
    text: str,
    weight: float,
    bucket_count: int = DEFAULT_FEATURE_BUCKETS,
):
    if not text or weight <= 0:
        return
    for unit in iter_subword_units(text):
        counter[hash_feature_unit(unit, bucket_count=bucket_count)] += float(weight)


def update_weighted_term_counter(counter: Counter, text: str, weight: float):
    if not text or weight <= 0:
        return
    for unit in iter_surface_units(text):
        counter[unit] += float(weight)


def merge_sparse_feature_weights(
    base: dict[str, float],
    delta: dict[str, float],
    top_k: int = 128,
) -> dict[str, float]:
    merged = Counter()
    for token, weight in (base or {}).items():
        merged[token] += float(weight)
    for token, weight in (delta or {}).items():
        merged[token] += float(weight)
    return {
        token: round(weight, 4)
        for token, weight in merged.most_common(top_k)
        if weight > 0
    }


def estimate_doc_bytes(doc: dict) -> int:
    return len(
        json.dumps(doc, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )


def log_normalize(value: float, max_val: float) -> float:
    if value <= 0:
        return 0.0
    return min(math.log10(value + 1) / math.log10(max_val + 1), 1.0)


def bounded_normalize(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return max(0.0, min((value - low) / (high - low), 1.0))


def compute_influence(
    total_view: int, total_videos: int, total_like: int, total_coin: int
) -> float:
    view_score = log_normalize(total_view, INFLUENCE_MAX_VIEW)
    scale_score = log_normalize(total_videos, INFLUENCE_MAX_VIDEOS)
    like_score = log_normalize(total_like, INFLUENCE_MAX_LIKE)
    coin_score = log_normalize(total_coin, INFLUENCE_MAX_COIN)
    w = INFLUENCE_WEIGHTS
    return round(
        w["view"] * view_score
        + w["scale"] * scale_score
        + w["like"] * like_score
        + w["coin"] * coin_score,
        4,
    )


def compute_quality(
    avg_favorite_rate: float,
    avg_coin_rate: float,
    avg_like_rate: float,
    avg_stat_score: float,
    total_videos: int,
) -> float:
    fav_score = bounded_normalize(
        avg_favorite_rate,
        QUALITY_RANGES["favorite_rate"]["low"],
        QUALITY_RANGES["favorite_rate"]["high"],
    )
    coin_score = bounded_normalize(
        avg_coin_rate,
        QUALITY_RANGES["coin_rate"]["low"],
        QUALITY_RANGES["coin_rate"]["high"],
    )
    like_score = bounded_normalize(
        avg_like_rate,
        QUALITY_RANGES["like_rate"]["low"],
        QUALITY_RANGES["like_rate"]["high"],
    )
    stat_quality = min(avg_stat_score / 100.0, 1.0)
    confidence = min(total_videos / QUALITY_CONFIDENCE_MIN_VIDEOS, 1.0)
    w = QUALITY_WEIGHTS
    raw_quality = (
        w["favorite_rate"] * fav_score
        + w["coin_rate"] * coin_score
        + w["like_rate"] * like_score
        + w["stat_quality"] * stat_quality
    )
    quality = raw_quality * (
        QUALITY_CONFIDENCE_FLOOR + (1.0 - QUALITY_CONFIDENCE_FLOOR) * confidence
    )
    return round(quality, 4)


def compute_activity(
    days_since_last: float,
    publish_freq: float,
    total_videos: int,
    days_span: float,
) -> float:
    recency = math.exp(-days_since_last / ACTIVITY_RECENCY_TAU)
    freq_score = bounded_normalize(publish_freq, low=1 / 90, high=1.0)
    persistence = min(days_span / ACTIVITY_PERSISTENCE_DAYS, 1.0)
    volume_gate = min(total_videos / ACTIVITY_MIN_VIDEOS, 1.0)
    w = ACTIVITY_WEIGHTS
    return round(
        w["recency"] * recency
        + w["frequency"] * freq_score
        + w["persistence"] * persistence
        + w["volume"] * volume_gate,
        4,
    )


def build_profile_term_counter(
    profile: dict,
    field_weights: dict[str, float] = None,
) -> Counter:
    field_weights = field_weights or DEFAULT_PROFILE_FIELD_WEIGHTS
    counter = Counter()
    update_weighted_term_counter(
        counter,
        profile.get("name") or profile.get("owner_name") or "",
        field_weights.get("owner_name", 0.0),
    )
    for value in profile.get("top_tags") or []:
        update_weighted_term_counter(counter, value, field_weights.get("tags", 0.0))
    for value in profile.get("sample_titles") or []:
        update_weighted_term_counter(counter, value, field_weights.get("title", 0.0))
    for value in profile.get("desc_samples") or []:
        update_weighted_term_counter(counter, value, field_weights.get("desc", 0.0))
    return counter


def compute_profile_idf(
    profiles: list[dict],
    field_weights: dict[str, float] = None,
    min_df: int = DEFAULT_SEMANTIC_MIN_DF,
) -> dict[str, float]:
    doc_freq = Counter()
    total_docs = 0
    for profile in profiles:
        term_counter = build_profile_term_counter(profile, field_weights=field_weights)
        if not term_counter:
            continue
        total_docs += 1
        doc_freq.update(set(term_counter.keys()))
    if total_docs <= 0:
        return {}
    idf = {}
    for term, freq in doc_freq.items():
        if freq < min_df:
            continue
        idf[term] = math.log((1 + total_docs) / (1 + freq)) + 1.0
    return idf


def build_sparse_semantic_vector(
    weighted_terms: dict[str, float],
    bucket_count: int = DEFAULT_VECTOR_BUCKETS,
    vector_limit: int = DEFAULT_VECTOR_LIMIT,
) -> tuple[list[int], list[float]]:
    bucket_counter = Counter()
    for term, weight in weighted_terms.items():
        bucket_counter[hash_bucket_id(term, bucket_count=bucket_count)] += float(weight)
    if not bucket_counter:
        return [], []
    items = bucket_counter.most_common(vector_limit)
    norm = math.sqrt(sum(weight * weight for _, weight in items)) or 1.0
    ids = [int(bucket_id) for bucket_id, _ in items]
    weights = [round(weight / norm, 6) for _, weight in items]
    return ids, weights


def build_semantic_profile(
    raw_profile: dict,
    idf: dict[str, float],
    field_weights: dict[str, float] = None,
    semantic_top_terms: int = DEFAULT_SEMANTIC_TOP_TERMS,
    vector_bucket_count: int = DEFAULT_VECTOR_BUCKETS,
    vector_limit: int = DEFAULT_VECTOR_LIMIT,
    semantic_min_idf: float = DEFAULT_SEMANTIC_MIN_IDF,
) -> dict:
    field_weights = field_weights or DEFAULT_PROFILE_FIELD_WEIGHTS
    term_counter = build_profile_term_counter(raw_profile, field_weights=field_weights)
    weighted_terms = {}
    for term, tf in term_counter.items():
        idf_value = idf.get(term)
        if not idf_value or idf_value < semantic_min_idf:
            continue
        if term in COMMON_LOW_INFO_TERMS:
            continue
        weighted_terms[term] = round(math.log1p(tf) * idf_value, 6)
    top_terms = dict(
        sorted(weighted_terms.items(), key=lambda item: item[1], reverse=True)[
            :semantic_top_terms
        ]
    )
    vector_bucket_ids, vector_bucket_weights = build_sparse_semantic_vector(
        top_terms,
        bucket_count=vector_bucket_count,
        vector_limit=vector_limit,
    )
    domain_parts = []
    domain_parts.extend(raw_profile.get("top_tags") or [])
    domain_parts.extend(raw_profile.get("topic_phrases") or [])
    domain_parts.extend(list(top_terms.keys())[: semantic_top_terms // 2])
    domain_text = " ".join(dict.fromkeys(part for part in domain_parts if part)).strip()
    return {
        "_id": raw_profile["_id"],
        "mid": raw_profile["mid"],
        "name": raw_profile.get("name") or "",
        "total_videos": int(raw_profile.get("total_videos") or 0),
        "total_view": int(raw_profile.get("total_view") or 0),
        "total_like": int(raw_profile.get("total_like") or 0),
        "total_coin": int(raw_profile.get("total_coin") or 0),
        "total_favorite": int(raw_profile.get("total_favorite") or 0),
        "influence_score": float(raw_profile.get("influence_score") or 0.0),
        "quality_score": float(raw_profile.get("quality_score") or 0.0),
        "activity_score": float(raw_profile.get("activity_score") or 0.0),
        "latest_pubdate": int(raw_profile.get("latest_pubdate") or 0),
        "recent_30d_videos": int(raw_profile.get("recent_30d_videos") or 0),
        "recent_7d_videos": int(raw_profile.get("recent_7d_videos") or 0),
        "days_since_last": int(raw_profile.get("days_since_last") or 0),
        "top_tags": list(raw_profile.get("top_tags") or [])[:PROFILE_TOP_TAGS_LIMIT],
        "topic_phrases": list(raw_profile.get("topic_phrases") or [])[
            :PROFILE_TOPIC_PHRASES_LIMIT
        ],
        "domain_text": domain_text,
        "semantic_terms": list(top_terms.keys()),
        "vector_bucket_ids": vector_bucket_ids,
        "vector_bucket_weights": vector_bucket_weights,
        "primary_tid": int(raw_profile.get("primary_tid") or 0),
        "primary_ptid": int(raw_profile.get("primary_ptid") or 0),
        "latest_pic": raw_profile.get("latest_pic") or "",
        "snapshot_at": int(raw_profile.get("snapshot_at") or 0),
        "profile_version": OWNER_SEMANTIC_PROFILE_VERSION,
    }


class OwnerSemanticProfileRefiner:
    def __init__(
        self,
        field_weights: dict[str, float] = None,
        semantic_top_terms: int = DEFAULT_SEMANTIC_TOP_TERMS,
        vector_bucket_count: int = DEFAULT_VECTOR_BUCKETS,
        vector_limit: int = DEFAULT_VECTOR_LIMIT,
        semantic_min_df: int = DEFAULT_SEMANTIC_MIN_DF,
        semantic_min_idf: float = DEFAULT_SEMANTIC_MIN_IDF,
    ):
        self.field_weights = field_weights or DEFAULT_PROFILE_FIELD_WEIGHTS
        self.semantic_top_terms = semantic_top_terms
        self.vector_bucket_count = vector_bucket_count
        self.vector_limit = vector_limit
        self.semantic_min_df = semantic_min_df
        self.semantic_min_idf = semantic_min_idf

    def refine(self, raw_profiles: list[dict]) -> tuple[list[dict], dict]:
        idf = compute_profile_idf(
            raw_profiles,
            field_weights=self.field_weights,
            min_df=self.semantic_min_df,
        )
        refined_profiles = [
            build_semantic_profile(
                profile,
                idf=idf,
                field_weights=self.field_weights,
                semantic_top_terms=self.semantic_top_terms,
                vector_bucket_count=self.vector_bucket_count,
                vector_limit=self.vector_limit,
                semantic_min_idf=self.semantic_min_idf,
            )
            for profile in raw_profiles
        ]
        stats = {
            "profile_count": len(refined_profiles),
            "idf_term_count": len(idf),
            "avg_raw_doc_bytes": round(
                sum(estimate_doc_bytes(doc) for doc in raw_profiles)
                / max(len(raw_profiles), 1),
                2,
            ),
            "avg_refined_doc_bytes": round(
                sum(estimate_doc_bytes(doc) for doc in refined_profiles)
                / max(len(refined_profiles), 1),
                2,
            ),
            "semantic_top_terms": self.semantic_top_terms,
            "vector_bucket_count": self.vector_bucket_count,
            "vector_limit": self.vector_limit,
            "semantic_min_df": self.semantic_min_df,
            "semantic_min_idf": self.semantic_min_idf,
            "size_reduction_ratio": round(
                1
                - (
                    sum(estimate_doc_bytes(doc) for doc in refined_profiles)
                    / max(sum(estimate_doc_bytes(doc) for doc in raw_profiles), 1)
                ),
                4,
            ),
        }
        return refined_profiles, stats


def compute_mode(values: list[int]) -> int:
    if not values:
        return 0
    counter = Counter(value for value in values if value is not None)
    return counter.most_common(1)[0][0] if counter else 0


def merge_unique_lists(*parts: list[str], limit: int) -> list[str]:
    merged = []
    seen = set()
    for part in parts:
        for value in part or []:
            if not value or value in seen:
                continue
            seen.add(value)
            merged.append(value)
            if len(merged) >= limit:
                return merged
    return merged


def build_profile_upsert_filter(profile: dict) -> dict:
    profile_id = profile.get("_id")
    if profile_id is None:
        raise ValueError("profile must contain _id before Mongo upsert")
    return {"_id": profile_id}


def build_owner_shard_expr(
    owner_shard_count: int = 1,
    owner_shard_id: int = 0,
) -> dict | None:
    if owner_shard_count in (None, 0, 1):
        return None
    if owner_shard_count < 1:
        raise ValueError("owner_shard_count must be >= 1")
    if owner_shard_id < 0 or owner_shard_id >= owner_shard_count:
        raise ValueError(f"owner_shard_id must be in [0, {owner_shard_count})")
    return {"$eq": [{"$mod": ["$owner.mid", owner_shard_count]}, owner_shard_id]}


def build_owner_range_query(
    owner_mid_min: int = None,
    owner_mid_max: int = None,
) -> dict | None:
    range_query = {}
    if owner_mid_min is not None:
        range_query["$gte"] = int(owner_mid_min)
    if owner_mid_max is not None:
        range_query["$lte"] = int(owner_mid_max)
    if not range_query:
        return None
    return {"owner.mid": range_query}


def split_owner_mid_range(
    owner_mid_min: int,
    owner_mid_max: int,
    owner_shard_count: int = 1,
    owner_shard_id: int = 0,
) -> tuple[int, int]:
    if owner_mid_min is None or owner_mid_max is None:
        raise ValueError("owner_mid_min and owner_mid_max must both be set")
    if int(owner_mid_max) < int(owner_mid_min):
        raise ValueError("owner_mid_max must be >= owner_mid_min")
    if owner_shard_count < 1:
        raise ValueError("owner_shard_count must be >= 1")
    if owner_shard_id < 0 or owner_shard_id >= owner_shard_count:
        raise ValueError(f"owner_shard_id must be in [0, {owner_shard_count})")

    total_span = int(owner_mid_max) - int(owner_mid_min) + 1
    shard_span = max(math.ceil(total_span / owner_shard_count), 1)
    shard_min = int(owner_mid_min) + shard_span * owner_shard_id
    shard_max = min(int(owner_mid_max), shard_min + shard_span - 1)
    return shard_min, shard_max


def merge_profile_docs(
    base: dict,
    delta: dict,
    top_k: int = DEFAULT_PROFILE_TOP_K,
) -> dict:
    merged = dict(base or {})
    base_features = dict(
        (base or {}).get("feature_weights") or (base or {}).get("token_weights") or {}
    )
    delta_features = dict(
        (delta or {}).get("feature_weights") or (delta or {}).get("token_weights") or {}
    )
    merged["feature_weights"] = merge_sparse_feature_weights(
        base_features,
        delta_features,
        top_k=top_k,
    )
    merged["topic_terms"] = list(merged["feature_weights"].keys())
    merged["mid"] = delta.get("mid", base.get("mid"))
    merged["name"] = delta.get("name") or base.get("name")
    for field in [
        "total_videos",
        "total_view",
        "total_like",
        "total_coin",
        "total_favorite",
        "recent_30d_videos",
        "recent_7d_videos",
    ]:
        merged[field] = int(base.get(field, 0)) + int(delta.get(field, 0))
    merged["total_stat_score"] = float(base.get("total_stat_score", 0.0)) + float(
        delta.get("total_stat_score", 0.0)
    )
    merged["latest_pubdate"] = max(
        int(base.get("latest_pubdate", 0)),
        int(delta.get("latest_pubdate", 0)),
    )
    merged["earliest_pubdate"] = min(
        [
            value
            for value in [
                int(base.get("earliest_pubdate", 0)),
                int(delta.get("earliest_pubdate", 0)),
            ]
            if value > 0
        ]
        or [0]
    )
    merged["sample_titles"] = merge_unique_lists(
        base.get("sample_titles") or [],
        delta.get("sample_titles") or [],
        limit=PROFILE_SAMPLE_TITLES_LIMIT,
    )
    merged["top_tags"] = merge_unique_lists(
        base.get("top_tags") or [],
        delta.get("top_tags") or [],
        limit=PROFILE_TOP_TAGS_LIMIT,
    )
    merged["topic_phrases"] = merge_unique_lists(
        base.get("topic_phrases") or [],
        delta.get("topic_phrases") or [],
        limit=PROFILE_TOPIC_PHRASES_LIMIT,
    )
    merged["desc_samples"] = merge_unique_lists(
        base.get("desc_samples") or [],
        delta.get("desc_samples") or [],
        limit=PROFILE_DESC_SAMPLES_LIMIT,
    )
    merged["latest_bvid"] = delta.get("latest_bvid") or base.get("latest_bvid")
    merged["latest_pic"] = delta.get("latest_pic") or base.get("latest_pic")
    merged["primary_tid"] = delta.get("primary_tid") or base.get("primary_tid") or 0
    merged["primary_ptid"] = delta.get("primary_ptid") or base.get("primary_ptid") or 0
    merged["influence_score"] = max(
        float(base.get("influence_score") or 0.0),
        float(delta.get("influence_score") or 0.0),
    )
    merged["quality_score"] = max(
        float(base.get("quality_score") or 0.0),
        float(delta.get("quality_score") or 0.0),
    )
    merged["activity_score"] = max(
        float(base.get("activity_score") or 0.0),
        float(delta.get("activity_score") or 0.0),
    )
    merged["days_since_last"] = min(
        int(base.get("days_since_last") or 10**9),
        int(delta.get("days_since_last") or 10**9),
    )
    merged["snapshot_at"] = int(
        delta.get("snapshot_at") or base.get("snapshot_at") or 0
    )
    merged["profile_version"] = (
        delta.get("profile_version")
        or base.get("profile_version")
        or OWNER_PROFILE_VERSION
    )
    return merged


class OwnerProfileAccumulator:
    def __init__(
        self,
        doc: dict,
        field_weights: dict[str, float] = None,
        now_ts: int = None,
        top_k: int = DEFAULT_PROFILE_TOP_K,
        feature_buckets: int = DEFAULT_FEATURE_BUCKETS,
    ):
        self.field_weights = field_weights or DEFAULT_PROFILE_FIELD_WEIGHTS
        self.now_ts = now_ts or int(time.time())
        self.top_k = top_k
        self.feature_buckets = feature_buckets

        owner = doc.get("owner") or {}
        self.mid = owner.get("mid")
        self.name = owner.get("name") or ""
        self.total_videos = 0
        self.total_view = 0
        self.total_like = 0
        self.total_coin = 0
        self.total_favorite = 0
        self.total_stat_score = 0.0
        self.latest_pubdate = 0
        self.earliest_pubdate = 0
        self.latest_bvid = ""
        self.latest_pic = ""
        self.recent_30d_videos = 0
        self.recent_7d_videos = 0
        self.tag_counter = Counter()
        self.feature_counter = Counter()
        self.tid_list: list[int] = []
        self.ptid_list: list[int] = []
        self.sample_titles: list[str] = []
        self.desc_samples: list[str] = []
        self._seen_titles = set()
        self._seen_desc = set()
        self.add(doc)

    def add(self, doc: dict):
        owner = doc.get("owner") or {}
        if owner.get("name"):
            self.name = owner["name"]

        pubdate = int(doc.get("pubdate") or 0)
        stat = doc.get("stat") or {}
        tags = (doc.get("tags") or "").strip()
        title = (doc.get("title") or "").strip()
        desc = (doc.get("desc") or "").strip()

        self.total_videos += 1
        self.total_view += int(stat.get("view") or 0)
        self.total_like += int(stat.get("like") or 0)
        self.total_coin += int(stat.get("coin") or 0)
        self.total_favorite += int(stat.get("favorite") or 0)
        self.total_stat_score += float(doc.get("stat_score") or 0.0)

        if pubdate >= self.latest_pubdate:
            self.latest_pubdate = pubdate
            self.latest_bvid = doc.get("bvid") or self.latest_bvid
            self.latest_pic = doc.get("pic") or self.latest_pic
        if self.earliest_pubdate == 0 or (
            pubdate > 0 and pubdate < self.earliest_pubdate
        ):
            self.earliest_pubdate = pubdate

        age_days = max((self.now_ts - pubdate) / 86400, 0) if pubdate else 10**9
        if age_days <= 30:
            self.recent_30d_videos += 1
        if age_days <= 7:
            self.recent_7d_videos += 1

        for tag in tags.split(","):
            tag = tag.strip()
            if tag:
                self.tag_counter[tag] += 1

        tid = doc.get("tid")
        if tid is not None:
            self.tid_list.append(tid)
        ptid = doc.get("ptid")
        if ptid is not None:
            self.ptid_list.append(ptid)

        update_weighted_feature_counter(
            self.feature_counter,
            self.name,
            self.field_weights.get("owner_name", 0.0),
            bucket_count=self.feature_buckets,
        )
        update_weighted_feature_counter(
            self.feature_counter,
            tags,
            self.field_weights.get("tags", 0.0),
            bucket_count=self.feature_buckets,
        )
        update_weighted_feature_counter(
            self.feature_counter,
            title,
            self.field_weights.get("title", 0.0),
            bucket_count=self.feature_buckets,
        )
        update_weighted_feature_counter(
            self.feature_counter,
            desc[:160],
            self.field_weights.get("desc", 0.0),
            bucket_count=self.feature_buckets,
        )

        if (
            title
            and title not in self._seen_titles
            and len(self.sample_titles) < PROFILE_SAMPLE_TITLES_LIMIT
        ):
            self.sample_titles.append(title)
            self._seen_titles.add(title)

        desc_sample = desc[:80]
        if (
            desc_sample
            and desc_sample != "-"
            and desc_sample not in self._seen_desc
            and len(self.desc_samples) < PROFILE_DESC_SAMPLES_LIMIT
        ):
            self.desc_samples.append(desc_sample)
            self._seen_desc.add(desc_sample)

    def build(self) -> dict:
        total_videos = max(self.total_videos, 1)
        total_view = max(self.total_view, 1)
        avg_favorite_rate = self.total_favorite / total_view
        avg_coin_rate = self.total_coin / total_view
        avg_like_rate = self.total_like / total_view
        avg_stat_score = self.total_stat_score / total_videos
        days_span = max((self.latest_pubdate - self.earliest_pubdate) / 86400, 1)
        publish_freq = total_videos / days_span
        days_since_last = max((self.now_ts - self.latest_pubdate) / 86400, 0)
        influence_score = compute_influence(
            self.total_view,
            self.total_videos,
            self.total_like,
            self.total_coin,
        )
        quality_score = compute_quality(
            avg_favorite_rate,
            avg_coin_rate,
            avg_like_rate,
            avg_stat_score,
            self.total_videos,
        )
        activity_score = compute_activity(
            days_since_last,
            publish_freq,
            self.total_videos,
            days_span,
        )
        feature_weights = {
            token: round(weight, 4)
            for token, weight in self.feature_counter.most_common(self.top_k)
            if weight > 0
        }
        top_tags = [
            tag for tag, _ in self.tag_counter.most_common(PROFILE_TOP_TAGS_LIMIT)
        ]
        topic_phrases = merge_unique_lists(
            top_tags[: min(PROFILE_TOPIC_PHRASES_LIMIT, 8)],
            self.sample_titles[:PROFILE_SAMPLE_TITLES_LIMIT],
            limit=PROFILE_TOPIC_PHRASES_LIMIT,
        )
        return {
            "_id": self.mid,
            "mid": self.mid,
            "name": self.name,
            "total_videos": self.total_videos,
            "total_view": self.total_view,
            "total_like": self.total_like,
            "total_coin": self.total_coin,
            "total_favorite": self.total_favorite,
            "influence_score": influence_score,
            "quality_score": quality_score,
            "activity_score": activity_score,
            "days_since_last": int(days_since_last),
            "total_stat_score": round(self.total_stat_score, 4),
            "latest_pubdate": self.latest_pubdate,
            "earliest_pubdate": self.earliest_pubdate,
            "latest_bvid": self.latest_bvid,
            "latest_pic": self.latest_pic,
            "recent_30d_videos": self.recent_30d_videos,
            "recent_7d_videos": self.recent_7d_videos,
            "top_tags": top_tags,
            "sample_titles": self.sample_titles,
            "desc_samples": self.desc_samples,
            "topic_phrases": topic_phrases,
            "topic_terms": list(feature_weights.keys()),
            "feature_weights": feature_weights,
            "primary_tid": compute_mode(self.tid_list),
            "primary_ptid": compute_mode(self.ptid_list),
            "profile_version": OWNER_PROFILE_VERSION,
            "snapshot_at": self.now_ts,
        }


class OwnerProfileBuilder:
    def __init__(
        self,
        mongo_collection: str = "videos",
        output_collection: str = None,
        max_owners: int = None,
        start_pubdate: str = None,
        end_pubdate: str = None,
        max_scanned_videos: int = None,
        allow_full_scan: bool = False,
        field_weights: dict[str, float] = None,
        top_k: int = DEFAULT_PROFILE_TOP_K,
        feature_buckets: int = DEFAULT_FEATURE_BUCKETS,
        output_db_name: str = ALGO_MONGO_DBNAME,
        semantic_refine: bool = True,
        semantic_top_terms: int = DEFAULT_SEMANTIC_TOP_TERMS,
        semantic_vector_buckets: int = DEFAULT_VECTOR_BUCKETS,
        semantic_vector_limit: int = DEFAULT_VECTOR_LIMIT,
        semantic_min_df: int = DEFAULT_SEMANTIC_MIN_DF,
        semantic_min_idf: float = DEFAULT_SEMANTIC_MIN_IDF,
        mongo_batch_size: int = 1000,
        mongo_read_batch_size: int = 5000,
        log_every: int = 200000,
        mongo_hint: str = None,
        owner_shard_mode: str = "mod",
        owner_shard_count: int = 1,
        owner_shard_id: int = 0,
        owner_mid_min: int = None,
        owner_mid_max: int = None,
    ):
        self.mongo_collection = mongo_collection
        self.output_collection = output_collection
        self.max_owners = max_owners
        self.start_pubdate = start_pubdate
        self.end_pubdate = end_pubdate
        self.max_scanned_videos = max_scanned_videos
        self.allow_full_scan = allow_full_scan
        self.field_weights = field_weights or DEFAULT_PROFILE_FIELD_WEIGHTS
        self.top_k = top_k
        self.feature_buckets = feature_buckets
        self.output_db_name = output_db_name
        self.semantic_refine = semantic_refine
        self.semantic_top_terms = semantic_top_terms
        self.semantic_vector_buckets = semantic_vector_buckets
        self.semantic_vector_limit = semantic_vector_limit
        self.semantic_min_df = semantic_min_df
        self.semantic_min_idf = semantic_min_idf
        self.mongo_batch_size = mongo_batch_size
        self.mongo_read_batch_size = mongo_read_batch_size
        self.log_every = log_every
        self.mongo_hint = mongo_hint or (
            "owner.mid_1" if owner_shard_mode == "range" else None
        )
        self.owner_shard_mode = owner_shard_mode
        self.owner_shard_count = owner_shard_count
        self.owner_shard_id = owner_shard_id
        self.owner_mid_min = owner_mid_min
        self.owner_mid_max = owner_mid_max
        self.resolved_owner_mid_min = owner_mid_min
        self.resolved_owner_mid_max = owner_mid_max
        self.init_mongo()

    def init_mongo(self):
        self.mongo = MongoOperator(
            configs=MONGO_ENVS, connect_cls=self.__class__, verbose_args=False
        )
        self.db = self.mongo.client[MONGO_ENVS.get("dbname", "bili")]
        self.output_db = self.mongo.client[self.output_db_name]
        self.videos_col = self.db[self.mongo_collection]
        self.output_col = (
            self.output_db[self.output_collection] if self.output_collection else None
        )

    def validate_budget(self):
        if self.owner_shard_mode not in {"mod", "range"}:
            raise ValueError("owner_shard_mode must be one of: mod, range")
        build_owner_shard_expr(self.owner_shard_count, self.owner_shard_id)
        if self.owner_mid_min is not None and self.owner_mid_max is not None:
            split_owner_mid_range(
                self.owner_mid_min,
                self.owner_mid_max,
                max(self.owner_shard_count, 1),
                min(self.owner_shard_id, max(self.owner_shard_count - 1, 0)),
            )
        if self.allow_full_scan:
            return
        if self.max_owners or self.max_scanned_videos:
            return
        if self.start_pubdate or self.end_pubdate:
            return
        raise ValueError(
            "Refusing owner profile full scan without sampling budget. Set --max-owners, "
            "--max-scanned-videos, a pubdate window, or pass --allow-full-scan explicitly."
        )

    def build_base_query(self) -> dict:
        query = {"owner.mid": {"$exists": True}, "owner.name": {"$exists": True}}
        pubdate_filter = {}
        if self.start_pubdate:
            pubdate_filter["$gte"] = str_to_ts(self.start_pubdate)
        if self.end_pubdate:
            pubdate_filter["$lte"] = str_to_ts(self.end_pubdate)
        if pubdate_filter:
            query["pubdate"] = pubdate_filter
        return query

    def get_owner_mid_bounds(self, query: dict) -> tuple[int | None, int | None]:
        pipeline = [
            {"$match": query},
            {
                "$group": {
                    "_id": None,
                    "min_mid": {"$min": "$owner.mid"},
                    "max_mid": {"$max": "$owner.mid"},
                }
            },
        ]
        result = list(self.videos_col.aggregate(pipeline, allowDiskUse=False))
        if not result:
            return None, None
        item = result[0]
        return item.get("min_mid"), item.get("max_mid")

    def plan_owner_mid_ranges(
        self,
        query: dict,
        bucket_count: int = None,
    ) -> list[dict]:
        bucket_count = bucket_count or self.owner_shard_count or 1
        if bucket_count <= 1:
            min_mid, max_mid = self.get_owner_mid_bounds(query)
            if min_mid is None or max_mid is None:
                return []
            return [
                {
                    "owner_mid_min": int(min_mid),
                    "owner_mid_max": int(max_mid),
                    "owner_count": None,
                }
            ]

        pipeline = [
            {"$match": query},
            {"$group": {"_id": "$owner.mid"}},
            {"$sort": {"_id": 1}},
            {
                "$bucketAuto": {
                    "groupBy": "$_id",
                    "buckets": bucket_count,
                    "output": {
                        "owner_mid_min": {"$min": "$_id"},
                        "owner_mid_max": {"$max": "$_id"},
                        "owner_count": {"$sum": 1},
                    },
                }
            },
        ]
        result = list(self.videos_col.aggregate(pipeline, allowDiskUse=True))
        return [
            {
                "owner_mid_min": int(item.get("owner_mid_min")),
                "owner_mid_max": int(item.get("owner_mid_max")),
                "owner_count": int(item.get("owner_count") or 0),
            }
            for item in result
            if item.get("owner_mid_min") is not None
            and item.get("owner_mid_max") is not None
        ]

    def resolve_owner_mid_range(self, query: dict) -> tuple[int | None, int | None]:
        if self.owner_shard_mode != "range":
            return self.owner_mid_min, self.owner_mid_max

        owner_mid_min = self.owner_mid_min
        owner_mid_max = self.owner_mid_max
        if owner_mid_min is None or owner_mid_max is None:
            if self.owner_shard_count > 1:
                planned_ranges = self.plan_owner_mid_ranges(
                    query,
                    bucket_count=self.owner_shard_count,
                )
                if len(planned_ranges) > self.owner_shard_id:
                    shard_range = planned_ranges[self.owner_shard_id]
                    return (
                        int(shard_range["owner_mid_min"]),
                        int(shard_range["owner_mid_max"]),
                    )
            auto_min, auto_max = self.get_owner_mid_bounds(query)
            owner_mid_min = auto_min if owner_mid_min is None else owner_mid_min
            owner_mid_max = auto_max if owner_mid_max is None else owner_mid_max

        if owner_mid_min is None or owner_mid_max is None:
            return None, None

        if self.owner_shard_count <= 1:
            return int(owner_mid_min), int(owner_mid_max)

        return split_owner_mid_range(
            int(owner_mid_min),
            int(owner_mid_max),
            self.owner_shard_count,
            self.owner_shard_id,
        )

    def build_query(self) -> dict:
        query = self.build_base_query()
        if self.owner_shard_mode == "range":
            owner_mid_min, owner_mid_max = self.resolve_owner_mid_range(query)
            self.resolved_owner_mid_min = owner_mid_min
            self.resolved_owner_mid_max = owner_mid_max
            range_query = build_owner_range_query(owner_mid_min, owner_mid_max)
            if range_query is not None:
                query.update(range_query)
            return query

        shard_expr = build_owner_shard_expr(
            self.owner_shard_count,
            self.owner_shard_id,
        )
        if shard_expr is not None:
            query["$expr"] = shard_expr
        self.resolved_owner_mid_min = self.owner_mid_min
        self.resolved_owner_mid_max = self.owner_mid_max
        return query

    def get_cursor(self):
        self.validate_budget()
        query = self.build_query()
        logger.note("> Owner profile query:")
        logger.mesg(json.dumps(query, ensure_ascii=False, indent=2), indent=2)
        cursor = self.videos_col.find(query, VIDEO_PROJECTION)
        if self.mongo_hint:
            cursor = cursor.hint(self.mongo_hint)
        return cursor.sort("owner.mid", 1).batch_size(self.mongo_read_batch_size)

    def dump_profiles(self, profiles: list[dict], path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as wf:
            for profile in profiles:
                wf.write(json.dumps(profile, ensure_ascii=False) + "\n")
        logger.file(f"  * owner profiles: {path}")

    def write_profiles_to_mongo(self, profiles: list[dict]) -> int:
        if self.output_col is None or not profiles:
            return 0
        operations = [
            ReplaceOne(build_profile_upsert_filter(profile), profile, upsert=True)
            for profile in profiles
        ]
        result = self.output_col.bulk_write(operations, ordered=False)
        return int(
            (result.upserted_count or 0)
            + (result.modified_count or 0)
            + (result.matched_count or 0)
        )

    def build_profiles(self) -> dict:
        cursor = self.get_cursor()
        scanned_videos = 0
        profile_count = 0
        persisted_count = 0
        current_mid = None
        accumulator = None
        raw_profiles = []
        started_at = perf_counter()
        now_ts = int(time.time())

        def flush_profile(acc: OwnerProfileAccumulator | None):
            nonlocal profile_count, raw_profiles
            if acc is None:
                return False
            raw_profiles.append(acc.build())
            profile_count += 1
            return bool(self.max_owners and profile_count >= self.max_owners)

        for doc in cursor:
            scanned_videos += 1
            if self.max_scanned_videos and scanned_videos > self.max_scanned_videos:
                break

            mid = (doc.get("owner") or {}).get("mid")
            if mid is None:
                continue

            if current_mid is None:
                current_mid = mid
                accumulator = OwnerProfileAccumulator(
                    doc,
                    field_weights=self.field_weights,
                    now_ts=now_ts,
                    top_k=self.top_k,
                    feature_buckets=self.feature_buckets,
                )
            elif mid != current_mid:
                should_stop = flush_profile(accumulator)
                if should_stop:
                    break
                current_mid = mid
                accumulator = OwnerProfileAccumulator(
                    doc,
                    field_weights=self.field_weights,
                    now_ts=now_ts,
                    top_k=self.top_k,
                    feature_buckets=self.feature_buckets,
                )
            else:
                accumulator.add(doc)

            if scanned_videos % self.log_every == 0:
                elapsed = perf_counter() - started_at
                rate = scanned_videos / max(elapsed, 0.001)
                logger.note(
                    f"> Owner profile progress: videos={scanned_videos:,}, "
                    f"profiles={profile_count:,}, rate={rate:.0f}/s"
                )

        if accumulator is not None and (
            not self.max_owners or profile_count < self.max_owners
        ):
            flush_profile(accumulator)

        semantic_summary = {}
        refined_profiles = raw_profiles
        if self.semantic_refine:
            refiner = OwnerSemanticProfileRefiner(
                field_weights=self.field_weights,
                semantic_top_terms=self.semantic_top_terms,
                vector_bucket_count=self.semantic_vector_buckets,
                vector_limit=self.semantic_vector_limit,
                semantic_min_df=self.semantic_min_df,
                semantic_min_idf=self.semantic_min_idf,
            )
            refined_profiles, semantic_summary = refiner.refine(raw_profiles)

        if self.output_col is not None and refined_profiles:
            for start_idx in range(0, len(refined_profiles), self.mongo_batch_size):
                persisted_count += self.write_profiles_to_mongo(
                    refined_profiles[start_idx : start_idx + self.mongo_batch_size]
                )

        elapsed = perf_counter() - started_at
        summary = {
            "profile_count": profile_count,
            "raw_profile_count": len(raw_profiles),
            "refined_profile_count": len(refined_profiles),
            "persisted_count": persisted_count,
            "scanned_videos": scanned_videos,
            "elapsed_s": round(elapsed, 2),
            "video_rate": round(scanned_videos / max(elapsed, 0.001), 2),
            "profile_rate": round(profile_count / max(elapsed, 0.001), 2),
            "output_collection": self.output_collection,
            "output_db": self.output_db_name,
            "profile_version": OWNER_PROFILE_VERSION,
            "semantic_profile_version": (
                OWNER_SEMANTIC_PROFILE_VERSION if self.semantic_refine else None
            ),
            "feature_buckets": self.feature_buckets,
            "mongo_hint": self.mongo_hint,
            "owner_shard_mode": self.owner_shard_mode,
            "owner_shard_count": self.owner_shard_count,
            "owner_shard_id": self.owner_shard_id,
            "owner_mid_min": self.resolved_owner_mid_min,
            "owner_mid_max": self.resolved_owner_mid_max,
        }
        summary.update(semantic_summary)
        logger.success(
            f"  ✓ Owner profiles built: {profile_count:,} from {scanned_videos:,} videos in {elapsed:.1f}s"
        )
        logger.note("> Owner profile summary:")
        logger.mesg(dict_to_str(summary), indent=2)
        return summary


class OwnerProfileArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("-c", "--mongo-collection", type=str, default="videos")
        self.add_argument("-m", "--max-owners", type=int, default=None)
        self.add_argument("-s", "--start-pubdate", type=str, default=None)
        self.add_argument("-e", "--end-pubdate", type=str, default=None)
        self.add_argument("--max-scanned-videos", type=int, default=None)
        self.add_argument("--allow-full-scan", action="store_true")
        self.add_argument("--mongo-output-collection", type=str, default=None)
        self.add_argument("--mongo-output-db", type=str, default=ALGO_MONGO_DBNAME)
        self.add_argument(
            "--snapshots-path", type=Path, default=OWNER_PROFILE_SNAPSHOTS_PATH
        )
        self.add_argument("--mongo-batch-size", type=int, default=1000)
        self.add_argument("--mongo-read-batch-size", type=int, default=5000)
        self.add_argument("--mongo-hint", type=str, default=None)
        self.add_argument("--top-k", type=int, default=DEFAULT_PROFILE_TOP_K)
        self.add_argument(
            "--feature-buckets", type=int, default=DEFAULT_FEATURE_BUCKETS
        )
        self.add_argument(
            "--semantic-top-terms", type=int, default=DEFAULT_SEMANTIC_TOP_TERMS
        )
        self.add_argument(
            "--semantic-vector-buckets", type=int, default=DEFAULT_VECTOR_BUCKETS
        )
        self.add_argument(
            "--semantic-vector-limit", type=int, default=DEFAULT_VECTOR_LIMIT
        )
        self.add_argument(
            "--semantic-min-df", type=int, default=DEFAULT_SEMANTIC_MIN_DF
        )
        self.add_argument(
            "--semantic-min-idf", type=float, default=DEFAULT_SEMANTIC_MIN_IDF
        )
        self.add_argument("--disable-semantic-refine", action="store_true")
        self.add_argument("--log-every", type=int, default=200000)
        self.add_argument("--owner-shard-mode", choices=["mod", "range"], default="mod")
        self.add_argument("--owner-shard-count", type=int, default=1)
        self.add_argument("--owner-shard-id", type=int, default=0)
        self.add_argument("--owner-mid-min", type=int, default=None)
        self.add_argument("--owner-mid-max", type=int, default=None)
        self.add_argument("--plan-owner-shards", action="store_true")
        self.add_argument("--build-only", action="store_true")


def main(args: argparse.Namespace):
    logger.note("> Owner profile experiment:")
    logger.mesg(dict_to_str(vars(args)), indent=2)

    builder = OwnerProfileBuilder(
        mongo_collection=args.mongo_collection,
        output_collection=args.mongo_output_collection,
        max_owners=args.max_owners,
        start_pubdate=args.start_pubdate,
        end_pubdate=args.end_pubdate,
        max_scanned_videos=args.max_scanned_videos,
        allow_full_scan=args.allow_full_scan,
        top_k=args.top_k,
        feature_buckets=args.feature_buckets,
        output_db_name=args.mongo_output_db,
        semantic_refine=not args.disable_semantic_refine,
        semantic_top_terms=args.semantic_top_terms,
        semantic_vector_buckets=args.semantic_vector_buckets,
        semantic_vector_limit=args.semantic_vector_limit,
        semantic_min_df=args.semantic_min_df,
        semantic_min_idf=args.semantic_min_idf,
        mongo_batch_size=args.mongo_batch_size,
        mongo_read_batch_size=args.mongo_read_batch_size,
        log_every=args.log_every,
        mongo_hint=args.mongo_hint,
        owner_shard_mode=args.owner_shard_mode,
        owner_shard_count=args.owner_shard_count,
        owner_shard_id=args.owner_shard_id,
        owner_mid_min=args.owner_mid_min,
        owner_mid_max=args.owner_mid_max,
    )
    if args.plan_owner_shards:
        planned_ranges = builder.plan_owner_mid_ranges(builder.build_base_query())
        logger.note("> Planned owner shard ranges:")
        logger.mesg(json.dumps(planned_ranges, ensure_ascii=False, indent=2), indent=2)
        return
    summary = builder.build_profiles()

    if args.build_only:
        return

    if not args.mongo_output_collection:
        logger.note(
            "> Snapshots are not persisted to Mongo; use --mongo-output-collection to enable precompute storage."
        )
    else:
        logger.note(
            "> Mongo precompute collection is ready for ES indexing or downstream model training."
        )
    logger.mesg(dict_to_str(summary), indent=2)


if __name__ == "__main__":
    parsed_args = OwnerProfileArgParser().parse_args()
    main(parsed_args)
