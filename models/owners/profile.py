import argparse
import hashlib
import json
import math
import re
import time

from collections import Counter
from pathlib import Path
from time import perf_counter

from pymongo import ReplaceOne
from sedb import MongoOperator
from tclogger import dict_to_str, logger, str_to_ts

from configs.envs import DATA_ROOT, MONGO_ENVS


OWNER_PROFILE_ROOT = DATA_ROOT / "owners"
OWNER_PROFILE_SNAPSHOTS_PATH = OWNER_PROFILE_ROOT / "owner_profile_snapshots.jsonl"
DEFAULT_FEATURE_BUCKETS = 4096
PROFILE_SAMPLE_TITLES_LIMIT = 6
PROFILE_TOP_TAGS_LIMIT = 16
PROFILE_TOPIC_PHRASES_LIMIT = 12
DEFAULT_PROFILE_TOP_K = 96
OWNER_PROFILE_VERSION = "v2slim1"

DEFAULT_PROFILE_FIELD_WEIGHTS = {
    "owner_name": 3.0,
    "tags": 4.0,
    "title": 2.0,
    "desc": 0.5,
}

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


def hash_feature_unit(unit: str, bucket_count: int = DEFAULT_FEATURE_BUCKETS) -> str:
    digest = hashlib.blake2b(unit.encode("utf-8"), digest_size=8).hexdigest()
    return f"b{int(digest, 16) % bucket_count}"


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
    merged["latest_bvid"] = delta.get("latest_bvid") or base.get("latest_bvid")
    merged["latest_pic"] = delta.get("latest_pic") or base.get("latest_pic")
    merged["primary_tid"] = delta.get("primary_tid") or base.get("primary_tid") or 0
    merged["primary_ptid"] = delta.get("primary_ptid") or base.get("primary_ptid") or 0
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
        self._seen_titles = set()
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

    def build(self) -> dict:
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
            "total_stat_score": round(self.total_stat_score, 4),
            "latest_pubdate": self.latest_pubdate,
            "earliest_pubdate": self.earliest_pubdate,
            "latest_bvid": self.latest_bvid,
            "latest_pic": self.latest_pic,
            "recent_30d_videos": self.recent_30d_videos,
            "recent_7d_videos": self.recent_7d_videos,
            "top_tags": top_tags,
            "sample_titles": self.sample_titles,
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
        self.videos_col = self.db[self.mongo_collection]
        self.output_col = (
            self.db[self.output_collection] if self.output_collection else None
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
        profiles = []
        started_at = perf_counter()
        now_ts = int(time.time())

        def flush_profile(acc: OwnerProfileAccumulator | None):
            nonlocal profile_count, persisted_count, profiles
            if acc is None:
                return False
            profiles.append(acc.build())
            profile_count += 1
            if self.output_col is not None and len(profiles) >= self.mongo_batch_size:
                persisted_count += self.write_profiles_to_mongo(profiles)
                profiles = []
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

        if self.output_col is not None and profiles:
            persisted_count += self.write_profiles_to_mongo(profiles)

        elapsed = perf_counter() - started_at
        summary = {
            "profile_count": profile_count,
            "persisted_count": persisted_count,
            "scanned_videos": scanned_videos,
            "elapsed_s": round(elapsed, 2),
            "video_rate": round(scanned_videos / max(elapsed, 0.001), 2),
            "profile_rate": round(profile_count / max(elapsed, 0.001), 2),
            "output_collection": self.output_collection,
            "profile_version": OWNER_PROFILE_VERSION,
            "feature_buckets": self.feature_buckets,
            "mongo_hint": self.mongo_hint,
            "owner_shard_mode": self.owner_shard_mode,
            "owner_shard_count": self.owner_shard_count,
            "owner_shard_id": self.owner_shard_id,
            "owner_mid_min": self.resolved_owner_mid_min,
            "owner_mid_max": self.resolved_owner_mid_max,
        }
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
