"""Scalable CoreTok training with iterative tuning and holdout retrieval eval."""

import argparse
import json
import math
import os
import random
import shutil
import statistics
import time

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable

from sedb import MongoOperator
from tclogger import dict_to_str, logger, str_to_ts

from configs.envs import DATA_ROOT, MONGO_ENVS
from models.coretok.core import CoreTagTokenizer, CoreTexTokenizer
from models.coretok.pipeline import CoreTokTrainingPipeline


CORETOK_ROOT = DATA_ROOT / "coretok"
CORETOK_RUNS_ROOT = CORETOK_ROOT / "runs"

VIDEO_PROJECTION = {
    "_id": 0,
    "owner.mid": 1,
    "owner.name": 1,
    "title": 1,
    "tags": 1,
    "desc": 1,
    "pubdate": 1,
}


@dataclass
class ScaleSpec:
    name: str
    max_videos: int
    max_owners: int
    min_owner_videos: int
    eval_owner_count: int
    candidate_limit: int
    query_per_owner: int
    stage1_epochs: int
    stage2_epochs: int


@dataclass
class TuningConfig:
    tag_novelty: float
    tag_reuse: float
    text_novelty: float
    text_reuse: float
    stage1_epochs: int
    stage2_epochs: int


DEFAULT_SCALES = {
    "tiny": ScaleSpec(
        name="tiny",
        max_videos=4000,
        max_owners=96,
        min_owner_videos=6,
        eval_owner_count=24,
        candidate_limit=4,
        query_per_owner=3,
        stage1_epochs=1,
        stage2_epochs=1,
    ),
    "small": ScaleSpec(
        name="small",
        max_videos=50000,
        max_owners=800,
        min_owner_videos=6,
        eval_owner_count=160,
        candidate_limit=6,
        query_per_owner=4,
        stage1_epochs=2,
        stage2_epochs=1,
    ),
    "medium": ScaleSpec(
        name="medium",
        max_videos=150000,
        max_owners=2400,
        min_owner_videos=8,
        eval_owner_count=320,
        candidate_limit=8,
        query_per_owner=4,
        stage1_epochs=2,
        stage2_epochs=2,
    ),
}


def default_tuning_configs(scale: ScaleSpec) -> list[TuningConfig]:
    configs = [
        TuningConfig(0.35, 0.40, 0.68, 0.50, scale.stage1_epochs, scale.stage2_epochs),
        TuningConfig(0.40, 0.42, 0.70, 0.52, scale.stage1_epochs, scale.stage2_epochs),
        TuningConfig(0.45, 0.45, 0.72, 0.55, scale.stage1_epochs, scale.stage2_epochs),
        TuningConfig(0.50, 0.48, 0.76, 0.58, scale.stage1_epochs, scale.stage2_epochs),
        TuningConfig(0.55, 0.50, 0.82, 0.60, scale.stage1_epochs, scale.stage2_epochs),
        TuningConfig(
            0.40, 0.50, 0.76, 0.55, scale.stage1_epochs + 1, scale.stage2_epochs
        ),
        TuningConfig(
            0.48, 0.52, 0.74, 0.58, scale.stage1_epochs + 1, scale.stage2_epochs + 1
        ),
    ]
    return configs[: scale.candidate_limit]


def _split_tags(tag_text: str) -> list[str]:
    return [part.strip() for part in (tag_text or "").split(",") if part.strip()]


def _make_support_profile(videos: list[dict]) -> dict:
    tag_counter = {}
    sample_titles = []
    desc_samples = []
    for video in videos:
        for tag in _split_tags(video.get("tags") or ""):
            tag_counter[tag] = tag_counter.get(tag, 0) + 1
        title = (video.get("title") or "").strip()
        desc = (video.get("desc") or "").strip()
        if title and title not in sample_titles and len(sample_titles) < 6:
            sample_titles.append(title)
        if desc and desc not in desc_samples and len(desc_samples) < 3:
            desc_samples.append(desc[:120])
    top_tags = [
        tag
        for tag, _ in sorted(
            tag_counter.items(), key=lambda item: item[1], reverse=True
        )[:16]
    ]
    return {
        "top_tags": top_tags,
        "sample_titles": sample_titles,
        "desc_samples": desc_samples,
    }


def _build_queries(videos: list[dict], query_per_owner: int) -> list[str]:
    queries = []
    seen = set()
    for video in videos:
        candidates = []
        title = (video.get("title") or "").strip()
        desc = (video.get("desc") or "").strip()
        tags = _split_tags(video.get("tags") or "")
        if title:
            candidates.append(title)
        candidates.extend(tags[:2])
        if desc:
            candidates.append(desc[:80])
        for candidate in candidates:
            normalized = candidate.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            queries.append(normalized)
            if len(queries) >= query_per_owner:
                return queries
    return queries


def split_owner_rows(
    owner_rows: list[dict], eval_owner_count: int, seed: int
) -> tuple[list[dict], list[dict]]:
    rows = list(owner_rows)
    rng = random.Random(seed)
    rng.shuffle(rows)
    eval_count = min(eval_owner_count, max(len(rows) // 4, 1))
    eval_rows = rows[:eval_count]
    train_rows = rows[eval_count:]
    return train_rows, eval_rows


def build_eval_dataset(
    owner_rows: list[dict], query_per_owner: int
) -> tuple[list[dict], list[dict]]:
    profiles = []
    queries = []
    for row in owner_rows:
        videos = sorted(
            row.get("videos") or [], key=lambda item: item.get("pubdate") or 0
        )
        if len(videos) < 4:
            continue
        split_at = max(2, math.ceil(len(videos) * 0.6))
        support = videos[:split_at]
        holdout = videos[split_at:]
        if len(holdout) < 1:
            continue
        profile = {
            "mid": row["mid"],
            "name": row.get("name") or "",
            **_make_support_profile(support),
        }
        owner_queries = _build_queries(holdout, query_per_owner=query_per_owner)
        if not owner_queries:
            continue
        profiles.append(profile)
        for query in owner_queries:
            queries.append({"mid": row["mid"], "query": query})
    return profiles, queries


def encode_profile(pipeline: CoreTokTrainingPipeline, profile: dict) -> dict:
    tag_counter = {}
    text_counter = {}
    evaluator = pipeline.importance
    tag_tokenizer = pipeline.tag_tokenizer
    text_tokenizer = pipeline.text_tokenizer

    if tag_tokenizer is not None:
        for tag in profile.get("top_tags") or []:
            token_ids = tag_tokenizer.encode(tag, allow_new_tokens=False)
            for token_id, score in (
                evaluator.score_sequence(token_ids)
                if evaluator and token_ids
                else [
                    (token_id, 1.0 / max(len(token_ids), 1)) for token_id in token_ids
                ]
            ):
                tag_counter[token_id] = tag_counter.get(token_id, 0.0) + float(score)

    if text_tokenizer is not None:
        text_values = [profile.get("name") or ""]
        text_values.extend(profile.get("sample_titles") or [])
        text_values.extend(profile.get("desc_samples") or [])
        for value in text_values:
            token_ids = text_tokenizer.encode(value, allow_new_tokens=False)
            for token_id, score in (
                evaluator.score_sequence(token_ids)
                if evaluator and token_ids
                else [
                    (token_id, 1.0 / max(len(token_ids), 1)) for token_id in token_ids
                ]
            ):
                text_counter[token_id] = text_counter.get(token_id, 0.0) + float(score)

    return {
        "mid": profile["mid"],
        "tag_weights": tag_counter,
        "text_weights": text_counter,
    }


def encode_query(pipeline: CoreTokTrainingPipeline, query: str) -> dict:
    tag_ids = []
    if pipeline.tag_tokenizer is not None:
        tag_ids = pipeline.tag_tokenizer.encode(query, allow_new_tokens=False)
    text_ids = []
    if pipeline.text_tokenizer is not None:
        text_ids = pipeline.text_tokenizer.encode(query, allow_new_tokens=False)
    evaluator = pipeline.importance
    if evaluator is not None:
        tag_weights = dict(evaluator.score_sequence(tag_ids)) if tag_ids else {}
        text_weights = dict(evaluator.score_sequence(text_ids)) if text_ids else {}
    else:
        tag_weights = {
            token_id: round(1.0 / max(len(tag_ids), 1), 4) for token_id in tag_ids
        }
        text_weights = {
            token_id: round(1.0 / max(len(text_ids), 1), 4) for token_id in text_ids
        }
    return {
        "tag_weights": tag_weights,
        "text_weights": text_weights,
    }


def score_query_profile(query_encoding: dict, profile_encoding: dict) -> float:
    score = 0.0
    for token_id, weight in (query_encoding.get("tag_weights") or {}).items():
        score += (
            min(weight, (profile_encoding.get("tag_weights") or {}).get(token_id, 0.0))
            * 1.2
        )
    for token_id, weight in (query_encoding.get("text_weights") or {}).items():
        score += min(
            weight, (profile_encoding.get("text_weights") or {}).get(token_id, 0.0)
        )
    return round(score, 6)


def evaluate_owner_retrieval(
    pipeline: CoreTokTrainingPipeline,
    profiles: list[dict],
    queries: list[dict],
) -> dict:
    started = time.perf_counter()
    stage_started = time.perf_counter()
    encoded_profiles = [encode_profile(pipeline, profile) for profile in profiles]
    encode_profiles_seconds = time.perf_counter() - stage_started

    recall_hits = {1: 0, 5: 0, 10: 0}
    reciprocal_ranks = []
    query_coverage = 0
    empty_queries = 0
    query_cache = {}
    query_cache_hits = 0
    encode_queries_seconds = 0.0
    score_seconds = 0.0

    for item in queries:
        query_text = item["query"]
        query_encoding = query_cache.get(query_text)
        if query_encoding is None:
            stage_started = time.perf_counter()
            query_encoding = encode_query(pipeline, query_text)
            encode_queries_seconds += time.perf_counter() - stage_started
            query_cache[query_text] = query_encoding
        else:
            query_cache_hits += 1

        if not query_encoding["tag_weights"] and not query_encoding["text_weights"]:
            empty_queries += 1
            reciprocal_ranks.append(0.0)
            continue

        query_coverage += 1
        target_mid = item["mid"]
        stage_started = time.perf_counter()
        scored = []
        target_score = None
        for profile in encoded_profiles:
            score = score_query_profile(query_encoding, profile)
            profile_mid = profile["mid"]
            scored.append((profile_mid, score))
            if profile_mid == target_mid:
                target_score = score
        score_seconds += time.perf_counter() - stage_started

        rank = 0
        if target_score is not None:
            better_count = sum(
                1
                for profile_mid, score in scored
                if profile_mid != target_mid
                and (
                    score > target_score
                    or (score == target_score and profile_mid < target_mid)
                )
            )
            rank = better_count + 1

        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        for cutoff in recall_hits.keys():
            if rank and rank <= cutoff:
                recall_hits[cutoff] += 1

    total_queries = max(len(queries), 1)
    total_seconds = time.perf_counter() - started
    return {
        "query_count": len(queries),
        "profile_count": len(profiles),
        "query_coverage": round(query_coverage / total_queries, 4),
        "empty_query_ratio": round(empty_queries / total_queries, 4),
        "mrr": round(sum(reciprocal_ranks) / total_queries, 4),
        "recall_at_1": round(recall_hits[1] / total_queries, 4),
        "recall_at_5": round(recall_hits[5] / total_queries, 4),
        "recall_at_10": round(recall_hits[10] / total_queries, 4),
        "perf": {
            "encode_profiles_seconds": round(encode_profiles_seconds, 4),
            "encode_queries_seconds": round(encode_queries_seconds, 4),
            "score_seconds": round(score_seconds, 4),
            "eval_seconds": round(total_seconds, 4),
            "query_cache_size": len(query_cache),
            "query_cache_hits": query_cache_hits,
            "queries_per_second": (
                round(len(queries) / total_seconds, 4) if total_seconds > 0 else 0.0
            ),
        },
    }


class CoreTokExperimentMonitor:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / "events.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        self.status_path = self.run_dir / "status.json"
        self.events = []

    def log_event(self, payload: dict):
        event = {"ts": int(time.time()), **payload}
        self.events.append(event)
        with self.events_path.open("a", encoding="utf-8") as wf:
            wf.write(json.dumps(event, ensure_ascii=False) + "\n")

    def update_status(self, payload: dict):
        status = {"ts": int(time.time()), **payload}
        self.status_path.write_text(
            json.dumps(status, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def write_summary(self, payload: dict):
        self.summary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


class OwnerVideoCorpusBuilder:
    def __init__(self, db_name: str = "bili", collection: str = "videos"):
        self.mongo = MongoOperator(
            configs=MONGO_ENVS,
            connect_cls=self.__class__,
            verbose_args=False,
        )
        self.collection = self.mongo.client[db_name][collection]

    def collect_owner_rows(
        self,
        *,
        max_videos: int,
        max_owners: int,
        min_owner_videos: int,
        start_date: str | None,
        end_date: str | None,
        progress_interval: int = 0,
        progress_callback: Callable[[dict], None] | None = None,
    ) -> list[dict]:
        query = {"owner.mid": {"$exists": True}, "owner.name": {"$exists": True}}
        pubdate_filter = {}
        if start_date:
            pubdate_filter["$gte"] = str_to_ts(start_date)
        if end_date:
            pubdate_filter["$lte"] = str_to_ts(end_date)
        if pubdate_filter:
            query["pubdate"] = pubdate_filter

        cursor = self.collection.find(query, VIDEO_PROJECTION).sort("owner.mid", 1)
        current_mid = None
        current_name = ""
        current_videos = []
        scanned_videos = 0
        owner_rows = []

        def flush_current():
            nonlocal current_mid, current_name, current_videos, owner_rows
            if current_mid is None:
                return
            if len(current_videos) >= min_owner_videos:
                owner_rows.append(
                    {
                        "mid": current_mid,
                        "name": current_name,
                        "videos": list(current_videos),
                    }
                )
            current_mid = None
            current_name = ""
            current_videos = []

        for doc in cursor:
            owner = doc.get("owner") or {}
            mid = owner.get("mid")
            if mid is None:
                continue
            scanned_videos += 1
            if (
                progress_callback
                and progress_interval
                and scanned_videos % progress_interval == 0
            ):
                progress_callback(
                    {
                        "scanned_videos": scanned_videos,
                        "accepted_owner_count": len(owner_rows),
                        "current_owner_mid": current_mid,
                    }
                )
            if max_videos and scanned_videos > max_videos:
                break
            if current_mid is None:
                current_mid = mid
                current_name = owner.get("name") or ""
            elif mid != current_mid:
                flush_current()
                if max_owners and len(owner_rows) >= max_owners:
                    break
                current_mid = mid
                current_name = owner.get("name") or ""
            current_videos.append(
                {
                    "title": doc.get("title") or "",
                    "tags": doc.get("tags") or "",
                    "desc": doc.get("desc") or "",
                    "pubdate": doc.get("pubdate") or 0,
                }
            )
        if (not max_owners or len(owner_rows) < max_owners) and current_mid is not None:
            flush_current()
        return owner_rows[:max_owners]


def collect_training_texts(owner_rows: list[dict]) -> tuple[list[str], list[str]]:
    tags = []
    texts = []
    for row in owner_rows:
        for video in row.get("videos") or []:
            tags.extend(_split_tags(video.get("tags") or ""))
            title = (video.get("title") or "").strip()
            desc = (video.get("desc") or "").strip()
            if title:
                texts.append(title)
            if desc:
                texts.append(desc[:160])
    return tags, texts


def train_pipeline(
    tags: list[str], texts: list[str], config: TuningConfig
) -> CoreTokTrainingPipeline:
    pipeline, _ = train_pipeline_with_stats(tags, texts, config)
    return pipeline


def train_pipeline_with_stats(
    tags: list[str], texts: list[str], config: TuningConfig
) -> tuple[CoreTokTrainingPipeline, dict]:
    pipeline = CoreTokTrainingPipeline(
        tag_tokenizer=CoreTagTokenizer(),
        text_tokenizer=None,
    )
    pipeline.tag_tokenizer.novelty_threshold = config.tag_novelty
    pipeline.tag_tokenizer.reuse_threshold = config.tag_reuse

    stage_started = time.perf_counter()
    tag_sequences = pipeline.train_stage1(tags, epochs=config.stage1_epochs)
    stage1_seconds = time.perf_counter() - stage_started

    pipeline.text_tokenizer = CoreTexTokenizer(lexicon=pipeline.tag_tokenizer.lexicon)
    pipeline.text_tokenizer.novelty_threshold = config.text_novelty
    pipeline.text_tokenizer.reuse_threshold = config.text_reuse

    stage_started = time.perf_counter()
    text_sequences = pipeline.train_stage2(texts, epochs=config.stage2_epochs)
    stage2_seconds = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    pipeline.train_importance(tag_sequences, text_sequences)
    importance_seconds = time.perf_counter() - stage_started

    timings = {
        "stage1_seconds": round(stage1_seconds, 4),
        "stage2_seconds": round(stage2_seconds, 4),
        "importance_seconds": round(importance_seconds, 4),
        "train_seconds": round(
            stage1_seconds + stage2_seconds + importance_seconds,
            4,
        ),
    }
    return pipeline, timings


def summarize_pipeline(pipeline: CoreTokTrainingPipeline) -> dict:
    lexicon = (
        pipeline.text_tokenizer.lexicon
        if pipeline.text_tokenizer is not None
        else pipeline.tag_tokenizer.lexicon
    )
    tag_stats = pipeline.tag_corpus_stats
    text_stats = pipeline.text_corpus_stats
    return {
        "lexicon_size": len(lexicon.id_to_token),
        "tag_stop_candidate_count": len(tag_stats.stop_candidates) if tag_stats else 0,
        "text_stop_candidate_count": (
            len(text_stats.stop_candidates) if text_stats else 0
        ),
        "tag_docs": tag_stats.total_docs if tag_stats else 0,
        "text_docs": text_stats.total_docs if text_stats else 0,
    }


def compute_stability(events: list[dict]) -> dict:
    if not events:
        return {"stable": False, "std_recall_at_5": None, "best_recall_at_5": 0.0}
    scores = [event["metrics"]["recall_at_5"] for event in events]
    std_score = statistics.pstdev(scores) if len(scores) > 1 else 0.0
    best_score = max(scores)
    return {
        "stable": std_score <= 0.03 and best_score >= 0.2,
        "std_recall_at_5": round(std_score, 4),
        "best_recall_at_5": round(best_score, 4),
    }


def parse_seed_list(seed_list: str | None, default_seed: int) -> list[int]:
    if not seed_list:
        return [default_seed]
    seeds = []
    seen = set()
    for part in seed_list.split(","):
        value = part.strip()
        if not value:
            continue
        seed = int(value)
        if seed in seen:
            continue
        seen.add(seed)
        seeds.append(seed)
    return seeds or [default_seed]


def aggregate_scale_summaries(scale_name: str, summaries: list[dict]) -> dict:
    if not summaries:
        return {
            "scale": scale_name,
            "seed_count": 0,
            "seeds": [],
            "dataset": {},
            "best_config_counts": {},
            "metrics": {},
            "stability": {"stable": False, "std_recall_at_5": None},
            "per_seed": [],
        }

    best_entries = [summary.get("best") or {} for summary in summaries]
    metric_names = [
        "query_coverage",
        "empty_query_ratio",
        "mrr",
        "recall_at_1",
        "recall_at_5",
        "recall_at_10",
    ]

    metric_stats = {}
    for name in metric_names:
        values = [
            entry.get("metrics", {}).get(name)
            for entry in best_entries
            if entry.get("metrics", {}).get(name) is not None
        ]
        if not values:
            continue
        metric_stats[name] = {
            "mean": round(statistics.fmean(values), 4),
            "min": round(min(values), 4),
            "max": round(max(values), 4),
            "std": round(statistics.pstdev(values), 4) if len(values) > 1 else 0.0,
        }

    config_counts = {}
    for entry in best_entries:
        config = entry.get("config")
        if not config:
            continue
        config_key = json.dumps(config, sort_keys=True, ensure_ascii=False)
        config_counts[config_key] = config_counts.get(config_key, 0) + 1

    stable_seed_count = sum(
        1 for summary in summaries if summary.get("stability", {}).get("stable")
    )
    recall_stats = metric_stats.get("recall_at_5", {})
    mean_recall = recall_stats.get("mean", 0.0)
    std_recall = recall_stats.get("std")
    stable_seed_ratio = stable_seed_count / len(summaries)
    aggregate_stable = (
        std_recall is not None
        and std_recall <= 0.08
        and mean_recall >= 0.3
        and stable_seed_ratio >= 0.5
    )

    return {
        "scale": scale_name,
        "seed_count": len(summaries),
        "seeds": [summary.get("seed") for summary in summaries],
        "dataset": {
            "owner_count": max(
                summary["dataset"].get("owner_count", 0) for summary in summaries
            ),
            "train_owner_count_mean": round(
                statistics.fmean(
                    summary["dataset"].get("train_owner_count", 0)
                    for summary in summaries
                ),
                2,
            ),
            "eval_owner_count_mean": round(
                statistics.fmean(
                    summary["dataset"].get("eval_owner_count", 0)
                    for summary in summaries
                ),
                2,
            ),
            "eval_query_count_mean": round(
                statistics.fmean(
                    summary["dataset"].get("eval_query_count", 0)
                    for summary in summaries
                ),
                2,
            ),
        },
        "best_config_counts": config_counts,
        "metrics": metric_stats,
        "stability": {
            "stable": aggregate_stable,
            "stable_seed_ratio": round(stable_seed_ratio, 4),
            "stable_seed_count": stable_seed_count,
            "mean_recall_at_5": round(mean_recall, 4),
            "std_recall_at_5": std_recall,
        },
        "per_seed": summaries,
    }


def detect_system_info() -> dict:
    memory_total_bytes = None
    meminfo_path = Path("/proc/meminfo")
    if meminfo_path.exists():
        for line in meminfo_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                parts = line.split()
                if len(parts) >= 2:
                    memory_total_bytes = int(parts[1]) * 1024
                break
    disk_usage = shutil.disk_usage(CORETOK_RUNS_ROOT)
    return {
        "cpu_count": os.cpu_count() or 1,
        "memory_total_bytes": memory_total_bytes,
        "runs_root_free_bytes": disk_usage.free,
        "runs_root_total_bytes": disk_usage.total,
    }


def owner_rows_cache_path(scale: ScaleSpec, run_dir: Path) -> Path:
    return run_dir / scale.name / "owner_rows.json"


def load_or_collect_owner_rows(
    scale: ScaleSpec,
    *,
    start_date: str | None,
    end_date: str | None,
    run_dir: Path,
    progress_interval: int,
    monitor: CoreTokExperimentMonitor,
) -> tuple[list[dict], dict]:
    cache_path = owner_rows_cache_path(scale, run_dir)
    if cache_path.exists():
        owner_rows = json.loads(cache_path.read_text(encoding="utf-8"))
        stats = {
            "source": "cache",
            "cache_path": str(cache_path),
            "owner_count": len(owner_rows),
            "collection_seconds": 0.0,
        }
        monitor.log_event({"event": "dataset_cache_hit", **stats})
        return owner_rows, stats

    builder = OwnerVideoCorpusBuilder()
    monitor.update_status(
        {
            "phase": "collecting_owner_rows",
            "scale": scale.name,
            "progress": {"scanned_videos": 0, "accepted_owner_count": 0},
        }
    )
    started = time.perf_counter()

    def on_progress(payload: dict):
        monitor.log_event(
            {"event": "dataset_collection_progress", "scale": scale.name, **payload}
        )
        monitor.update_status(
            {
                "phase": "collecting_owner_rows",
                "scale": scale.name,
                "progress": payload,
            }
        )

    owner_rows = builder.collect_owner_rows(
        max_videos=scale.max_videos,
        max_owners=scale.max_owners,
        min_owner_videos=scale.min_owner_videos,
        start_date=start_date,
        end_date=end_date,
        progress_interval=progress_interval,
        progress_callback=on_progress,
    )
    collection_seconds = time.perf_counter() - started
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(owner_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    stats = {
        "source": "mongo",
        "cache_path": str(cache_path),
        "owner_count": len(owner_rows),
        "collection_seconds": round(collection_seconds, 4),
        "owners_per_second": (
            round(len(owner_rows) / collection_seconds, 4)
            if collection_seconds > 0
            else 0.0
        ),
    }
    monitor.log_event(
        {"event": "dataset_collection_complete", "scale": scale.name, **stats}
    )
    return owner_rows, stats


def build_seed_dataset(scale: ScaleSpec, owner_rows: list[dict], seed: int) -> dict:
    train_rows, eval_rows = split_owner_rows(
        owner_rows, scale.eval_owner_count, seed=seed
    )
    tags, texts = collect_training_texts(train_rows)
    eval_profiles, eval_queries = build_eval_dataset(
        eval_rows, query_per_owner=scale.query_per_owner
    )
    return {
        "seed": seed,
        "tags": tags,
        "texts": texts,
        "eval_profiles": eval_profiles,
        "eval_queries": eval_queries,
        "dataset": {
            "owner_count": len(owner_rows),
            "train_owner_count": len(train_rows),
            "eval_owner_count": len(eval_rows),
            "tag_count": len(tags),
            "text_count": len(texts),
            "eval_profile_count": len(eval_profiles),
            "eval_query_count": len(eval_queries),
        },
    }


def summarize_iteration_performance(iteration_events: list[dict]) -> dict:
    if not iteration_events:
        return {"bottleneck_stage": None, "stages": {}}

    stage_names = [
        "stage1_seconds",
        "stage2_seconds",
        "importance_seconds",
        "encode_profiles_seconds",
        "encode_queries_seconds",
        "score_seconds",
        "eval_seconds",
        "train_seconds",
        "total_seconds",
    ]
    stage_stats = {}
    for stage_name in stage_names:
        values = [
            event.get("performance", {}).get(stage_name)
            for event in iteration_events
            if event.get("performance", {}).get(stage_name) is not None
        ]
        if not values:
            continue
        stage_stats[stage_name] = {
            "mean": round(statistics.fmean(values), 4),
            "max": round(max(values), 4),
            "min": round(min(values), 4),
        }

    bottleneck_stage = None
    actionable_stage_names = [
        "stage1_seconds",
        "stage2_seconds",
        "importance_seconds",
        "encode_profiles_seconds",
        "encode_queries_seconds",
        "score_seconds",
    ]
    actionable_stage_stats = {
        name: stage_stats[name]
        for name in actionable_stage_names
        if name in stage_stats
    }
    if actionable_stage_stats:
        bottleneck_stage = max(
            actionable_stage_stats.items(), key=lambda item: item[1]["mean"]
        )[0]
    return {"bottleneck_stage": bottleneck_stage, "stages": stage_stats}


def run_tuning_iteration(
    *,
    scale_name: str,
    seed: int,
    iteration: int,
    config_payload: dict,
    tags: list[str],
    texts: list[str],
    eval_profiles: list[dict],
    eval_queries: list[dict],
    candidate_bundle_path: str,
) -> dict:
    started = time.perf_counter()
    config = TuningConfig(**config_payload)
    pipeline, train_perf = train_pipeline_with_stats(tags, texts, config)
    metrics = evaluate_owner_retrieval(pipeline, eval_profiles, eval_queries)
    model_stats = summarize_pipeline(pipeline)
    pipeline.save_bundle(candidate_bundle_path, bundle_version=f"coretok-{scale_name}")
    performance = {
        **train_perf,
        **(metrics.get("perf") or {}),
        "total_seconds": round(time.perf_counter() - started, 4),
    }
    return {
        "event": "iteration_complete",
        "scale": scale_name,
        "seed": seed,
        "iteration": iteration,
        "config": config_payload,
        "metrics": {key: value for key, value in metrics.items() if key != "perf"},
        "model_stats": model_stats,
        "performance": performance,
        "candidate_bundle_path": candidate_bundle_path,
    }


def resolve_worker_count(max_workers: int | None, task_count: int) -> int:
    if task_count <= 0:
        return 1
    available = os.cpu_count() or 1
    if max_workers is not None:
        available = max(1, min(max_workers, available))
    return max(1, min(available, task_count))


def run_scale(
    scale: ScaleSpec,
    *,
    start_date: str | None,
    end_date: str | None,
    seeds: list[int],
    run_dir: Path,
    max_workers: int | None,
    progress_interval: int,
) -> list[dict]:
    logger.note(f"> CoreTok scale: {scale.name}")
    logger.mesg(dict_to_str(asdict(scale)), indent=2)

    scale_monitor = CoreTokExperimentMonitor(run_dir / scale.name)
    system_info = detect_system_info()
    scale_monitor.log_event(
        {
            "event": "scale_started",
            "scale": scale.name,
            "seeds": seeds,
            "system": system_info,
        }
    )

    owner_rows, collection_stats = load_or_collect_owner_rows(
        scale,
        start_date=start_date,
        end_date=end_date,
        run_dir=run_dir,
        progress_interval=progress_interval,
        monitor=scale_monitor,
    )

    seed_monitors = {}
    seed_states = {}
    tasks = []
    tuning_configs = default_tuning_configs(scale)

    for seed in seeds:
        seed_run_dir = run_dir / f"seed-{seed}" if len(seeds) > 1 else run_dir
        monitor = CoreTokExperimentMonitor(seed_run_dir / scale.name)
        dataset = build_seed_dataset(scale, owner_rows, seed)
        monitor.log_event(
            {
                "event": "scale_dataset_ready",
                "scale": scale.name,
                "seed": seed,
                **dataset["dataset"],
                "dataset_source": collection_stats["source"],
            }
        )
        monitor.update_status(
            {
                "phase": "running_iterations",
                "scale": scale.name,
                "seed": seed,
                "completed_iterations": 0,
                "total_iterations": len(tuning_configs),
                "dataset": dataset["dataset"],
            }
        )
        seed_monitors[seed] = monitor
        seed_states[seed] = {
            "dataset": dataset["dataset"],
            "events": [],
            "best": None,
            "seed_run_dir": seed_run_dir,
        }
        for iteration, config in enumerate(tuning_configs, start=1):
            candidate_bundle_path = (
                seed_run_dir
                / scale.name
                / "candidates"
                / f"iter-{iteration}.bundle.json"
            )
            tasks.append(
                {
                    "scale_name": scale.name,
                    "seed": seed,
                    "iteration": iteration,
                    "config_payload": asdict(config),
                    "tags": dataset["tags"],
                    "texts": dataset["texts"],
                    "eval_profiles": dataset["eval_profiles"],
                    "eval_queries": dataset["eval_queries"],
                    "candidate_bundle_path": str(candidate_bundle_path),
                }
            )

    worker_count = resolve_worker_count(max_workers, len(tasks))
    scale_monitor.log_event(
        {
            "event": "iteration_pool_started",
            "scale": scale.name,
            "task_count": len(tasks),
            "worker_count": worker_count,
            "dataset_collection": collection_stats,
        }
    )
    scale_monitor.update_status(
        {
            "phase": "running_iterations",
            "scale": scale.name,
            "task_count": len(tasks),
            "worker_count": worker_count,
            "completed_tasks": 0,
            "total_tasks": len(tasks),
        }
    )

    completed_tasks = 0
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_to_task = {
            executor.submit(run_tuning_iteration, **task): task for task in tasks
        }
        for future in as_completed(future_to_task):
            event = future.result()
            seed = event["seed"]
            state = seed_states[seed]
            monitor = seed_monitors[seed]
            state["events"].append(event)
            monitor.log_event(event)

            score = (
                event["metrics"]["recall_at_5"],
                event["metrics"]["mrr"],
                event["metrics"]["query_coverage"],
            )
            best = state["best"]
            if best is None or score > (
                best["metrics"]["recall_at_5"],
                best["metrics"]["mrr"],
                best["metrics"]["query_coverage"],
            ):
                state["best"] = event
                best_bundle_path = (
                    state["seed_run_dir"] / scale.name / "best_bundle.json"
                )
                best_bundle_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(event["candidate_bundle_path"], best_bundle_path)
                monitor.log_event(
                    {
                        "event": "best_model_updated",
                        "scale": scale.name,
                        "seed": seed,
                        "iteration": event["iteration"],
                        "bundle_path": str(best_bundle_path),
                        "metrics": event["metrics"],
                        "performance": event["performance"],
                    }
                )

            completed_tasks += 1
            monitor.update_status(
                {
                    "phase": "running_iterations",
                    "scale": scale.name,
                    "seed": seed,
                    "completed_iterations": len(state["events"]),
                    "total_iterations": len(tuning_configs),
                    "last_iteration": event["iteration"],
                    "best_so_far": state["best"]["metrics"] if state["best"] else None,
                    "latest_performance": event["performance"],
                }
            )
            scale_monitor.update_status(
                {
                    "phase": "running_iterations",
                    "scale": scale.name,
                    "task_count": len(tasks),
                    "worker_count": worker_count,
                    "completed_tasks": completed_tasks,
                    "total_tasks": len(tasks),
                }
            )

    summaries = []
    for seed in seeds:
        state = seed_states[seed]
        monitor = seed_monitors[seed]
        iteration_events = sorted(state["events"], key=lambda item: item["iteration"])
        stability = compute_stability(iteration_events)
        performance = summarize_iteration_performance(iteration_events)
        summary = {
            "scale": scale.name,
            "seed": seed,
            "system": system_info,
            "collection": collection_stats,
            "dataset": state["dataset"],
            "best": {
                "iteration": state["best"]["iteration"] if state["best"] else None,
                "config": state["best"]["config"] if state["best"] else None,
                "metrics": state["best"]["metrics"] if state["best"] else None,
                "model_stats": state["best"]["model_stats"] if state["best"] else None,
                "performance": state["best"]["performance"] if state["best"] else None,
            },
            "stability": stability,
            "performance": performance,
        }
        monitor.write_summary(summary)
        monitor.update_status(
            {
                "phase": "completed",
                "scale": scale.name,
                "seed": seed,
                "completed_iterations": len(iteration_events),
                "total_iterations": len(tuning_configs),
                "best": summary["best"],
                "stability": stability,
                "performance": performance,
            }
        )
        summaries.append(summary)
        logger.success(
            f"  ✓ Scale {scale.name} seed={seed}: best recall@5={summary['best']['metrics']['recall_at_5'] if summary['best']['metrics'] else 0.0}, stable={stability['stable']}, bottleneck={performance['bottleneck_stage']}"
        )

    scale_monitor.write_summary(
        {
            "scale": scale.name,
            "system": system_info,
            "collection": collection_stats,
            "seed_count": len(seeds),
            "summaries": summaries,
        }
    )
    scale_monitor.update_status(
        {
            "phase": "completed",
            "scale": scale.name,
            "seed_count": len(seeds),
            "completed_tasks": len(tasks),
            "total_tasks": len(tasks),
            "collection": collection_stats,
        }
    )
    return summaries


def parse_scales(value: str) -> list[ScaleSpec]:
    scale_names = [part.strip() for part in value.split(",") if part.strip()]
    return [DEFAULT_SCALES[name] for name in scale_names]


def apply_scale_overrides(scale: ScaleSpec, args: argparse.Namespace) -> ScaleSpec:
    return replace(
        scale,
        max_videos=args.max_videos_override or scale.max_videos,
        max_owners=args.max_owners_override or scale.max_owners,
        eval_owner_count=args.eval_owner_count_override or scale.eval_owner_count,
        candidate_limit=args.candidate_limit_override or scale.candidate_limit,
    )


def main(args: argparse.Namespace):
    run_name = args.run_name or time.strftime("%Y%m%d-%H%M%S")
    run_dir = CORETOK_RUNS_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seed_list(args.seed_list, args.seed)
    logger.note("> CoreTok training run:")
    logger.mesg(
        dict_to_str({**vars(args), "resolved_seeds": seeds, "run_dir": str(run_dir)}),
        indent=2,
    )

    scale_summaries = []
    for raw_scale in parse_scales(args.scales):
        scale = apply_scale_overrides(raw_scale, args)
        seed_summaries = run_scale(
            scale,
            start_date=args.start_date,
            end_date=args.end_date,
            seeds=seeds,
            run_dir=run_dir,
            max_workers=args.max_workers,
            progress_interval=args.progress_interval,
        )

        if len(seed_summaries) == 1:
            scale_summary = seed_summaries[0]
        else:
            scale_summary = aggregate_scale_summaries(scale.name, seed_summaries)
            aggregate_dir = run_dir / scale.name
            aggregate_dir.mkdir(parents=True, exist_ok=True)
            (aggregate_dir / "seed_sweep_summary.json").write_text(
                json.dumps(scale_summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.success(
                f"  ✓ Scale {scale.name} seed sweep: mean recall@5={scale_summary['stability']['mean_recall_at_5']}, std={scale_summary['stability']['std_recall_at_5']}, stable={scale_summary['stability']['stable']}"
            )

        scale_summaries.append(scale_summary)
        if not scale_summary["stability"]["stable"] and args.stop_on_unstable:
            logger.warn(f"× Stop after unstable scale: {scale.name}")
            break

    (run_dir / "run_summary.json").write_text(
        json.dumps(scale_summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


class CoreTokTrainArgParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()
        self.add_argument("--scales", type=str, default="tiny,small")
        self.add_argument("--start-date", type=str, default=None)
        self.add_argument("--end-date", type=str, default=None)
        self.add_argument("--seed", type=int, default=42)
        self.add_argument("--seed-list", type=str, default=None)
        self.add_argument("--run-name", type=str, default=None)
        self.add_argument("--stop-on-unstable", action="store_true")
        self.add_argument("--max-videos-override", type=int, default=None)
        self.add_argument("--max-owners-override", type=int, default=None)
        self.add_argument("--eval-owner-count-override", type=int, default=None)
        self.add_argument("--candidate-limit-override", type=int, default=None)
        self.add_argument("--max-workers", type=int, default=None)
        self.add_argument("--progress-interval", type=int, default=5000)


if __name__ == "__main__":
    main(CoreTokTrainArgParser().parse_args())
