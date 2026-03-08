"""Scalable CoreTok training with iterative tuning and holdout retrieval eval."""

import argparse
import json
import math
import random
import statistics
import time

from dataclasses import asdict, dataclass, replace
from pathlib import Path

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
    encoded_profiles = [encode_profile(pipeline, profile) for profile in profiles]
    recall_hits = {1: 0, 5: 0, 10: 0}
    reciprocal_ranks = []
    query_coverage = 0
    empty_queries = 0

    for item in queries:
        query_encoding = encode_query(pipeline, item["query"])
        if not query_encoding["tag_weights"] and not query_encoding["text_weights"]:
            empty_queries += 1
            reciprocal_ranks.append(0.0)
            continue
        query_coverage += 1
        scored = sorted(
            (
                {
                    "mid": profile["mid"],
                    "score": score_query_profile(query_encoding, profile),
                }
                for profile in encoded_profiles
            ),
            key=lambda result: (result["score"], -result["mid"]),
            reverse=True,
        )
        rank = 0
        for index, result in enumerate(scored, start=1):
            if result["mid"] == item["mid"]:
                rank = index
                break
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        for cutoff in recall_hits.keys():
            if rank and rank <= cutoff:
                recall_hits[cutoff] += 1

    total_queries = max(len(queries), 1)
    return {
        "query_count": len(queries),
        "profile_count": len(profiles),
        "query_coverage": round(query_coverage / total_queries, 4),
        "empty_query_ratio": round(empty_queries / total_queries, 4),
        "mrr": round(sum(reciprocal_ranks) / total_queries, 4),
        "recall_at_1": round(recall_hits[1] / total_queries, 4),
        "recall_at_5": round(recall_hits[5] / total_queries, 4),
        "recall_at_10": round(recall_hits[10] / total_queries, 4),
    }


class CoreTokExperimentMonitor:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / "events.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        self.events = []

    def log_event(self, payload: dict):
        event = {"ts": int(time.time()), **payload}
        self.events.append(event)
        with self.events_path.open("a", encoding="utf-8") as wf:
            wf.write(json.dumps(event, ensure_ascii=False) + "\n")

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
    pipeline = CoreTokTrainingPipeline(
        tag_tokenizer=CoreTagTokenizer(),
        text_tokenizer=None,
    )
    pipeline.tag_tokenizer.novelty_threshold = config.tag_novelty
    pipeline.tag_tokenizer.reuse_threshold = config.tag_reuse
    tag_sequences = pipeline.train_stage1(tags, epochs=config.stage1_epochs)
    pipeline.text_tokenizer = CoreTexTokenizer(lexicon=pipeline.tag_tokenizer.lexicon)
    pipeline.text_tokenizer.novelty_threshold = config.text_novelty
    pipeline.text_tokenizer.reuse_threshold = config.text_reuse
    text_sequences = pipeline.train_stage2(texts, epochs=config.stage2_epochs)
    pipeline.train_importance(tag_sequences, text_sequences)
    return pipeline


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


def run_scale(
    scale: ScaleSpec,
    *,
    start_date: str | None,
    end_date: str | None,
    seed: int,
    run_dir: Path,
) -> dict:
    logger.note(f"> CoreTok scale: {scale.name}")
    logger.mesg(dict_to_str(asdict(scale)), indent=2)
    builder = OwnerVideoCorpusBuilder()
    owner_rows = builder.collect_owner_rows(
        max_videos=scale.max_videos,
        max_owners=scale.max_owners,
        min_owner_videos=scale.min_owner_videos,
        start_date=start_date,
        end_date=end_date,
    )
    train_rows, eval_rows = split_owner_rows(
        owner_rows, scale.eval_owner_count, seed=seed
    )
    tags, texts = collect_training_texts(train_rows)
    eval_profiles, eval_queries = build_eval_dataset(
        eval_rows, query_per_owner=scale.query_per_owner
    )

    monitor = CoreTokExperimentMonitor(run_dir / scale.name)
    monitor.log_event(
        {
            "event": "scale_dataset_ready",
            "scale": scale.name,
            "owner_count": len(owner_rows),
            "train_owner_count": len(train_rows),
            "eval_owner_count": len(eval_rows),
            "tag_count": len(tags),
            "text_count": len(texts),
            "eval_profile_count": len(eval_profiles),
            "eval_query_count": len(eval_queries),
        }
    )

    best = None
    iteration_events = []
    for iteration, config in enumerate(default_tuning_configs(scale), start=1):
        logger.note(
            f"  * Iteration {iteration}/{scale.candidate_limit}: {asdict(config)}"
        )
        pipeline = train_pipeline(tags, texts, config)
        metrics = evaluate_owner_retrieval(pipeline, eval_profiles, eval_queries)
        model_stats = summarize_pipeline(pipeline)
        event = {
            "event": "iteration_complete",
            "scale": scale.name,
            "iteration": iteration,
            "config": asdict(config),
            "metrics": metrics,
            "model_stats": model_stats,
        }
        monitor.log_event(event)
        iteration_events.append(event)
        score = (metrics["recall_at_5"], metrics["mrr"], metrics["query_coverage"])
        if best is None or score > (
            best["metrics"]["recall_at_5"],
            best["metrics"]["mrr"],
            best["metrics"]["query_coverage"],
        ):
            best = {**event, "pipeline": pipeline}
            bundle_path = run_dir / scale.name / "best_bundle.json"
            pipeline.save_bundle(bundle_path, bundle_version=f"coretok-{scale.name}")
            monitor.log_event(
                {
                    "event": "best_model_updated",
                    "scale": scale.name,
                    "iteration": iteration,
                    "bundle_path": str(bundle_path),
                    "metrics": metrics,
                }
            )
        monitor.write_summary(
            {
                "scale": scale.name,
                "dataset": {
                    "owner_count": len(owner_rows),
                    "train_owner_count": len(train_rows),
                    "eval_owner_count": len(eval_rows),
                    "tag_count": len(tags),
                    "text_count": len(texts),
                    "eval_profile_count": len(eval_profiles),
                    "eval_query_count": len(eval_queries),
                },
                "best_so_far": {
                    "iteration": best["iteration"] if best else None,
                    "config": best["config"] if best else None,
                    "metrics": best["metrics"] if best else None,
                    "model_stats": best["model_stats"] if best else None,
                },
                "completed_iterations": iteration_events,
            }
        )

    stability = compute_stability(iteration_events)
    summary = {
        "scale": scale.name,
        "dataset": {
            "owner_count": len(owner_rows),
            "train_owner_count": len(train_rows),
            "eval_owner_count": len(eval_rows),
            "tag_count": len(tags),
            "text_count": len(texts),
            "eval_profile_count": len(eval_profiles),
            "eval_query_count": len(eval_queries),
        },
        "best": {
            "iteration": best["iteration"] if best else None,
            "config": best["config"] if best else None,
            "metrics": best["metrics"] if best else None,
            "model_stats": best["model_stats"] if best else None,
        },
        "stability": stability,
    }
    monitor.write_summary(summary)
    logger.success(
        f"  ✓ Scale {scale.name}: best recall@5={summary['best']['metrics']['recall_at_5'] if summary['best']['metrics'] else 0.0}, stable={stability['stable']}"
    )
    return summary


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
    logger.note("> CoreTok training run:")
    logger.mesg(dict_to_str({**vars(args), "run_dir": str(run_dir)}), indent=2)

    scale_summaries = []
    for raw_scale in parse_scales(args.scales):
        scale = apply_scale_overrides(raw_scale, args)
        summary = run_scale(
            scale,
            start_date=args.start_date,
            end_date=args.end_date,
            seed=args.seed,
            run_dir=run_dir,
        )
        scale_summaries.append(summary)
        if not summary["stability"]["stable"] and args.stop_on_unstable:
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
        self.add_argument("--run-name", type=str, default=None)
        self.add_argument("--stop-on-unstable", action="store_true")
        self.add_argument("--max-videos-override", type=int, default=None)
        self.add_argument("--max-owners-override", type=int, default=None)
        self.add_argument("--eval-owner-count-override", type=int, default=None)
        self.add_argument("--candidate-limit-override", type=int, default=None)


if __name__ == "__main__":
    main(CoreTokTrainArgParser().parse_args())
