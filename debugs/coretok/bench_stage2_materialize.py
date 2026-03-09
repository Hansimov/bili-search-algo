import argparse
import json
import sys
import time

from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.coretok.core import (  # noqa: E402
    build_candidate_plan,
    CoreCorpusStats,
    CoreTagTokenizer,
    CoreTexTokenizer,
    normalize_core_text,
)
from models.coretok.train import (
    build_frequency_items,
    collect_training_texts,
)  # noqa: E402


DEFAULT_OWNER_ROWS_PATH = (
    "/home/asimov/repos/bili-search-algo/data/coretok/runs/"
    "perf-small100k-real2000-c1/small/owner_rows.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner-rows-path", type=str, default=DEFAULT_OWNER_ROWS_PATH)
    parser.add_argument("--text-limit", type=int, default=20000)
    parser.add_argument("--tag-limit", type=int, default=30000)
    return parser.parse_args()


def load_owner_rows(path: str) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_ranked_text_plans(
    text_tokenizer: CoreTexTokenizer,
    frequency_items: list[tuple[str, int]],
    candidate_plans: dict[str, dict],
) -> tuple[list[tuple[str, int, list[str]]], Counter]:
    ranked = []
    candidate_counter = Counter()
    for text, freq in frequency_items:
        normalized = normalize_core_text(text)
        ranked_candidates = text_tokenizer._rank_text_candidates(
            normalized,
            candidate_plan=candidate_plans.get(normalized),
        )
        if not ranked_candidates:
            continue
        ranked.append((normalized, int(freq), ranked_candidates))
        for candidate in ranked_candidates:
            candidate_counter[candidate] += int(freq)
    return ranked, candidate_counter


def build_tokenizers(
    owner_rows: list[dict],
    *,
    text_limit: int,
    tag_limit: int,
) -> tuple[CoreTexTokenizer, list[tuple[str, int]], dict[str, dict], Counter, dict]:
    tags, texts = collect_training_texts(owner_rows)
    tag_frequency_items = build_frequency_items(tags)[:tag_limit]
    text_frequency_items = build_frequency_items(texts)[:text_limit]

    tag_counter = Counter(dict(tag_frequency_items))
    text_counter = Counter(dict(text_frequency_items))
    tag_corpus_stats = CoreCorpusStats().fit(for_stage1=True, text_counter=tag_counter)
    text_corpus_stats = CoreCorpusStats().fit(
        for_stage1=False,
        text_counter=text_counter,
    )
    tag_candidate_plans = {
        tag: build_candidate_plan(tag, for_stage1=True, corpus_stats=tag_corpus_stats)
        for tag, _ in tag_frequency_items
    }
    text_candidate_plans = {
        text: build_candidate_plan(
            text,
            for_stage1=False,
            corpus_stats=text_corpus_stats,
        )
        for text, _ in text_frequency_items
    }

    tag_tokenizer = CoreTagTokenizer(corpus_stats=tag_corpus_stats)
    tag_tokenizer.fit(
        epochs=1,
        frequency_items=tag_frequency_items,
        candidate_plans=tag_candidate_plans,
    )
    text_tokenizer = CoreTexTokenizer(
        lexicon=tag_tokenizer.lexicon,
        corpus_stats=text_corpus_stats,
    )
    ranked_plans, candidate_counter = build_ranked_text_plans(
        text_tokenizer,
        text_frequency_items,
        text_candidate_plans,
    )
    stats = {
        "tag_unique_count": len(tag_frequency_items),
        "text_unique_count": len(text_frequency_items),
        "ranked_text_count": len(ranked_plans),
        "unique_candidate_count": len(candidate_counter),
        "candidate_occurrence_count": sum(
            len(candidates) for _, _, candidates in ranked_plans
        ),
    }
    return text_tokenizer, ranked_plans, text_candidate_plans, candidate_counter, stats


def bench_current(
    text_tokenizer: CoreTexTokenizer,
    ranked_plans: list[tuple[str, int, list[str]]],
) -> dict:
    original_find_best_match = text_tokenizer.lexicon.find_best_match
    call_counter = Counter()

    def wrapped_find_best_match(token: str):
        call_counter["find_best_match"] += 1
        return original_find_best_match(token)

    text_tokenizer.lexicon.find_best_match = wrapped_find_best_match
    started = time.perf_counter()
    emitted = 0
    for text, _, ranked_candidates in ranked_plans:
        token_ids = []
        for candidate in ranked_candidates:
            token_id = text_tokenizer._materialize_token(
                candidate,
                allow_new_tokens=True,
                source_text=text,
            )
            if token_id is not None and token_id not in token_ids:
                token_ids.append(token_id)
        emitted += len(token_ids)
    seconds = time.perf_counter() - started
    text_tokenizer.lexicon.find_best_match = original_find_best_match
    return {
        "strategy": "current",
        "seconds": round(seconds, 4),
        "find_best_match_calls": int(call_counter["find_best_match"]),
        "emitted_token_count": emitted,
        "lexicon_size": len(text_tokenizer.lexicon.id_to_token),
    }


def bench_candidate_cache(
    text_tokenizer: CoreTexTokenizer,
    ranked_plans: list[tuple[str, int, list[str]]],
) -> dict:
    original_find_best_match = text_tokenizer.lexicon.find_best_match
    call_counter = Counter()

    def wrapped_find_best_match(token: str):
        call_counter["find_best_match"] += 1
        return original_find_best_match(token)

    text_tokenizer.lexicon.find_best_match = wrapped_find_best_match
    materialize_cache = {}
    started = time.perf_counter()
    emitted = 0
    cache_hits = 0
    for text, _, ranked_candidates in ranked_plans:
        token_ids = []
        for candidate in ranked_candidates:
            token_id = materialize_cache.get(candidate)
            if candidate in materialize_cache:
                cache_hits += 1
                if token_id is not None:
                    text_tokenizer.lexicon.touch_token(token_id)
            else:
                token_id = text_tokenizer._materialize_token(
                    candidate,
                    allow_new_tokens=True,
                    source_text=text,
                )
                materialize_cache[candidate] = token_id
            if token_id is not None and token_id not in token_ids:
                token_ids.append(token_id)
        emitted += len(token_ids)
    seconds = time.perf_counter() - started
    text_tokenizer.lexicon.find_best_match = original_find_best_match
    return {
        "strategy": "candidate_cache",
        "seconds": round(seconds, 4),
        "find_best_match_calls": int(call_counter["find_best_match"]),
        "cache_hits": cache_hits,
        "cache_size": len(materialize_cache),
        "emitted_token_count": emitted,
        "lexicon_size": len(text_tokenizer.lexicon.id_to_token),
    }


def bench_base_match_cache(
    text_tokenizer: CoreTexTokenizer,
    ranked_plans: list[tuple[str, int, list[str]]],
    candidate_counter: Counter,
) -> dict:
    original_find_base_match = text_tokenizer._find_base_match
    call_counter = Counter()

    def wrapped_find_base_match(token: str):
        call_counter["find_base_match"] += 1
        return original_find_base_match(token)

    text_tokenizer._find_base_match = wrapped_find_base_match
    materialize_cache = {}
    started = time.perf_counter()
    emitted = 0
    cache_hits = 0
    for _, _, ranked_candidates in ranked_plans:
        token_ids = []
        for candidate in ranked_candidates:
            cached = materialize_cache.get(candidate)
            if cached is not None:
                cache_hits += 1
                token_id = cached
                if token_id is not None:
                    text_tokenizer.lexicon.touch_token(token_id)
            else:
                token_id, _ = text_tokenizer._materialize_text_candidate(
                    candidate,
                    allow_new_tokens=True,
                    candidate_total_freq=int(candidate_counter.get(candidate, 1)),
                    min_new_token_freq=1,
                )
                materialize_cache[candidate] = token_id
            if token_id is not None and token_id not in token_ids:
                token_ids.append(token_id)
        emitted += len(token_ids)
    seconds = time.perf_counter() - started
    text_tokenizer._find_base_match = original_find_base_match
    return {
        "strategy": "base_match_cache",
        "seconds": round(seconds, 4),
        "find_base_match_calls": int(call_counter["find_base_match"]),
        "cache_hits": cache_hits,
        "cache_size": len(materialize_cache),
        "emitted_token_count": emitted,
        "lexicon_size": len(text_tokenizer.lexicon.id_to_token),
    }


def main():
    args = parse_args()
    owner_rows = load_owner_rows(args.owner_rows_path)

    results = []
    shared_stats = None
    for strategy in ("current", "candidate_cache", "base_match_cache"):
        tokenizer, ranked_plans, _, candidate_counter, stats = build_tokenizers(
            owner_rows,
            text_limit=args.text_limit,
            tag_limit=args.tag_limit,
        )
        shared_stats = stats
        if strategy == "current":
            result = bench_current(tokenizer, ranked_plans)
        elif strategy == "candidate_cache":
            result = bench_candidate_cache(tokenizer, ranked_plans)
        else:
            result = bench_base_match_cache(tokenizer, ranked_plans, candidate_counter)
        results.append(result)

    print(
        json.dumps(
            {
                "owner_rows_path": args.owner_rows_path,
                "text_limit": args.text_limit,
                "tag_limit": args.tag_limit,
                "corpus_stats": shared_stats,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
