import argparse
import importlib.util
import json
import math
import sys

from collections import Counter
from pathlib import Path

from sedb import MongoOperator

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.envs import MONGO_ENVS
from models.coretok.pipeline import CoreTokTrainingPipeline


DEFAULT_BUNDLE_PATH = (
    "/home/asimov/repos/bili-search-algo/data/coretok/runs/"
    "perf-small-w12-numpy1/seed-7/small/best_bundle.json"
)

VIDEO_PROJECTION = {
    "_id": 0,
    "title": 1,
    "owner.name": 1,
    "tags": 1,
    "tid": 1,
    "pubdate": 1,
    "stat.view": 1,
}

VIEW_BUCKETS = [
    ("view<10", None, 10),
    ("10<=view<100", 10, 100),
    ("100<=view<1k", 100, 1_000),
    ("1k<=view<1w", 1_000, 10_000),
    ("1w<=view<10w", 10_000, 100_000),
    ("10w<=view<100w", 100_000, 1_000_000),
    ("view>=100w", 1_000_000, None),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-path", type=str, default=DEFAULT_BUNDLE_PATH)
    parser.add_argument("--input-path", type=str, default=None)
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--sample-output-path", type=str, default=None)
    parser.add_argument("--sample-size", type=int, default=12)
    parser.add_argument("--max-examples", type=int, default=12)
    parser.add_argument("--include-rows", action="store_true")
    parser.add_argument("--window-count", type=int, default=20)
    parser.add_argument(
        "--sampling-mode",
        type=str,
        default="head",
        choices=["head", "sample", "uniform_windows", "tid_view"],
    )
    return parser.parse_args()


def build_sample_pipeline(bundle_path: str) -> CoreTokTrainingPipeline:
    return CoreTokTrainingPipeline.from_bundle_path(bundle_path)


def write_json(payload: dict | list, output_path: str | None):
    if not output_path:
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_docs(input_path: str) -> list[dict]:
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    return payload.get("docs") or payload.get("rows") or []


def title_length_bucket(title: str) -> str:
    length = len((title or "").strip())
    if length <= 8:
        return "len<=8"
    if length <= 16:
        return "9<=len<=16"
    if length <= 32:
        return "17<=len<=32"
    return "len>32"


def _normalize(value: str) -> str:
    return "".join((value or "").lower().split())


def find_non_substring_tokens(title: str, tokens: list[str]) -> list[str]:
    normalized_title = _normalize(title)
    suspicious = []
    for token in tokens:
        normalized_token = _normalize(token)
        if normalized_token and normalized_token not in normalized_title:
            suspicious.append(token)
    return suspicious


def get_view_value(doc: dict) -> int:
    stat = doc.get("stat") or {}
    try:
        return int(stat.get("view") or 0)
    except (TypeError, ValueError):
        return 0


def get_doc_key(doc: dict) -> tuple:
    return (
        doc.get("tid") or 0,
        get_view_value(doc),
        (doc.get("owner") or {}).get("name") or "",
        doc.get("title") or "",
        doc.get("pubdate") or 0,
    )


def view_bucket(view: int) -> str:
    for label, lower, upper in VIEW_BUCKETS:
        lower_ok = lower is None or view >= lower
        upper_ok = upper is None or view < upper
        if lower_ok and upper_ok:
            return label
    return "view>=100w"


def load_known_tids() -> list[int]:
    region_file = Path(
        "/home/asimov/repos/bili-search/converters/field/region_infos.py"
    )
    if not region_file.exists():
        return []
    spec = importlib.util.spec_from_file_location("region_infos", region_file)
    if spec is None or spec.loader is None:
        return []
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    region_codes = getattr(module, "REGION_CODES", {}) or {}
    tids = []
    for region in region_codes.values():
        children = region.get("children") or {}
        if children:
            tids.extend(
                child.get("tid") for child in children.values() if child.get("tid")
            )
        elif region.get("tid"):
            tids.append(region["tid"])
    return sorted(set(tids))


def sample_random_titles(collection, sample_size: int) -> list[dict]:
    return list(
        collection.aggregate(
            [
                {"$match": {"title": {"$exists": True, "$ne": ""}}},
                {"$sample": {"size": sample_size}},
                {"$project": VIDEO_PROJECTION},
            ]
        )
    )


def sample_head_titles(collection, sample_size: int) -> list[dict]:
    docs = []
    cursor = collection.find({}, VIDEO_PROJECTION).limit(
        max(sample_size * 8, sample_size)
    )
    for doc in cursor:
        title = (doc.get("title") or "").strip()
        if not title:
            continue
        docs.append(doc)
        if len(docs) >= sample_size:
            break
    return docs


def sample_uniform_window_titles(
    collection,
    sample_size: int,
    window_count: int,
) -> list[dict]:
    lower_doc = (
        collection.find(
            {"pubdate": {"$exists": True}},
            {"_id": 0, "pubdate": 1},
        )
        .sort("pubdate", 1)
        .limit(1)
    )
    upper_doc = (
        collection.find(
            {"pubdate": {"$exists": True}},
            {"_id": 0, "pubdate": 1},
        )
        .sort("pubdate", -1)
        .limit(1)
    )
    lower = next(iter(lower_doc), {}).get("pubdate") or 0
    upper = next(iter(upper_doc), {}).get("pubdate") or lower
    if upper <= lower:
        return sample_head_titles(collection, sample_size)

    docs = []
    seen = set()
    window_count = max(1, min(200, window_count))
    span = max(1, upper - lower + 1)
    step = max(1, math.ceil(span / window_count))
    target_per_window = max(1, math.ceil(sample_size / window_count))
    fetch_limit = max(target_per_window * 3, target_per_window)

    for index in range(window_count):
        start = lower + index * step
        end = min(upper + 1, start + step)
        window_query = {
            "title": {"$exists": True, "$ne": ""},
            "pubdate": {"$gte": start, "$lt": end},
        }
        head_docs = list(
            collection.find(window_query, VIDEO_PROJECTION)
            .sort("pubdate", 1)
            .limit(fetch_limit)
        )
        tail_docs = list(
            collection.find(window_query, VIDEO_PROJECTION)
            .sort("pubdate", -1)
            .limit(fetch_limit)
        )
        merged = []
        max_len = max(len(head_docs), len(tail_docs))
        for doc_index in range(max_len):
            if doc_index < len(head_docs):
                merged.append(head_docs[doc_index])
            if doc_index < len(tail_docs):
                merged.append(tail_docs[doc_index])
            if len(merged) >= target_per_window * 2:
                break
        for doc in merged:
            key = get_doc_key(doc)
            if key in seen:
                continue
            seen.add(key)
            docs.append(doc)
            if len(docs) >= sample_size:
                return docs
    return docs[:sample_size]


def sample_tid_view_titles(collection, sample_size: int) -> list[dict]:
    tids = load_known_tids()
    tid_set = set(tids)
    docs = []
    seen = set()
    tid_counts = Counter()
    bucket_counts = Counter()
    tid_bucket_counts = Counter()
    target_per_bucket = max(1, math.ceil(sample_size / max(len(VIEW_BUCKETS), 1)))
    target_per_tid = max(1, math.ceil(sample_size / max(len(tids), 1))) if tids else 8
    max_per_tid_bucket = 2
    max_scanned = max(sample_size * 300, 200_000)
    query = {
        "title": {"$exists": True, "$ne": ""},
        "tid": {"$exists": True},
        "stat.view": {"$exists": True},
    }
    cursor = collection.find(query, VIDEO_PROJECTION).sort("pubdate", -1)

    scanned = 0
    for doc in cursor:
        scanned += 1
        if scanned > max_scanned:
            break
        tid = doc.get("tid")
        if tid_set and tid not in tid_set:
            continue
        key = get_doc_key(doc)
        if key in seen:
            continue
        bucket = view_bucket(get_view_value(doc))
        tid_bucket_key = (tid, bucket)

        should_add = False
        if tid_counts[tid] == 0:
            should_add = True
        elif bucket_counts[bucket] < target_per_bucket:
            should_add = True
        elif tid_bucket_counts[tid_bucket_key] < 1:
            should_add = True
        elif (
            tid_counts[tid] < target_per_tid
            and tid_bucket_counts[tid_bucket_key] < max_per_tid_bucket
        ):
            should_add = True
        elif (
            len(docs) < sample_size // 2
            and tid_bucket_counts[tid_bucket_key] < max_per_tid_bucket
        ):
            should_add = True

        if not should_add:
            continue

        seen.add(key)
        docs.append(doc)
        tid_counts[tid] += 1
        bucket_counts[bucket] += 1
        tid_bucket_counts[tid_bucket_key] += 1
        if len(docs) >= sample_size:
            return docs

    return docs[:sample_size]


def sample_titles(
    sample_size: int, sampling_mode: str, window_count: int
) -> list[dict]:
    mongo = MongoOperator(
        configs=MONGO_ENVS,
        connect_cls=MongoOperator,
        verbose_args=False,
    )
    collection = mongo.client["bili"]["videos"]
    if sampling_mode == "sample":
        return sample_random_titles(collection, sample_size)
    if sampling_mode == "uniform_windows":
        return sample_uniform_window_titles(collection, sample_size, window_count)
    if sampling_mode == "tid_view":
        return sample_tid_view_titles(collection, sample_size)
    return sample_head_titles(collection, sample_size)


def encode_titles(
    pipeline: CoreTokTrainingPipeline,
    docs: list[dict],
    *,
    max_examples: int,
    include_rows: bool,
) -> dict:
    rows = []
    covered = 0
    tag_covered = 0
    text_covered = 0
    bucket_totals = Counter()
    bucket_covered = Counter()
    view_bucket_totals = Counter()
    view_bucket_covered = Counter()
    tid_totals = Counter()
    tid_covered = Counter()
    success_examples = []
    failure_examples = []
    suspicious_examples = []
    suspicious_count = 0

    for doc in docs:
        title = (doc.get("title") or "").strip()
        if not title:
            continue
        tag_ids = pipeline.tag_tokenizer.encode(title, allow_new_tokens=False)
        text_ids = pipeline.text_tokenizer.encode(title, allow_new_tokens=False)
        tag_tokens = pipeline.tag_tokenizer.decode(tag_ids)
        text_tokens = pipeline.text_tokenizer.decode(text_ids)
        current_tid = doc.get("tid")
        current_view = get_view_value(doc)
        current_view_bucket = view_bucket(current_view)
        length_bucket = title_length_bucket(title)

        bucket_totals[length_bucket] += 1
        view_bucket_totals[current_view_bucket] += 1
        tid_totals[current_tid] += 1

        if tag_tokens:
            tag_covered += 1
        if text_tokens:
            text_covered += 1
        if tag_tokens or text_tokens:
            covered += 1
            bucket_covered[length_bucket] += 1
            view_bucket_covered[current_view_bucket] += 1
            tid_covered[current_tid] += 1

        row = {
            "title": title,
            "owner": ((doc.get("owner") or {}).get("name") or ""),
            "tid": current_tid,
            "view": current_view,
            "tags": (doc.get("tags") or "")[:120],
            "tag_tokens": tag_tokens,
            "text_tokens": text_tokens,
        }
        rows.append(row)

        suspicious_tokens = find_non_substring_tokens(title, tag_tokens + text_tokens)
        if suspicious_tokens:
            suspicious_count += 1
        if (tag_tokens or text_tokens) and len(success_examples) < max_examples:
            success_examples.append(row)
        if not tag_tokens and not text_tokens and len(failure_examples) < max_examples:
            failure_examples.append(row)
        if suspicious_tokens and len(suspicious_examples) < max_examples:
            suspicious_examples.append(
                {
                    **row,
                    "suspicious_tokens": suspicious_tokens,
                }
            )

    coverage_by_title_length = {}
    for bucket, total in bucket_totals.items():
        coverage_by_title_length[bucket] = {
            "count": total,
            "covered": bucket_covered.get(bucket, 0),
            "coverage": round(bucket_covered.get(bucket, 0) / max(total, 1), 4),
        }

    coverage_by_view_bucket = {}
    for bucket, total in view_bucket_totals.items():
        coverage_by_view_bucket[bucket] = {
            "count": total,
            "covered": view_bucket_covered.get(bucket, 0),
            "coverage": round(view_bucket_covered.get(bucket, 0) / max(total, 1), 4),
        }

    coverage_by_tid = {}
    for tid, total in tid_totals.most_common():
        coverage_by_tid[str(tid)] = {
            "count": total,
            "covered": tid_covered.get(tid, 0),
            "coverage": round(tid_covered.get(tid, 0) / max(total, 1), 4),
        }

    return {
        "sample_count": len(rows),
        "covered_count": covered,
        "tag_covered_count": tag_covered,
        "text_covered_count": text_covered,
        "coverage": round(covered / max(len(rows), 1), 4),
        "tag_coverage": round(tag_covered / max(len(rows), 1), 4),
        "text_coverage": round(text_covered / max(len(rows), 1), 4),
        "suspicious_count": suspicious_count,
        "suspicious_ratio": round(suspicious_count / max(len(rows), 1), 4),
        "coverage_by_title_length": coverage_by_title_length,
        "coverage_by_view_bucket": coverage_by_view_bucket,
        "coverage_by_tid": coverage_by_tid,
        "success_examples": success_examples,
        "failure_examples": failure_examples,
        "suspicious_examples": suspicious_examples,
        "rows": rows if include_rows else [],
    }


def main():
    args = parse_args()
    pipeline = build_sample_pipeline(args.bundle_path)
    if args.input_path:
        docs = load_docs(args.input_path)
    else:
        docs = sample_titles(
            args.sample_size,
            args.sampling_mode,
            args.window_count,
        )

    write_json(
        {
            "bundle_path": args.bundle_path,
            "sampling_mode": args.sampling_mode,
            "window_count": args.window_count,
            "sample_count": len(docs),
            "docs": docs,
        },
        args.sample_output_path,
    )

    result = {
        "bundle_path": args.bundle_path,
        "sampling_mode": args.sampling_mode,
        "window_count": args.window_count,
        "input_path": args.input_path,
        "sample_output_path": args.sample_output_path,
        **encode_titles(
            pipeline,
            docs,
            max_examples=args.max_examples,
            include_rows=args.include_rows,
        ),
    }
    write_json(result, args.output_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
