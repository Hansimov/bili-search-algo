from __future__ import annotations

import argparse
import json
import os
import sys
import time

from pathlib import Path

from configs.envs import DATA_ROOT, MONGO_ENVS
from models.semantics.extractor import ExtractionConfig
from models.semantics.pipeline import SemanticPipeline
from models.semantics.storage import decode_term, encode_term, merge_groups
from models.semantics.vocab import DEFAULT_VOCAB_PATH


DEFAULT_OUTPUT_ROOT = DATA_ROOT / "semantics"
DEFAULT_SOURCE_FIELDS = {
    "_id": 0,
    "aid": 1,
    "bvid": 1,
    "title": 1,
    "desc": 1,
    "tags": 1,
    "rtags": 1,
    "owner": 1,
    "tid": 1,
    "pubdate": 1,
    "stat": 1,
}


def iter_jsonl(path: Path, limit: int = 0):
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            yield json.loads(line)
            count += 1
            if limit > 0 and count >= limit:
                break


def _mongo_default(name: str, fallback: str = "") -> str:
    return os.environ.get(name, fallback)


def iter_mongo(args):
    mongo_uri = args.mongo_uri or _mongo_default("SEMANTICS_MONGO_URI")
    mongo_db = args.mongo_db or _mongo_default("SEMANTICS_MONGO_DB", "bili")
    mongo_collection = args.mongo_collection or _mongo_default(
        "SEMANTICS_MONGO_COLLECTION", "videos"
    )
    filters = json.loads(args.filter or "{}")
    if mongo_uri:
        try:
            from pymongo import MongoClient
        except ImportError as exc:
            raise RuntimeError(
                "pymongo is required for --mongo-uri input; omit --mongo-uri to use project MongoOperator"
            ) from exc
        client = MongoClient(mongo_uri)
        collection = client[mongo_db][mongo_collection]
    else:
        from sedb import MongoOperator

        mongo_envs = dict(MONGO_ENVS)
        if args.mongo_db:
            mongo_envs["dbname"] = args.mongo_db
        operator = MongoOperator(
            mongo_envs, connect_msg="from SemanticPipeline", indent=0
        )
        client = getattr(operator, "client", None)
        collection = operator.db[mongo_collection]

    cursor = collection.find(
        filters,
        DEFAULT_SOURCE_FIELDS,
        no_cursor_timeout=True,
    ).batch_size(args.mongo_batch_size)
    count = 0
    try:
        for doc in cursor:
            yield doc
            count += 1
            if args.limit > 0 and count >= args.limit:
                break
    finally:
        cursor.close()
        if client is not None:
            client.close()


def build_command(args) -> None:
    docs = (
        iter_jsonl(args.input_jsonl, args.limit)
        if args.input_jsonl
        else iter_mongo(args)
    )
    started_at = time.time()
    pipeline = SemanticPipeline(
        output_root=args.output_root,
        version=args.version,
        num_groups=args.num_groups,
        workers=args.workers,
        group_chunk_size=args.group_chunk_size,
        vocab_path=args.vocab_path,
        vocab_limit=args.vocab_limit,
        config=ExtractionConfig(
            title_term_limit=args.title_term_limit,
            tag_term_limit=args.tag_term_limit,
            max_terms_per_doc=args.max_terms_per_doc,
            max_edges_per_doc=args.max_pairs_per_doc,
            negative_samples_per_doc=args.negative_samples_per_doc,
        ),
    )
    stats = pipeline.submit_iter(docs, log_every=args.log_every)
    elapsed = max(time.time() - started_at, 0.001)
    stats["elapsed_sec"] = round(elapsed, 3)
    stats["docs_per_sec"] = round(float(stats.get("seen", 0)) / elapsed, 2)
    processing_sec = float(stats.get("processing_sec") or elapsed)
    stats["docs_per_sec_processing"] = round(
        float(stats.get("seen", 0)) / max(processing_sec, 0.001), 2
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def merge_command(args) -> None:
    version_root = args.output_root / args.version
    stats = merge_groups(
        version_root,
        output_dir=args.output_dir,
        min_df=args.min_df,
        min_cooc=args.min_cooc,
        top_k=args.top_k,
        max_df_ratio=args.max_df_ratio,
        max_terms_per_doc=args.max_terms_per_doc,
        max_pairs_per_doc=args.max_pairs_per_doc,
        negative_samples_per_doc=args.negative_samples_per_doc,
        min_score=args.min_score,
        keep_merge_db=args.keep_merge_db,
        embedding_filter_enabled=args.embedding_filter,
        embedding_endpoints=args.embedding_endpoints,
        embedding_min_score=args.embedding_min_score,
        embedding_cjk_min_score=args.embedding_cjk_min_score,
        embedding_mixed_script_min_score=args.embedding_mixed_script_min_score,
        embedding_max_sources=args.embedding_max_sources,
        embedding_max_targets_per_source=args.embedding_max_targets_per_source,
        embedding_filter_near_synonym=args.embedding_filter_near_synonym,
        embedding_filter_doc_cooccurrence=args.embedding_filter_doc_cooccurrence,
        embedding_bridge_promotion_enabled=args.embedding_bridge_promotion,
        embedding_bridge_min_weight=args.embedding_bridge_min_weight,
        embedding_bridge_max_sources=args.embedding_bridge_max_sources,
        embedding_bridge_max_targets_per_source=args.embedding_bridge_max_targets_per_source,
        embedding_bridge_scorer=args.embedding_bridge_scorer,
        embedding_bridge_lsh_bits=args.embedding_bridge_lsh_bits,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def status_command(args) -> None:
    version_root = args.output_root / args.version
    segment_count = len(list(version_root.glob("group_*/segments/docs.seg.*.tsv")))
    cursor_count = len(list(version_root.glob("group_*/processed.sqlite")))
    print(
        json.dumps(
            {
                "version_root": str(version_root),
                "segments": segment_count,
                "cursors": cursor_count,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def inspect_command(args) -> None:
    merged_dir = args.output_root / args.version / "merged"
    term = encode_term(args.term)
    results = []
    for relation in ("rewrite", "synonym", "near_synonym", "doc_cooccurrence"):
        path = merged_dir / f"{relation}.tsv"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if parts and parts[0] == term:
                decoded_row = [
                    decode_term(part) if index == 0 or index % 2 == 1 else part
                    for index, part in enumerate(parts)
                ]
                results.append({"relation": relation, "row": decoded_row})
                break
    print(json.dumps(results, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build compact semantic bundles for es-tok"
    )
    parser.add_argument("--version", default="v1")
    parser.add_argument("--num-groups", type=int, default=10)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--vocab-path", type=Path, default=DEFAULT_VOCAB_PATH)
    parser.add_argument(
        "--vocab-limit",
        type=int,
        default=800000,
        help="Load at most this many entries from vocabs.txt; 0 means all",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--input-jsonl", type=Path)
    build_parser.add_argument("--filter", default="{}")
    build_parser.add_argument("--limit", type=int, default=0)
    build_parser.add_argument("--workers", type=int, default=10)
    build_parser.add_argument("--group-chunk-size", type=int, default=5000)
    build_parser.add_argument("--log-every", type=int, default=500000)
    build_parser.add_argument("--title-term-limit", type=int, default=8)
    build_parser.add_argument("--tag-term-limit", type=int, default=8)
    build_parser.add_argument("--max-terms-per-doc", type=int, default=12)
    build_parser.add_argument("--max-pairs-per-doc", type=int, default=48)
    build_parser.add_argument("--negative-samples-per-doc", type=int, default=4)
    build_parser.add_argument("--mongo-uri")
    build_parser.add_argument("--mongo-db")
    build_parser.add_argument("--mongo-collection")
    build_parser.add_argument("--mongo-batch-size", type=int, default=5000)
    build_parser.set_defaults(func=build_command)

    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--min-df", type=int, default=3)
    merge_parser.add_argument("--min-cooc", type=int, default=2)
    merge_parser.add_argument("--top-k", type=int, default=32)
    merge_parser.add_argument("--max-df-ratio", type=float, default=0.08)
    merge_parser.add_argument("--max-terms-per-doc", type=int, default=12)
    merge_parser.add_argument("--max-pairs-per-doc", type=int, default=48)
    merge_parser.add_argument("--negative-samples-per-doc", type=int, default=0)
    merge_parser.add_argument("--min-score", type=float, default=0.28)
    merge_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Write merged TSV files to this directory instead of <version>/merged",
    )
    merge_parser.add_argument(
        "--keep-merge-db",
        action="store_true",
        help="Keep the intermediate SQLite merge database for debugging",
    )
    merge_parser.add_argument(
        "--embedding-filter",
        action="store_true",
        help="Use TEI similarity to filter generated semantic TSV mappings",
    )
    merge_parser.add_argument(
        "--embedding-endpoints",
        help="Comma-separated or JSON-list TEI endpoints; defaults to SEMANTICS_TEI_ENDPOINTS",
    )
    merge_parser.add_argument("--embedding-min-score", type=float, default=0.52)
    merge_parser.add_argument("--embedding-cjk-min-score", type=float, default=0.58)
    merge_parser.add_argument(
        "--embedding-mixed-script-min-score", type=float, default=0.62
    )
    merge_parser.add_argument("--embedding-max-sources", type=int, default=20000)
    merge_parser.add_argument("--embedding-max-targets-per-source", type=int, default=24)
    merge_parser.add_argument(
        "--no-embedding-filter-near-synonym",
        dest="embedding_filter_near_synonym",
        action="store_false",
        help="Skip full near_synonym TEI filtering; useful when only bridge promotion is needed",
    )
    merge_parser.set_defaults(embedding_filter_near_synonym=True)
    merge_parser.add_argument(
        "--no-embedding-filter-doc-cooccurrence",
        dest="embedding_filter_doc_cooccurrence",
        action="store_false",
        help="Skip full doc_cooccurrence TEI filtering; useful when only bridge promotion is needed",
    )
    merge_parser.set_defaults(embedding_filter_doc_cooccurrence=True)
    merge_parser.add_argument(
        "--embedding-bridge-promotion",
        action="store_true",
        help="Promote high-confidence TEI-verified co-occurrence pairs into synonym.tsv",
    )
    merge_parser.add_argument("--embedding-bridge-min-weight", type=float, default=0.72)
    merge_parser.add_argument("--embedding-bridge-max-sources", type=int, default=1200)
    merge_parser.add_argument(
        "--embedding-bridge-max-targets-per-source", type=int, default=8
    )
    merge_parser.add_argument(
        "--embedding-bridge-scorer",
        choices=["embed", "lsh"],
        default="embed",
        help="Use full embedding dot product or TEI LSH hash similarity for bridge promotion",
    )
    merge_parser.add_argument(
        "--embedding-bridge-lsh-bits",
        type=int,
        default=2048,
        help="Number of LSH bits when --embedding-bridge-scorer=lsh",
    )
    merge_parser.set_defaults(func=merge_command)

    status_parser = subparsers.add_parser("status")
    status_parser.set_defaults(func=status_command)

    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("term")
    inspect_parser.set_defaults(func=inspect_command)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise
