from __future__ import annotations

import argparse
import json
import shutil
import time

from pathlib import Path
from tempfile import TemporaryDirectory

from models.semantics.pipeline import SemanticPipeline
from models.semantics.storage import merge_groups


def iter_synthetic_docs(count: int, vocab_size: int):
    for index in range(count):
        topic = f"主题{index % vocab_size}"
        neighbor = f"主题{(index + 7) % vocab_size}"
        yield {
            "bvid": f"BV{index:010d}",
            "title": f"{topic} {neighbor} 教程 讲解",
            "tags": [topic, neighbor, "教程"],
            "owner": {"name": f"作者{index % 128}"},
            "tid": index % 32,
        }


def write_vocab(path: Path, vocab_size: int) -> None:
    terms = ["教程", "讲解", "评测", "测评", "采访", "专访"]
    terms.extend(f"主题{index}" for index in range(vocab_size))
    path.write_text("\n".join(terms) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark models.semantics throughput"
    )
    parser.add_argument("--docs", type=int, default=100000)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--num-groups", type=int, default=10)
    parser.add_argument("--group-chunk-size", type=int, default=20000)
    parser.add_argument("--keep-output", type=Path)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()

    with TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        vocab_path = root / "vocabs.txt"
        write_vocab(vocab_path, args.vocab_size)

        output_root = root / "semantics"
        pipeline = SemanticPipeline(
            output_root=output_root,
            version="bench",
            num_groups=args.num_groups,
            workers=args.workers,
            group_chunk_size=args.group_chunk_size,
            vocab_path=vocab_path,
            vocab_limit=0,
        )
        started_at = time.time()
        stats = pipeline.submit_iter(
            iter_synthetic_docs(args.docs, args.vocab_size), log_every=0
        )
        elapsed = max(time.time() - started_at, 0.001)
        stats["elapsed_sec"] = round(elapsed, 3)
        stats["docs_per_sec"] = round(args.docs / elapsed, 2)
        if args.merge:
            merge_started_at = time.time()
            merge_stats = merge_groups(output_root / "bench", min_df=2, min_cooc=2)
            stats["merge_elapsed_sec"] = round(time.time() - merge_started_at, 3)
            stats["merge"] = merge_stats

        if args.keep_output:
            if args.keep_output.exists():
                shutil.rmtree(args.keep_output)
            shutil.copytree(output_root / "bench", args.keep_output)
            stats["kept_output"] = str(args.keep_output)

        print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
