import argparse
import json
import time

from pathlib import Path

from models.coretok.train import (
    DEFAULT_SCALES,
    TuningConfig,
    build_seed_dataset,
    train_pipeline_with_stats,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner-rows-path", required=True)
    parser.add_argument("--scale", default="medium")
    parser.add_argument("--owner-limit", type=int, default=250)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tag-novelty", type=float, default=0.35)
    parser.add_argument("--tag-reuse", type=float, default=0.40)
    parser.add_argument("--text-novelty", type=float, default=0.68)
    parser.add_argument("--text-reuse", type=float, default=0.50)
    parser.add_argument("--stage1-epochs", type=int, default=2)
    parser.add_argument("--stage2-epochs", type=int, default=1)
    parser.add_argument("--corpus-workers", type=int, default=None)
    parser.add_argument("--stage2-workers", type=int, default=None)
    parser.add_argument("--aggressive-stage2-materialize", action="store_true")
    args = parser.parse_args()

    owner_rows = json.loads(Path(args.owner_rows_path).read_text(encoding="utf-8"))
    owner_rows = owner_rows[: args.owner_limit]
    scale = DEFAULT_SCALES[args.scale]

    dataset_started = time.perf_counter()
    dataset = build_seed_dataset(
        scale,
        owner_rows,
        seed=args.seed,
        corpus_workers=args.corpus_workers,
    )
    dataset_seconds = time.perf_counter() - dataset_started

    config = TuningConfig(
        tag_novelty=args.tag_novelty,
        tag_reuse=args.tag_reuse,
        text_novelty=args.text_novelty,
        text_reuse=args.text_reuse,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
    )

    _, timings = train_pipeline_with_stats(
        [],
        [],
        config,
        prepared_corpus=dataset["prepared_corpus"],
        aggressive_stage2_materialize=args.aggressive_stage2_materialize,
        stage2_workers=args.stage2_workers,
    )

    output = {
        "owner_limit": args.owner_limit,
        "seed": args.seed,
        "scale": args.scale,
        "dataset_seconds": round(dataset_seconds, 4),
        "prepared_corpus_seconds": round(
            dataset["dataset"].get("prepared_corpus_seconds") or 0.0, 4
        ),
        "tag_count": dataset["dataset"].get("tag_count"),
        "text_count": dataset["dataset"].get("text_count"),
        **timings,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
