from __future__ import annotations

import argparse
import json
import time

from dataclasses import asdict, dataclass
from pathlib import Path

from configs.envs import DATA_ROOT
from models.fasttext_v2.corpus import (
    FastTextV2CorpusConfig,
    load_nodes_vocab_map,
    write_training_corpus,
)


DEFAULT_OUTPUT_ROOT = DATA_ROOT / "fasttext_v2"


@dataclass(frozen=True, slots=True)
class FastTextV2TrainConfig:
    vector_size: int = 160
    window: int = 5
    min_count: int = 3
    min_n: int = 3
    max_n: int = 8
    bucket: int = 1000000
    sg: int = 1
    sample: float = 1e-4
    epochs: int = 3
    workers: int = 8
    seed: int = 1


def train_fasttext_model(
    corpus_path: Path | str,
    model_path: Path | str,
    config: FastTextV2TrainConfig | None = None,
) -> dict[str, int | float | str | dict]:
    config = config or FastTextV2TrainConfig()
    try:
        from gensim.models import FastText
        from gensim.models.word2vec import LineSentence
    except Exception as exc:
        raise RuntimeError("gensim is required to train fasttext_v2 models") from exc

    corpus_path = Path(corpus_path)
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    model = FastText(
        vector_size=config.vector_size,
        window=config.window,
        min_count=config.min_count,
        min_n=config.min_n,
        max_n=config.max_n,
        bucket=config.bucket,
        sg=config.sg,
        sample=config.sample,
        workers=config.workers,
        seed=config.seed,
    )
    sentences = LineSentence(str(corpus_path))
    model.build_vocab(corpus_iterable=sentences)
    model.train(
        corpus_iterable=LineSentence(str(corpus_path)),
        total_examples=model.corpus_count,
        epochs=config.epochs,
    )
    model.save(str(model_path))
    model.wv.save(str(model_path.with_suffix(".kv")))
    meta = {
        "corpus_path": str(corpus_path),
        "model_path": str(model_path),
        "kv_path": str(model_path.with_suffix(".kv")),
        "elapsed_sec": round(time.time() - started_at, 3),
        "vocab_size": len(model.wv),
        "corpus_count": int(model.corpus_count),
        "corpus_total_words": int(model.corpus_total_words),
        "train_config": asdict(config),
    }
    model_path.with_suffix(".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return meta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build/train fasttext_v2 from compact semantics TSV segments"
    )
    parser.add_argument("--version-root", type=Path, default=DATA_ROOT / "semantics/v1")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--name", default="fasttext_v2")
    parser.add_argument("--max-docs", type=int, default=0)
    parser.add_argument("--max-terms-per-doc", type=int, default=16)
    parser.add_argument("--min-terms-per-doc", type=int, default=2)
    parser.add_argument("--min-token-score", type=float, default=0.01)
    parser.add_argument("--tag-repeat", type=int, default=2)
    parser.add_argument("--title-repeat", type=int, default=1)
    parser.add_argument("--merged-dir", type=Path, default=None)
    parser.add_argument("--no-vocab-filter", action="store_true")
    parser.add_argument("--min-vocab-df", type=int, default=5)
    parser.add_argument("--max-vocab-df", type=int, default=30000)
    parser.add_argument("--min-tag-df", type=int, default=1)
    parser.add_argument("--min-tag-ratio", type=float, default=0.2)
    parser.add_argument("--min-model-code-df", type=int, default=3)
    parser.add_argument("--max-event-tag-df", type=int, default=1000)
    parser.add_argument("--no-relations", action="store_true")
    parser.add_argument("--relation-kinds", default="synonym,near_synonym")
    parser.add_argument("--min-relation-weight", type=float, default=0.68)
    parser.add_argument("--max-relation-targets", type=int, default=12)
    parser.add_argument("--relation-repeat", type=int, default=2)
    parser.add_argument("--no-role-tokens", action="store_true")
    parser.add_argument("--no-doc-anchor", action="store_true")
    parser.add_argument("--relations-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--vector-size", type=int, default=160)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--min-n", type=int, default=3)
    parser.add_argument("--max-n", type=int, default=8)
    parser.add_argument("--bucket", type=int, default=1000000)
    parser.add_argument("--sg", type=int, default=1)
    parser.add_argument("--sample", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    corpus_path = args.output_root / f"{args.name}.corpus.txt"
    model_path = args.output_root / f"{args.name}.model"
    merged_dir = args.merged_dir or (args.version_root / "merged")
    vocab = None
    if not args.no_vocab_filter and merged_dir.exists():
        vocab = load_nodes_vocab_map(merged_dir, min_df=args.min_vocab_df)
    corpus_config = FastTextV2CorpusConfig(
        min_token_score=args.min_token_score,
        min_terms_per_doc=args.min_terms_per_doc,
        max_terms_per_doc=args.max_terms_per_doc,
        max_docs=args.max_docs,
        tag_repeat=args.tag_repeat,
        title_repeat=args.title_repeat,
        add_role_tokens=not args.no_role_tokens,
        add_doc_anchor=not args.no_doc_anchor,
        include_doc_sentences=not args.relations_only,
        vocab=vocab,
        min_vocab_df=args.min_vocab_df,
        max_vocab_df=args.max_vocab_df,
        min_tag_df=args.min_tag_df,
        min_tag_ratio=args.min_tag_ratio,
        min_model_code_df=args.min_model_code_df,
        max_event_tag_df=args.max_event_tag_df,
        include_relation_sentences=not args.no_relations,
        relation_kinds=tuple(
            item.strip()
            for item in str(args.relation_kinds).split(",")
            if item.strip()
        ),
        min_relation_weight=args.min_relation_weight,
        max_relation_targets=args.max_relation_targets,
        relation_repeat=args.relation_repeat,
    )
    corpus_meta = write_training_corpus(
        args.version_root,
        corpus_path,
        corpus_config,
        merged_dir=merged_dir if merged_dir.exists() else None,
    )
    if args.prepare_only:
        print(json.dumps({"corpus": corpus_meta}, ensure_ascii=False, indent=2))
        return

    train_config = FastTextV2TrainConfig(
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        min_n=args.min_n,
        max_n=args.max_n,
        bucket=args.bucket,
        sg=args.sg,
        sample=args.sample,
        epochs=args.epochs,
        workers=args.workers,
        seed=args.seed,
    )
    train_meta = train_fasttext_model(corpus_path, model_path, train_config)
    print(
        json.dumps(
            {"corpus": corpus_meta, "train": train_meta},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
