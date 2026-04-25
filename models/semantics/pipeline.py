from __future__ import annotations

import json
import time

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
from pathlib import Path

from models.semantics.cursor import GroupCursor
from models.semantics.extractor import ExtractionConfig, DocExtractor
from models.semantics.preprocess import content_hash_of, group_id_of, stable_doc_key
from models.semantics.storage import group_dir, write_doc_term_segment
from models.semantics.vocab import BaseVocab, DEFAULT_VOCAB_PATH


_VOCAB_CACHE: dict[tuple[str, int], BaseVocab] = {}


def _get_cached_vocab(vocab_path: str, vocab_limit: int) -> BaseVocab:
    key = (vocab_path, vocab_limit)
    vocab = _VOCAB_CACHE.get(key)
    if vocab is None:
        vocab = BaseVocab(Path(vocab_path), max_vocab_size=vocab_limit)
        _VOCAB_CACHE[key] = vocab
    return vocab


def _process_group_docs(
    *,
    version_root: str,
    group_id: int,
    docs: list[dict],
    vocab_path: str,
    vocab_limit: int,
    config_dict: dict,
) -> dict[str, int]:
    root = Path(version_root)
    cursor_path = group_dir(root, group_id) / "processed.sqlite"
    vocab = _get_cached_vocab(vocab_path, vocab_limit)
    config = ExtractionConfig(**config_dict)
    extractor = DocExtractor(vocab, config)
    extracted_docs = []
    marked_docs: list[tuple[str, str]] = []
    skipped = 0
    with GroupCursor(cursor_path) as cursor:
        for doc in docs:
            doc_key = stable_doc_key(doc)
            if not doc_key or doc_key in {"aid:", "bvid:"}:
                continue
            doc_hash = content_hash_of(doc)
            if cursor.is_unchanged(doc_key, doc_hash):
                skipped += 1
                continue
            marked_docs.append((doc_key, doc_hash))
            extracted = extractor.extract(doc)
            if extracted is None:
                continue
            extracted_docs.append(extracted)
        cursor.mark_many(marked_docs)

    if not extracted_docs:
        return {
            "seen": len(docs),
            "processed": 0,
            "skipped": skipped,
            "doc_rows": 0,
            "doc_terms": 0,
        }

    segment_stats = write_doc_term_segment(root, group_id, extracted_docs)
    return {
        "seen": len(docs),
        "processed": len(extracted_docs),
        "skipped": skipped,
        **segment_stats,
    }


class SemanticPipeline:
    def __init__(
        self,
        *,
        output_root: Path | str,
        version: str = "v1",
        num_groups: int = 10,
        workers: int = 10,
        group_chunk_size: int = 20000,
        vocab_path: Path | str = DEFAULT_VOCAB_PATH,
        config: ExtractionConfig | None = None,
        vocab_limit: int = 800000,
    ):
        self.output_root = Path(output_root)
        self.version = version
        self.version_root = self.output_root / version
        self.num_groups = max(1, num_groups)
        self.workers = max(1, workers)
        self.group_chunk_size = max(1, group_chunk_size)
        self.vocab_path = Path(vocab_path)
        self.vocab_limit = max(0, vocab_limit)
        self.config = config or ExtractionConfig()
        self.version_root.mkdir(parents=True, exist_ok=True)

    def submit_iter(
        self, docs_iter, *, log_every: int = 100000
    ) -> dict[str, int | str]:
        buffers: dict[int, list[dict]] = {
            group_id: [] for group_id in range(self.num_groups)
        }
        stats = {
            "seen": 0,
            "processed": 0,
            "skipped": 0,
            "doc_rows": 0,
            "doc_terms": 0,
        }

        def consume_result(result: dict[str, int]) -> None:
            for key in stats:
                stats[key] += int(result.get(key, 0))

        def submit_buffer(executor, futures: set, group_id: int) -> None:
            docs = buffers[group_id]
            if not docs:
                return
            payload = {
                "version_root": str(self.version_root),
                "group_id": group_id,
                "docs": docs,
                "vocab_path": str(self.vocab_path),
                "vocab_limit": self.vocab_limit,
                "config_dict": asdict(self.config),
            }
            buffers[group_id] = []
            if executor is None:
                consume_result(_process_group_docs(**payload))
            else:
                futures.add(executor.submit(_process_group_docs, **payload))

        vocab_started_at = time.time()
        if self.workers > 1:
            _get_cached_vocab(str(self.vocab_path), self.vocab_limit)
        vocab_load_sec = round(time.time() - vocab_started_at, 3)
        processing_started_at = time.time()
        executor = (
            ProcessPoolExecutor(max_workers=self.workers) if self.workers > 1 else None
        )
        futures: set = set()
        input_seen = 0
        next_log_at = log_every if log_every else 0
        try:
            for doc in docs_iter:
                input_seen += 1
                doc_key = stable_doc_key(doc)
                group_id = group_id_of(doc_key, self.num_groups)
                buffers[group_id].append(doc)
                if len(buffers[group_id]) >= self.group_chunk_size:
                    submit_buffer(executor, futures, group_id)
                if executor is not None and len(futures) >= self.workers * 2:
                    done, futures = wait(futures, return_when=FIRST_COMPLETED)
                    for future in done:
                        consume_result(future.result())
                if log_every and input_seen >= next_log_at:
                    print(
                        json.dumps(
                            {
                                "event": "semantics_progress",
                                "input_seen": input_seen,
                                **stats,
                            },
                            ensure_ascii=False,
                        )
                    )
                    next_log_at += log_every

            for group_id in range(self.num_groups):
                submit_buffer(executor, futures, group_id)
            if futures:
                done, _pending = wait(futures)
                for future in done:
                    consume_result(future.result())
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

        meta = {
            "version": self.version,
            "version_root": str(self.version_root),
            "vocab_load_sec": vocab_load_sec,
            "processing_sec": round(time.time() - processing_started_at, 3),
            **stats,
        }
        (self.version_root / "build_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return meta
