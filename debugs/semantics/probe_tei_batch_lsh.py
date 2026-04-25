from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

from itertools import combinations
from pathlib import Path

import numpy as np

from models.semantics.embedding_filter import hash_similarity


def resolve_tei_endpoints() -> list[str]:
    raw = os.getenv("SEMANTICS_TEI_ENDPOINTS") or os.getenv("TEI_CLIENTS_ENDPOINTS")
    if raw:
        if raw.strip().startswith("["):
            return [str(item) for item in json.loads(raw)]
        return [part.strip() for part in raw.split(",") if part.strip()]
    bili_search_root = Path("/home/asimov/repos/bili-search")
    if bili_search_root.exists():
        sys.path.insert(0, str(bili_search_root))
        try:
            from configs.envs import TEI_CLIENTS_ENDPOINTS

            return list(TEI_CLIENTS_ENDPOINTS)
        finally:
            sys.path.pop(0)
    return []


DEFAULT_TERMS = [
    "显卡",
    "gpu",
    "英伟达",
    "价格",
    "价格曲线",
    "奔驰",
    "洗地机",
]


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 0:
        return 0.0
    return float(np.dot(left, right) / denom)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--terms", nargs="*", default=DEFAULT_TERMS)
    parser.add_argument("--bitn", type=int, default=2048)
    args = parser.parse_args()

    warnings.filterwarnings("ignore", message='Field "model_.*protected namespace')
    from tfmx import TEIClients

    endpoints = resolve_tei_endpoints()
    if not endpoints:
        raise RuntimeError("Set SEMANTICS_TEI_ENDPOINTS or TEI_CLIENTS_ENDPOINTS")
    client = TEIClients(endpoints=endpoints)
    terms = list(dict.fromkeys(args.terms))
    embeddings = client.embed(terms, normalize=True, truncate=True)
    hashes = client.lsh(terms, bitn=args.bitn, normalize=True, truncate=True)
    vectors = {
        term: np.asarray(vector, dtype=np.float32)
        for term, vector in zip(terms, embeddings)
    }
    hashes_by_term = dict(zip(terms, hashes))
    pairs = []
    for left, right in combinations(terms, 2):
        pairs.append(
            {
                "left": left,
                "right": right,
                "cosine": round(cosine(vectors[left], vectors[right]), 4),
                "lsh": round(hash_similarity(hashes_by_term[left], hashes_by_term[right]), 4),
            }
        )
    pairs.sort(key=lambda item: (-item["cosine"], item["left"], item["right"]))
    print(
        json.dumps(
            {
                "terms": terms,
                "bitn": args.bitn,
                "hash_hex_chars": len(hashes[0]) if hashes else 0,
                "pairs": pairs,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
