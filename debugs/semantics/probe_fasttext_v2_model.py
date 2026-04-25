from __future__ import annotations

import argparse
import json

from pathlib import Path


DEFAULT_TERMS = ["显卡", "gpu", "英伟达", "价格", "奔驰", "洗地机"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--terms", nargs="*", default=DEFAULT_TERMS)
    parser.add_argument("--topn", type=int, default=8)
    args = parser.parse_args()

    from gensim.models import FastText

    model = FastText.load(str(args.model_path))
    result = {}
    for term in args.terms:
        if not term:
            continue
        try:
            result[term] = [
                {"term": item, "score": round(float(score), 4)}
                for item, score in model.wv.most_similar(term, topn=args.topn)
            ]
        except KeyError:
            result[term] = []
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
