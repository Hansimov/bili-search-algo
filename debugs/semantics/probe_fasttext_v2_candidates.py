from __future__ import annotations

import argparse
import json

from models.fasttext_v2.scoring import FastTextV2CandidateScorer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score fixed candidates with a fasttext_v2 model"
    )
    parser.add_argument("model_path")
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="source=candidate1,candidate2,...",
    )
    parser.add_argument("--raw", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    scorer = FastTextV2CandidateScorer.load(args.model_path, center=not args.raw)
    cases = args.case or [
        "显卡=gpu,英伟达,流量卡推荐,洗地机",
        "价格=铼的价格,多少钱,洗地机,奔驰",
        "奔驰=二手车,小米su7,洗地机,英伟达",
    ]
    output = {}
    for case in cases:
        source, _, rest = case.partition("=")
        candidates = [item.strip() for item in rest.split(",") if item.strip()]
        output[source] = [
            {"term": item.term, "score": round(item.score, 4)}
            for item in scorer.rank(source, candidates)
        ]
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
