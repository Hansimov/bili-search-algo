import argparse
import sys

from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.sentencepiece.vocab_filters import build_token_profile


def load_tokens(vocab_path: Path) -> list[str]:
    tokens = []
    with open(vocab_path, "r", encoding="utf-8") as rf:
        for line in rf:
            parts = line.rstrip("\n").split("\t")
            if parts and parts[0]:
                tokens.append(parts[0])
    return tokens


def summarize(tokens: list[str], sample_size: int = 25) -> dict:
    counter = Counter()
    ascii_samples = []
    malformed_samples = []
    for token in tokens:
        profile = build_token_profile(token)
        counter["total"] += 1
        if profile.has_cjk:
            counter["has_cjk"] += 1
        if profile.is_ascii_token:
            counter["ascii_token"] += 1
            if len(token) <= 3:
                counter["ascii_short"] += 1
            if len(ascii_samples) < sample_size:
                ascii_samples.append(token)
        if profile.is_pure_digits:
            counter["pure_digits"] += 1
        if profile.malformed:
            counter["malformed"] += 1
            if len(malformed_samples) < sample_size:
                malformed_samples.append(token)
        if profile.is_ascii_token and profile.has_connector:
            counter["ascii_connector"] += 1

    return {
        "counts": dict(counter),
        "ascii_samples": ascii_samples,
        "malformed_samples": malformed_samples,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Summarize SentencePiece vocab noise patterns for one or more vocab files."
    )
    parser.add_argument("paths", nargs="+", help=".vocab files to inspect")
    parser.add_argument("-n", "--sample-size", type=int, default=25)
    args = parser.parse_args()

    for raw_path in args.paths:
        vocab_path = Path(raw_path)
        tokens = load_tokens(vocab_path)
        summary = summarize(tokens, sample_size=args.sample_size)
        print(f"== {vocab_path} ==")
        for key, value in sorted(summary["counts"].items()):
            print(f"{key}: {value}")
        print(f"ascii_samples: {summary['ascii_samples']}")
        print(f"malformed_samples: {summary['malformed_samples']}")
        print()


if __name__ == "__main__":
    main()
