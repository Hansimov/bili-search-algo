from __future__ import annotations

import argparse
import json
import random

from collections import Counter, defaultdict
from pathlib import Path

from models.sentencepiece.vocab_filters import build_token_profile
from models.vocab_cleanup import (
    contains_cjk,
    count_digits,
    count_keyword_hits,
    count_latin,
    is_cta_phrase,
    is_curated_noise_token,
    is_title_template_noise,
    looks_like_random_ascii,
    looks_like_random_mixed_ascii,
    normalize_common_token,
)


def parse_token(line: str, suffix: str) -> str:
    line = line.rstrip("\n")
    if not line:
        return ""
    if suffix == ".csv":
        return line.split(",", 1)[0].strip()
    if "\t" in line:
        return line.split("\t", 1)[0].strip()
    return line.strip()


def reservoir_sample(path: Path, sample_size: int, seed: int) -> list[str]:
    randomizer = random.Random(seed)
    sample: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            token = parse_token(line, path.suffix)
            if not token or token == "word":
                continue
            if len(sample) < sample_size:
                sample.append(token)
                continue
            picked_idx = randomizer.randint(1, idx)
            if picked_idx <= sample_size:
                sample[picked_idx - 1] = token
    return sample


def classify_token(token: str) -> list[str]:
    flags: list[str] = []
    normalized = normalize_common_token(token, lowercase=False)
    profile = build_token_profile(normalized)
    has_cjk = contains_cjk(normalized)
    latin_len = count_latin(normalized)
    digit_len = count_digits(normalized)

    if is_curated_noise_token(normalized):
        flags.append("curated_noise")
    if profile.malformed:
        flags.append("malformed")
    if is_title_template_noise(normalized):
        flags.append("title_template_noise")
    if is_cta_phrase(normalized):
        flags.append("cta_phrase")
    if looks_like_random_ascii(normalized.lower()):
        flags.append("random_ascii")
    if looks_like_random_mixed_ascii(normalized.lower()):
        flags.append("random_mixed_ascii")
    if has_cjk and latin_len:
        flags.append("cjk_latin_mixed")
    if has_cjk and digit_len:
        flags.append("cjk_digit_mixed")
    if count_keyword_hits(normalized, ("直播间", "收藏", "投稿", "搬运")):
        flags.append("promotion_or_template_root")
    if not flags:
        flags.append("looks_meaningful")
    return flags


def build_report(path: Path, sample_size: int, seed: int) -> dict:
    sample = reservoir_sample(path, sample_size=sample_size, seed=seed)
    counts = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    for token in sample:
        flags = classify_token(token)
        for flag in flags:
            counts[flag] += 1
            if len(examples[flag]) < 20:
                examples[flag].append(token)
    return {
        "path": str(path),
        "sample_size": len(sample),
        "counts": dict(counts.most_common()),
        "examples": dict(examples),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Uniformly sample large vocab artifacts and summarize likely noise patterns."
    )
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    reports = [
        build_report(path=path, sample_size=args.sample_size, seed=args.seed)
        for path in args.paths
    ]
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
