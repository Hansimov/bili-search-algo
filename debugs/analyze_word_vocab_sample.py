from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from models.sentencepiece.vocab_filters import build_token_profile

RE_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
RE_LATIN = re.compile(r"[A-Za-z]")
RE_DIGIT = re.compile(r"\d")
RE_HASHY = re.compile(r"[#@~!！?？,，/\\|<>\[\](){}]+")
RE_CJK_PUNCT = re.compile(
    r'[：:，,、·・～~!！?？#＃@＠/／\\|｜\[\]()（）{}【】<>《》"]'
)
RE_MULTI_SPACE = re.compile(r"\s+")


def normalize_spaces(text: str) -> str:
    return RE_MULTI_SPACE.sub(" ", text).strip()


def classify_token(token: str) -> list[str]:
    flags: list[str] = []
    profile = build_token_profile(token)
    has_cjk = bool(RE_CJK.search(token))
    has_latin = bool(RE_LATIN.search(token))
    has_digit = bool(RE_DIGIT.search(token))
    has_space = " " in token
    if profile.malformed:
        flags.append("malformed")
    if has_cjk and profile.cjk_char_len > 8:
        flags.append("cjk_too_long")
    if RE_HASHY.search(token):
        flags.append("special_symbols")
    if has_cjk and has_latin:
        flags.append("cjk_latin_mixed")
    if has_cjk and has_digit:
        flags.append("cjk_digit_mixed")
    if not has_cjk and has_latin and has_digit:
        flags.append("latin_digit_mixed")
    if token != normalize_spaces(token):
        flags.append("spacing_noise")
    if has_space:
        flags.append("contains_space")
    if profile.is_ascii_token and len(token) <= 3:
        flags.append("short_ascii")
    if not flags:
        flags.append("looks_meaningful")
    return flags


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sample_path", type=Path)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    counts = Counter()
    samples: dict[str, list[str]] = {}
    total = 0

    with args.sample_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if args.limit is not None and total >= args.limit:
                break
            line = line.rstrip("\n")
            if not line:
                continue
            _, token = line.split("\t", 1)
            total += 1
            for flag in classify_token(token):
                counts[flag] += 1
                samples.setdefault(flag, [])
                if len(samples[flag]) < 25:
                    samples[flag].append(token)

    report = {
        "total": total,
        "counts": dict(counts.most_common()),
        "samples": samples,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
