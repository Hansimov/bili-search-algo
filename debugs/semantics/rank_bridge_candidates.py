from __future__ import annotations

import argparse
import json

from pathlib import Path

from models.semantics.storage import (
    _can_consider_semantic_bridge,
    _bridge_profile_score,
    decode_normalized_term,
)


def load_profiles(path: Path):
    profiles = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            try:
                profiles[decode_normalized_term(parts[0])] = (
                    int(parts[2]),
                    int(parts[3]),
                    int(parts[4]),
                )
            except ValueError:
                continue
    return profiles


def iter_mapping(path: Path, profiles):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            source = decode_normalized_term(parts[0])
            targets = []
            for index in range(1, len(parts) - 1, 2):
                target = decode_normalized_term(parts[index])
                try:
                    weight = float(parts[index + 1])
                except ValueError:
                    continue
                if _can_consider_semantic_bridge(source, target, profiles):
                    targets.append((target, weight))
            if targets:
                targets.sort(key=lambda item: (-item[1], item[0]))
                yield source, targets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--terms", nargs="*", default=[])
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    profiles = load_profiles(args.bundle / "nodes.tsv")
    rows = sorted(
        iter_mapping(args.bundle / "doc_cooccurrence.tsv", profiles),
        key=lambda item: (
            -_bridge_profile_score(profiles.get(item[0])),
            -item[1][0][1],
            -len(item[1]),
            item[0],
        ),
    )
    terms = set(args.terms)
    output = {
        "top": [
            {"rank": index + 1, "source": source, "targets": targets[:8]}
            for index, (source, targets) in enumerate(rows[: args.limit])
        ],
        "terms": {},
    }
    for index, (source, targets) in enumerate(rows, start=1):
        if source in terms:
            output["terms"][source] = {"rank": index, "targets": targets[:12]}
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
