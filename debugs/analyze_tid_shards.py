from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from pathlib import Path

from configs.envs import MONGO_ENVS
from data_utils.videos.filter import REGION_MONGO_FILTERS
from sedb import MongoOperator


IGNORED_REGIONS = {"recent", "test"}
DEFAULT_OUTPUT_PATH = Path("debugs/tid_shard_analysis.json")


@dataclass
class Unit:
    key: str
    region: str
    ptid: int | None
    tids: list[int]
    count: int


def normalize_region_rows(region: str, rows: list[dict]) -> list[Unit]:
    units = []
    for row in rows:
        ptid = row["_id"].get("ptid")
        tid = row["_id"].get("tid")
        if tid is None:
            continue
        units.append(
            Unit(
                key=f"{region}:tid:{tid}",
                region=region,
                ptid=ptid,
                tids=[tid],
                count=row["count"],
            )
        )
    return sorted(units, key=lambda unit: unit.count, reverse=True)


def fetch_region_tid_rows(
    collect, region: str, mongo_filter: dict
) -> tuple[str, list[dict]]:
    pipeline = [
        {"$match": mongo_filter},
        {
            "$group": {
                "_id": {"ptid": "$ptid", "tid": "$tid"},
                "count": {"$sum": 1},
            }
        },
        {"$sort": {"count": -1}},
    ]
    rows = list(collect.aggregate(pipeline, allowDiskUse=True))
    return region, rows


def fetch_tid_units(collect, workers: int = 8) -> list[Unit]:
    region_filters = {
        key: value
        for key, value in REGION_MONGO_FILTERS.items()
        if key not in IGNORED_REGIONS
    }
    units: list[Unit] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(fetch_region_tid_rows, collect, region, mongo_filter)
            for region, mongo_filter in region_filters.items()
        ]
        for future in futures:
            region, rows = future.result()
            units.extend(normalize_region_rows(region, rows))
    units.sort(key=lambda unit: unit.count, reverse=True)
    return units


def score_assignment(shards: list[list[Unit]], target: float) -> tuple[float, int, int]:
    totals = [sum(unit.count for unit in shard) for shard in shards]
    imbalance = sum(abs(total - target) for total in totals)
    distinct_region_penalty = sum(
        len({unit.region for unit in shard}) for shard in shards
    )
    fragmentation = defaultdict(int)
    for shard in shards:
        for region in {unit.region for unit in shard}:
            fragmentation[region] += 1
    fragmentation_penalty = sum(max(0, count - 1) for count in fragmentation.values())
    return imbalance, distinct_region_penalty, fragmentation_penalty


def assign_units(units: list[Unit], shard_count: int) -> list[list[Unit]]:
    shards: list[list[Unit]] = [[] for _ in range(shard_count)]
    totals = [0] * shard_count
    target = sum(unit.count for unit in units) / shard_count

    for unit in units:
        best_idx = None
        best_score = None
        for shard_idx in range(shard_count):
            trial_shards = [list(shard) for shard in shards]
            trial_shards[shard_idx].append(unit)
            score = score_assignment(trial_shards, target)
            if best_score is None or score < best_score:
                best_score = score
                best_idx = shard_idx
        shards[best_idx].append(unit)
        totals[best_idx] += unit.count
    return shards


def summarize_shards(shards: list[list[Unit]]) -> list[dict]:
    res = []
    for idx, shard in enumerate(shards):
        total = sum(unit.count for unit in shard)
        res.append(
            {
                "shard_idx": idx,
                "total_count": total,
                "regions": sorted({unit.region for unit in shard}),
                "units": [asdict(unit) for unit in shard],
            }
        )
    return res


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )
    args = parser.parse_args()

    mongo = MongoOperator(MONGO_ENVS)
    collect = mongo.db["videos"]
    units = fetch_tid_units(collect, workers=args.workers)
    shards = assign_units(units, args.workers)
    totals = [sum(unit.count for unit in shard) for shard in shards]
    output = {
        "workers": args.workers,
        "total_count": sum(unit.count for unit in units),
        "unit_count": len(units),
        "target_per_shard": sum(unit.count for unit in units) / args.workers,
        "max_shard_count": max(totals) if totals else 0,
        "min_shard_count": min(totals) if totals else 0,
        "ignored_regions": sorted(IGNORED_REGIONS),
        "region_filters": {
            key: value
            for key, value in REGION_MONGO_FILTERS.items()
            if key not in IGNORED_REGIONS
        },
        "units": [asdict(unit) for unit in units],
        "shards": summarize_shards(shards),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(args.output)
    for shard in output["shards"]:
        print(shard["shard_idx"], shard["total_count"], ",".join(shard["regions"]))


if __name__ == "__main__":
    main()
