import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from sedb import MongoOperator
from tclogger import logger, logstr, TCLogbar, ts_to_str, str_to_ts, dict_to_str
from typing import Literal, Union, Iterable

from configs.envs import MONGO_ENVS


class DateRanger:
    def get_seps(self) -> list[str]:
        date_seps = ["2009-06-01"]
        now = datetime.now()
        for year in range(2017, now.year + 1):
            if year < 2019:
                sep_num = 1
            elif year < 2021:
                sep_num = 2
            else:
                sep_num = 4
            if year == now.year:
                max_month = now.month
            else:
                max_month = 12
            this_year_seps = [
                f"{year}-{month:02}-01" for month in range(1, max_month, 12 // sep_num)
            ]
            date_seps.extend(this_year_seps)
        date_seps[-1] = f"{now.year}-{now.month:02}-{now.day:02}"
        return date_seps

    def seps_to_ranges(self, seps: list[str]) -> list[tuple[str, str]]:
        ranges = []
        for idx, sep in enumerate(seps[:-1]):
            ranges.append((sep, seps[idx + 1]))
        return ranges

    def get_ranges(self) -> list[tuple[str, str]]:
        return self.seps_to_ranges(self.get_seps())


class VideoStatsCounter:
    PERCENTILES = [0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999, 1.0]
    EXPS = range(2, 8)
    VIEW_BOUNDARIES = [0, *[10**exp for exp in EXPS], 1e10]

    def __init__(self, collection: str):
        self.collection = collection
        self.stats_json = Path(__file__).parent / "stats.json"
        self.stat_ratios_json = Path(__file__).parent / "stat_ratios.json"
        self.init_mongo()

    def init_mongo(self):
        self.mongo = MongoOperator(
            MONGO_ENVS, connect_msg=f"from {self.__class__.__name__}"
        )
        self.collect = self.mongo.db[self.collection]

    def get_cursor(
        self,
        filter_op: Literal["gt", "lt", "gte", "lte", "range"],
        filter_range: Union[int, str, tuple, list],
        filter_index: str = "pubdate",
        sort_index: str = "pubdate",
        sort_order: Literal["asc", "desc"] = "asc",
    ):
        self.cursor = self.mongo.get_cursor(
            collection=self.collection,
            filter_index=filter_index,
            filter_op=filter_op,
            filter_range=filter_range,
            sort_index=sort_index,
            sort_order=sort_order,
        )

    def construct_stat_agg_dict(self, stat_fields: list) -> dict:
        agg_dict = {}
        range_dict = {
            "count": {"$sum": 1},
        }
        agg_dict.update(range_dict)

        max_n = len(self.PERCENTILES) - 1  # avoid conflicts of align list
        for stat_field in stat_fields:
            field = stat_field.replace("stat.", "")
            field_agg_dict = {
                f"{field}_avg": {"$avg": f"${stat_field}"},
                f"{field}_std": {"$stdDevPop": f"${stat_field}"},
                f"{field}_median": {
                    "$median": {"input": f"${stat_field}", "method": "approximate"}
                },
                f"{field}_maxn": {"$maxN": {"input": f"${stat_field}", "n": max_n}},
                f"{field}_percentile": {
                    "$percentile": {
                        "input": f"${stat_field}",
                        "method": "approximate",
                        "p": deepcopy(self.PERCENTILES),
                    },
                },
            }
            agg_dict.update(field_agg_dict)
        return agg_dict

    def round_agg_result(self, result: dict, ndigits: int = 0) -> dict:
        new_result = deepcopy(result)
        round_func = round if ndigits == 0 else lambda x: round(x, ndigits=ndigits)
        for key in result:
            if isinstance(result[key], float):
                new_result[key] = round_func(result[key])
            elif isinstance(result[key], list):
                new_result[key] = [
                    round_func(item) if isinstance(item, float) else item
                    for item in result[key]
                ]
            else:
                pass
        return new_result

    def count_stats(self, filter_params: dict, stat_fields: list) -> dict:
        filter_dict = self.mongo.format_filter(**filter_params)
        pipeline = [
            {"$match": filter_dict},
            {
                "$group": {
                    "_id": None,
                    **self.construct_stat_agg_dict(stat_fields),
                }
            },
        ]
        result = {
            "date_range": list(filter_params["filter_range"]),
        }
        result.update(self.collect.aggregate(pipeline).next())
        result = self.round_agg_result(result)
        result["percentiles"] = deepcopy(self.PERCENTILES)
        return result

    def construct_stat_ratio_agg_dict(self, stat_fields: list) -> dict:
        ratio_agg_dict = {}
        for stat_field in stat_fields:
            field = stat_field.replace("stat.", "")
            field_view_ratio_dict = {
                f"{field}_vr": {
                    "$cond": {
                        "if": {"$lte": ["$stat.view", 0]},
                        "then": 0,
                        "else": {
                            "$divide": [
                                {"$multiply": [f"${stat_field}", 100]},
                                "$stat.view",
                            ]
                        },
                    }
                }
            }
            ratio_agg_dict.update(field_view_ratio_dict)
        project_agg_dict = {"$project": {"stat.view": 1, **ratio_agg_dict}}
        return project_agg_dict

    def construct_stat_ratio_bucket_dict(self, stat_fields: list) -> dict:
        range_dict = {
            "min_view": {"$min": "$stat.view"},
            "max_view": {"$max": "$stat.view"},
            "count": {"$sum": 1},
        }
        bucket_output_dict = {}
        bucket_output_dict.update(range_dict)

        ratio_bucket_agg_dict = {}
        for stat_field in stat_fields:
            field = stat_field.replace("stat.", "")
            field_ratio_agg_dict = {
                f"{field}_vr_avg": {"$avg": f"${field}_vr"},
                f"{field}_vr_std": {"$stdDevPop": f"${field}_vr"},
                f"{field}_vr_median": {
                    "$median": {"input": f"${field}_vr", "method": "approximate"}
                },
                f"{field}_vr_percentile": {
                    "$percentile": {
                        "input": f"${field}_vr",
                        "method": "approximate",
                        "p": deepcopy(self.PERCENTILES),
                    },
                },
            }
            ratio_bucket_agg_dict.update(field_ratio_agg_dict)
        bucket_output_dict.update(ratio_bucket_agg_dict)

        bucket_dict = {
            "$bucket": {
                "groupBy": "$stat.view",
                "boundaries": deepcopy(self.VIEW_BOUNDARIES),
                "default": "others",
                "output": bucket_output_dict,
            }
        }
        return bucket_dict

    def count_stat_ratios(self, filter_params: dict, stat_fields: list) -> dict:
        filter_dict = self.mongo.format_filter(**filter_params)
        pipeline = [
            {"$match": filter_dict},
            self.construct_stat_ratio_agg_dict(stat_fields),
            self.construct_stat_ratio_bucket_dict(stat_fields),
        ]
        agg_result = self.collect.aggregate(pipeline)
        result = {
            "date_range": list(filter_params["filter_range"]),
            "percentiles": deepcopy(self.PERCENTILES),
            "buckets": {},
        }
        for idx, bucket in enumerate(agg_result):
            bucket = self.round_agg_result(bucket, ndigits=4)
            bucket["bucket_idx"] = idx
            result["buckets"][bucket["_id"]] = bucket
        return result

    def save(self, stat_res: dict = None, stat_ratio_res: dict = None):
        if stat_res:
            if not self.stats_json.exists():
                stats_data = {}
            else:
                with open(self.stats_json, "r") as rf:
                    stats_data = json.load(rf)
            stats_data.update(stat_res)
            with open(self.stats_json, "w") as wf:
                json.dump(stats_data, wf, indent=4)

        if stat_ratio_res:
            if not self.stat_ratios_json.exists():
                stat_ratios_data = {}
            else:
                with open(self.stat_ratios_json, "r") as rf:
                    stat_ratios_data = json.load(rf)
            stat_ratios_data.update(stat_ratio_res)
            with open(self.stat_ratios_json, "w") as wf:
                json.dump(stat_ratios_data, wf, indent=4)

    def count_by_years(self):
        date_ranges = DateRanger().get_ranges()
        logger.note("> Date ranges:")
        for date_range in date_ranges:
            logger.mesg(f"* {date_range}", indent=2)

        for date_range in date_ranges[:]:
            logger.note(f"> {date_range}:")
            filter_params = {
                "filter_index": "pubdate",
                "filter_op": "range",
                "filter_range": date_range,
            }
            logger.note("> Counting stats:", indent=2)
            stat_result = counter.count_stats(
                filter_params=filter_params,
                stat_fields=["stat.view", "stat.coin", "stat.danmaku"],
            )
            stat_res = {date_range[0]: stat_result}
            self.save(stat_res=stat_res)
            logger.success(f"✓ Updated stats for: {logstr.file(date_range)}", indent=4)
            logger.mesg(dict_to_str(stat_result), indent=4)

            logger.note("> Counting stat ratios:", indent=2)
            stat_raio_result = counter.count_stat_ratios(
                filter_params=filter_params,
                stat_fields=["stat.coin", "stat.danmaku"],
            )
            stat_ratio_res = {date_range[0]: stat_raio_result}
            self.save(stat_ratio_res=stat_ratio_res)
            logger.success(
                f"✓ Updated stat ratios for: {logstr.file(date_range)} ", indent=4
            )
            for idx, v in enumerate(stat_raio_result["buckets"].values()):
                logger.note(f"* Bucket {(idx+1)}:", indent=4)
                logger.mesg(dict_to_str(v), indent=6)


if __name__ == "__main__":
    counter = VideoStatsCounter(collection="videos")
    counter.count_by_years()

    # python -m stats.count
