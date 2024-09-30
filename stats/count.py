from copy import deepcopy
from sedb import MongoOperator
from tclogger import logger, logstr, TCLogbar, ts_to_str, dict_to_str
from typing import Literal, Union

from configs.envs import MONGO_ENVS


class VideoStatsCounter:
    PERCENTILES = [0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999, 1.0]

    def __init__(self, collection: str):
        self.collection = collection
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

    def count_stats(self, filter_params: dict, stat_fields: list):
        logger.note("> Counting stats:")
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
        result = self.collect.aggregate(pipeline).next()
        result = self.round_agg_result(result)
        result["percentiles"] = deepcopy(self.PERCENTILES)
        logger.mesg(dict_to_str(result), indent=2)


if __name__ == "__main__":
    counter = VideoStatsCounter(collection="videos")
    filter_params = {
        "filter_index": "pubdate",
        "filter_op": "lte",
        "filter_range": "2013-01-01",
    }
    filter_params = {
        "filter_index": "pubdate",
        "filter_op": "range",
        "filter_range": ["2020-01-01", "2020-01-15"],
    }
    logger.note("> Filter params:")
    logger.mesg(dict_to_str(filter_params))
    counter.count_stats(
        filter_params=filter_params,
        stat_fields=["stat.view", "stat.coin", "stat.danmaku"],
    )

    # python -m stats.count
