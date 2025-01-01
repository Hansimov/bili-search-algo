import argparse

from typing import Literal, Union

REGION_MONGO_FILTERS = {
    "douga_anime": {"ptid": {"$in": [1, 13, 167]}},
    "music_dance": {"ptid": {"$in": [3, 129]}},
    "mobile_game": {"tid": 172},
    "other_game": {"ptid": 4, "tid": {"$ne": 172}},
    "tech_sports": {"ptid": {"$in": [188, 234, 223]}},
    "daily_life": {"tid": 21},
    "other_life": {"ptid": {"$in": [160, 211, 217]}, "tid": {"$ne": 21}},
    "cine_movie": {"ptid": {"$in": [181, 177, 23, 11, 165]}},
    "fashion_ent": {"ptid": {"$in": [155, 5]}},
    "know_info": {"ptid": {"$in": [36, 202]}},
}


def construct_mongo_filter_from_field_values(
    field: Literal["tid", "ptid"], values: Union[int, list[int]], reverse: bool = False
) -> dict:
    if not values:
        return {}
    mongo_filter = {}
    if isinstance(values, int):
        if not reverse:
            mongo_filter[field] = values
        else:
            mongo_filter[field] = {"$ne": values}
    else:
        if not reverse:
            mongo_filter[field] = {"$in": values}
        else:
            mongo_filter[field] = {"$nin": values}
    return mongo_filter


def construct_mongo_filter_from_args(args: argparse.Namespace):
    if args.filter_group:
        mongo_filter = REGION_MONGO_FILTERS[args.filter_group]
    else:
        if args.tid:
            filter_field, filter_field_values = "tid", args.tid
        elif args.ptid:
            filter_field, filter_field_values = "ptid", args.ptid
        else:
            filter_field, filter_field_values = None, None
        mongo_filter = construct_mongo_filter_from_field_values(
            filter_field, filter_field_values, reverse=args.reverse_filter
        )
    return mongo_filter
