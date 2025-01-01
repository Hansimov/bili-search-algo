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
