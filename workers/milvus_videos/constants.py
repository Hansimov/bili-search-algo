# used in generator
MONGO_VIDEOS_COLLECTION = "videos"
MONGO_VIDEOS_TAGS_COLLECTION = "videos_tags"
MILVUS_VIDEOS_COLLECTION = "videos"
MONGO_VIDEOS_TAGS_AGG_AS_NAME = f"{MONGO_VIDEOS_TAGS_COLLECTION}_agg"

MONGO_VIDEOS_COLLECTION_ID = "bvid"
MONGO_VIDEOS_TAGS_COLLECTION_ID = "bvid"
MILVUS_VIDEOS_COLLECTION_ID = "bvid"

MONGO_VIDEOS_FIELDS = [
    *["bvid", "title", "desc", "owner"],
    *["stat", "insert_at", "pubdate", "duration"],
]
MONGO_VIDEOS_TAGS_FIELDS = ["tags", "ptid"]


# used in converter
FIELD_VECTOR_WEIGHTS = {
    "title": 1.0,
    "tags": 2.0,
    "owner.name": 2.0,
    "desc": 0.5,
}
STAT_KEYS = ["view", "danmaku", "reply", "favorite", "coin", "share", "like"]
