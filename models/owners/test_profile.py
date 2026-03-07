from models.owners.profile import (
    OwnerProfileAccumulator,
    build_owner_range_query,
    build_owner_shard_expr,
    build_profile_upsert_filter,
    merge_profile_docs,
    split_owner_mid_range,
)


def make_video(
    mid: int,
    name: str,
    title: str,
    tags: str,
    desc: str,
    pubdate: int,
    view: int = 100,
):
    return {
        "owner": {"mid": mid, "name": name},
        "title": title,
        "tags": tags,
        "desc": desc,
        "pubdate": pubdate,
        "stat": {"view": view},
    }


def test_owner_profile_accumulator_builds_topic_terms_and_recency_stats():
    now_ts = 1_800_000_000
    acc = OwnerProfileAccumulator(
        make_video(
            1,
            "科技观察室",
            "相机测评",
            "科技, 数码, 相机",
            "长期更新芯片和相机内容",
            now_ts - 2 * 86400,
            view=500,
        ),
        now_ts=now_ts,
        top_k=16,
    )
    acc.add(
        make_video(
            1,
            "科技观察室",
            "芯片拆解",
            "科技, 芯片",
            "硬件评测",
            now_ts - 20 * 86400,
            view=800,
        )
    )

    profile = acc.build()

    assert profile["mid"] == 1
    assert profile["total_videos"] == 2
    assert profile["total_view"] == 1300
    assert profile["recent_7d_videos"] == 1
    assert profile["recent_30d_videos"] == 2
    assert profile["feature_weights"]
    assert all(term.startswith("b") for term in profile["topic_terms"])
    assert "科技" in profile["topic_phrases"]
    assert "相机" in profile["profile_text"]
    assert profile["sample_titles"] == ["相机测评", "芯片拆解"]


def test_merge_profile_docs_merges_sparse_terms_and_latest_fields():
    base = {
        "mid": 1,
        "name": "科技观察室",
        "total_videos": 10,
        "total_view": 10000,
        "recent_30d_videos": 3,
        "recent_7d_videos": 1,
        "latest_pubdate": 1700000000,
        "earliest_pubdate": 1600000000,
        "top_tags": ["科技", "数码"],
        "sample_titles": ["相机测评"],
        "feature_weights": {"b1": 10.0, "b2": 6.0},
        "topic_phrases": ["科技", "数码", "相机测评"],
        "snapshot_at": 1700000000,
        "profile_version": "v2",
    }
    delta = {
        "mid": 1,
        "name": "科技观察室",
        "total_videos": 2,
        "total_view": 2500,
        "recent_30d_videos": 2,
        "recent_7d_videos": 1,
        "latest_pubdate": 1710000000,
        "earliest_pubdate": 1690000000,
        "top_tags": ["芯片", "科技"],
        "sample_titles": ["芯片拆解"],
        "feature_weights": {"b1": 3.0, "b3": 8.0},
        "topic_phrases": ["芯片", "科技", "芯片拆解"],
        "snapshot_at": 1710000000,
        "profile_version": "v2",
    }

    merged = merge_profile_docs(base, delta, top_k=8)

    assert merged["total_videos"] == 12
    assert merged["total_view"] == 12500
    assert merged["latest_pubdate"] == 1710000000
    assert merged["earliest_pubdate"] == 1600000000
    assert merged["feature_weights"]["b1"] == 13.0
    assert merged["feature_weights"]["b3"] == 8.0
    assert merged["topic_terms"][0] == "b1"
    assert "芯片" in merged["top_tags"]
    assert "芯片拆解" in merged["topic_phrases"]
    assert "科技观察室" in merged["profile_text"]
    assert merged["sample_titles"] == ["相机测评", "芯片拆解"]


def test_build_owner_shard_expr_validates_and_builds_mod_filter():
    assert build_owner_shard_expr() is None
    assert build_owner_shard_expr(1, 0) is None
    assert build_owner_shard_expr(4, 2) == {"$eq": [{"$mod": ["$owner.mid", 4]}, 2]}


def test_build_owner_range_query_and_split_owner_mid_range():
    assert build_owner_range_query() is None
    assert build_owner_range_query(100, 199) == {
        "owner.mid": {"$gte": 100, "$lte": 199}
    }
    assert split_owner_mid_range(100, 199, 4, 2) == (150, 174)


def test_build_profile_upsert_filter_uses__id_key():
    assert build_profile_upsert_filter({"_id": 123, "mid": 123}) == {"_id": 123}
