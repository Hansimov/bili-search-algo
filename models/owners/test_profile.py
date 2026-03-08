from models.coretok.pipeline import CoreTokTrainingPipeline
from models.owners.profile import (
    OwnerCoreTokProfileRefiner,
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
    like: int = 20,
    coin: int = 5,
    favorite: int = 10,
):
    return {
        "owner": {"mid": mid, "name": name},
        "title": title,
        "tags": tags,
        "desc": desc,
        "pubdate": pubdate,
        "stat": {
            "view": view,
            "like": like,
            "coin": coin,
            "favorite": favorite,
        },
    }


def test_owner_profile_accumulator_builds_raw_profile_and_recency_stats():
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
            view=8000,
        )
    )

    profile = acc.build()

    assert profile["mid"] == 1
    assert profile["total_videos"] == 2
    assert profile["total_view"] == 8500
    assert profile["recent_7d_videos"] == 1
    assert profile["recent_30d_videos"] == 2
    assert profile["feature_weights"]
    assert all(term.startswith("b") for term in profile["topic_terms"])
    assert "科技" in profile["topic_phrases"]
    assert profile["influence_score"] > 0
    assert profile["quality_score"] > 0
    assert profile["activity_score"] > 0
    assert profile["sample_titles"] == ["相机测评", "芯片拆解"]
    assert profile["desc_samples"]


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
        "desc_samples": ["长期更新科技内容"],
        "snapshot_at": 1700000000,
        "profile_version": "v2",
        "influence_score": 0.4,
        "quality_score": 0.5,
        "activity_score": 0.6,
        "days_since_last": 30,
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
        "desc_samples": ["偏芯片拆解"],
        "snapshot_at": 1710000000,
        "profile_version": "v2",
        "influence_score": 0.45,
        "quality_score": 0.55,
        "activity_score": 0.7,
        "days_since_last": 10,
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
    assert merged["sample_titles"] == ["相机测评", "芯片拆解"]
    assert merged["desc_samples"] == ["长期更新科技内容", "偏芯片拆解"]
    assert merged["days_since_last"] == 10


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


def test_owner_coretok_profile_refiner_outputs_token_fields():
    pipeline = CoreTokTrainingPipeline()
    tag_sequences = pipeline.train_stage1(["科技", "数码", "游戏", "黑神话悟空"])
    text_sequences = pipeline.train_stage2(
        [
            "科技观察室",
            "相机测评",
            "长期做硬件和相机评测",
            "黑神话悟空流程攻略",
        ]
    )
    pipeline.train_importance(tag_sequences, text_sequences)

    raw_profiles = [
        {
            "_id": 1,
            "mid": 1,
            "name": "科技观察室",
            "top_tags": ["科技", "数码", "日常"],
            "sample_titles": ["相机测评", "芯片拆解"],
            "desc_samples": ["长期做硬件和相机评测"],
            "influence_score": 0.6,
            "quality_score": 0.7,
            "activity_score": 0.5,
            "latest_pubdate": 1700000000,
            "recent_30d_videos": 4,
            "recent_7d_videos": 1,
            "days_since_last": 5,
            "total_videos": 20,
            "total_view": 100000,
            "total_like": 3000,
            "total_coin": 500,
            "total_favorite": 1000,
            "snapshot_at": 1700000100,
        }
    ]

    refiner = OwnerCoreTokProfileRefiner(
        pipeline=pipeline,
        bundle_version="coretok-test",
        core_tag_limit=8,
        core_text_limit=8,
    )
    refined_profiles, stats = refiner.refine(raw_profiles)

    assert len(refined_profiles) == 1
    refined = refined_profiles[0]
    assert refined["core_tokenizer_version"] == "coretok-test"
    assert refined["profile_domain_ready"] is True
    assert refined["core_tag_token_ids"]
    assert refined["core_text_token_ids"]
    assert len(refined["core_tag_token_ids"]) == len(refined["core_tag_token_weights"])
    assert len(refined["core_text_token_ids"]) == len(
        refined["core_text_token_weights"]
    )
    assert stats["domain_ready_count"] == 1
    assert "size_reduction_ratio" in stats
