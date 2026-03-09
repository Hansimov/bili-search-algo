from models.coretok.train import (
    aggregate_scale_summaries,
    build_prepared_training_corpus,
    build_eval_dataset,
    collect_training_texts,
    compute_stability,
    default_tuning_configs,
    evaluate_owner_retrieval,
    parse_seed_list,
    split_owner_rows,
    train_pipeline,
    DEFAULT_SCALES,
)


def make_owner_row(mid: int, name: str, topics: list[str]) -> dict:
    videos = []
    for index, topic in enumerate(topics, start=1):
        videos.append(
            {
                "title": f"{topic} 深度解析 {index}",
                "tags": f"{topic}, 测评",
                "desc": f"这是关于 {topic} 的视频 {index}",
                "pubdate": 1_700_000_000 + index,
            }
        )
    return {"mid": mid, "name": name, "videos": videos}


def test_split_owner_rows_and_collect_training_texts_keep_train_eval_disjoint():
    owner_rows = [
        make_owner_row(1, "A", ["黑神话", "黑神话流程", "黑神话剧情", "黑神话攻略"]),
        make_owner_row(2, "B", ["相机", "镜头", "测评", "摄影"]),
        make_owner_row(3, "C", ["纳塔", "原神", "剧情", "拆包"]),
        make_owner_row(4, "D", ["芯片", "数码", "硬件", "跑分"]),
    ]

    train_rows, eval_rows = split_owner_rows(owner_rows, eval_owner_count=2, seed=7)

    assert train_rows
    assert eval_rows
    assert not {row["mid"] for row in train_rows} & {row["mid"] for row in eval_rows}

    tags, texts = collect_training_texts(train_rows)
    assert tags
    assert texts


def test_holdout_eval_pipeline_returns_nonzero_retrieval_metrics():
    train_rows = [
        make_owner_row(
            1, "黑神话研究所", ["黑神话悟空", "黑神话流程", "黑神话剧情", "黑神话攻略"]
        ),
        make_owner_row(
            2, "相机实验室", ["相机测评", "镜头评测", "摄影器材", "相机参数"]
        ),
        make_owner_row(
            3, "纳塔档案馆", ["纳塔剧情", "原神纳塔", "纳塔拆包", "纳塔角色"]
        ),
    ]
    eval_rows = [
        make_owner_row(
            11, "黑神话档案馆", ["黑神话Boss", "黑神话结局", "黑神话流程", "黑神话攻略"]
        ),
        make_owner_row(
            12, "摄影频道", ["相机推荐", "镜头解析", "摄影实战", "相机测评"]
        ),
    ]
    tags, texts = collect_training_texts(train_rows)
    profiles, queries = build_eval_dataset(eval_rows, query_per_owner=3)
    pipeline = train_pipeline(
        tags, texts, default_tuning_configs(DEFAULT_SCALES["tiny"])[0]
    )

    metrics = evaluate_owner_retrieval(pipeline, profiles, queries)

    assert metrics["query_count"] == len(queries)
    assert metrics["profile_count"] == len(profiles)
    assert metrics["query_coverage"] > 0
    assert metrics["recall_at_5"] >= metrics["recall_at_1"]


def test_prepared_training_corpus_contains_reusable_candidate_plans():
    prepared = build_prepared_training_corpus(
        ["黑神话悟空", "黑神话流程", "相机测评"],
        ["黑神话悟空流程解析", "相机拆解视频"],
    )

    assert prepared["tag_candidate_plans"]
    assert prepared["text_candidate_plans"]
    assert prepared["tag_unique_count"] == len(prepared["tag_candidate_plans"])
    assert prepared["text_unique_count"] == len(prepared["text_candidate_plans"])


def test_train_pipeline_exposes_stage2_substage_stats():
    pipeline = train_pipeline(
        ["黑神话悟空", "相机测评"],
        ["黑神话悟空流程解析", "相机拆解视频", "相机拆解视频"],
        default_tuning_configs(DEFAULT_SCALES["tiny"])[0],
    )

    assert pipeline.text_tokenizer is not None
    assert pipeline.text_tokenizer.last_fit_stats["stage2_unique_candidate_count"] > 0
    assert "stage2_materialize_seconds" in pipeline.text_tokenizer.last_fit_stats
    assert "stage2_materialize_cache_hits" in pipeline.text_tokenizer.last_fit_stats
    assert (
        pipeline.text_tokenizer.last_fit_stats["stage2_aggressive_materialize"] is True
    )


def test_compute_stability_flags_consistent_scores_as_stable():
    stability = compute_stability(
        [
            {"metrics": {"recall_at_5": 0.28}},
            {"metrics": {"recall_at_5": 0.27}},
            {"metrics": {"recall_at_5": 0.29}},
        ]
    )

    assert stability["stable"] is True
    assert stability["best_recall_at_5"] == 0.29


def test_parse_seed_list_deduplicates_and_falls_back_to_default_seed():
    assert parse_seed_list(None, 42) == [42]
    assert parse_seed_list("7, 11, 7, 13", 42) == [7, 11, 13]
    assert parse_seed_list(" , ", 42) == [42]


def test_aggregate_scale_summaries_reports_cross_seed_stability():
    aggregate = aggregate_scale_summaries(
        "tiny",
        [
            {
                "scale": "tiny",
                "seed": 3,
                "dataset": {
                    "owner_count": 60,
                    "train_owner_count": 45,
                    "eval_owner_count": 15,
                    "eval_query_count": 48,
                },
                "best": {
                    "config": {"tag_novelty": 0.35, "text_novelty": 0.68},
                    "metrics": {
                        "query_coverage": 0.62,
                        "empty_query_ratio": 0.38,
                        "mrr": 0.41,
                        "recall_at_1": 0.33,
                        "recall_at_5": 0.36,
                        "recall_at_10": 0.42,
                    },
                },
                "stability": {"stable": True},
            },
            {
                "scale": "tiny",
                "seed": 7,
                "dataset": {
                    "owner_count": 60,
                    "train_owner_count": 46,
                    "eval_owner_count": 14,
                    "eval_query_count": 44,
                },
                "best": {
                    "config": {"tag_novelty": 0.35, "text_novelty": 0.68},
                    "metrics": {
                        "query_coverage": 0.58,
                        "empty_query_ratio": 0.42,
                        "mrr": 0.39,
                        "recall_at_1": 0.31,
                        "recall_at_5": 0.34,
                        "recall_at_10": 0.4,
                    },
                },
                "stability": {"stable": False},
            },
            {
                "scale": "tiny",
                "seed": 11,
                "dataset": {
                    "owner_count": 60,
                    "train_owner_count": 47,
                    "eval_owner_count": 13,
                    "eval_query_count": 40,
                },
                "best": {
                    "config": {"tag_novelty": 0.4, "text_novelty": 0.7},
                    "metrics": {
                        "query_coverage": 0.6,
                        "empty_query_ratio": 0.4,
                        "mrr": 0.43,
                        "recall_at_1": 0.35,
                        "recall_at_5": 0.38,
                        "recall_at_10": 0.45,
                    },
                },
                "stability": {"stable": True},
            },
        ],
    )

    assert aggregate["seed_count"] == 3
    assert aggregate["stability"]["stable"] is True
    assert aggregate["stability"]["stable_seed_count"] == 2
    assert aggregate["metrics"]["recall_at_5"]["mean"] == 0.36
    assert aggregate["best_config_counts"]
