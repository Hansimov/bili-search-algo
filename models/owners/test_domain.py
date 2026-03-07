from models.owners.domain import (
    OwnerDomainCentroidClassifier,
    OwnerDomainLinearClassifier,
    OwnerDomainNaiveBayesClassifier,
    DEFAULT_WEIGHTED_FIELD_WEIGHTS,
    evaluate_multiple_models,
    tune_weighted_naive_bayes,
    match_filter_spec,
    select_owner_label,
)


def make_video(tid: int, ptid: int, title: str, tags: str, owner_name: str = "UP"):
    return {
        "title": title,
        "tags": tags,
        "desc": title,
        "tid": tid,
        "ptid": ptid,
        "owner": {"mid": 1, "name": owner_name},
        "stat": {"view": 100},
        "pubdate": 1700000000,
    }


def test_match_filter_spec_supports_or_and_in():
    doc = {"tid": 21, "ptid": 160}
    spec = {"$or": [{"ptid": {"$in": [160, 211]}}, {"tid": 21}]}

    assert match_filter_spec(doc, spec) is True
    assert match_filter_spec({"tid": 1, "ptid": 2}, spec) is False


def test_select_owner_label_returns_dominant_group():
    videos = [
        make_video(36, 202, "科技开箱", "科技, 数码"),
        make_video(36, 202, "芯片测试", "科技, 测评"),
        make_video(36, 202, "相机体验", "数码, 科技"),
        make_video(188, 223, "运动随拍", "运动"),
    ]

    label_info = select_owner_label(videos, dominant_ratio=0.6, min_label_videos=2)

    assert label_info is not None
    assert label_info["label"] == "know_info"
    assert label_info["label_count"] == 3


def test_owner_domain_centroid_classifier_predicts_expected_label():
    train_samples = [
        {"label": "know_info", "text": "影视飓风 科技 数码 相机 测评 芯片"},
        {"label": "music_dance", "text": "洛天依 音乐 翻唱 舞蹈 演唱会"},
        {"label": "know_info", "text": "半佛 科技 商业 数码 手机"},
        {"label": "music_dance", "text": "乐队 音乐 现场 舞台 翻唱"},
    ]

    classifier = OwnerDomainCentroidClassifier(min_token_freq=1)
    classifier.fit(train_samples)
    pred_label, scores = classifier.predict("数码 相机 科技 测评")

    assert pred_label == "know_info"
    assert scores["know_info"] > scores["music_dance"]


def test_owner_domain_naive_bayes_classifier_predicts_expected_label():
    train_samples = [
        {"label": "know_info", "text": "影视飓风 科技 数码 相机 测评 芯片"},
        {"label": "music_dance", "text": "洛天依 音乐 翻唱 舞蹈 演唱会"},
        {"label": "know_info", "text": "半佛 科技 商业 数码 手机"},
        {"label": "music_dance", "text": "乐队 音乐 现场 舞台 翻唱"},
    ]

    classifier = OwnerDomainNaiveBayesClassifier(min_token_freq=1)
    classifier.fit(train_samples)
    pred_label, scores = classifier.predict("数码 相机 科技 测评")

    assert pred_label == "know_info"
    assert scores["know_info"] > scores["music_dance"]


def test_evaluate_multiple_models_compare_returns_both_results():
    samples = [
        {
            "mid": 1,
            "owner_name": "A1",
            "label": "know_info",
            "text": "科技 数码 相机 芯片",
        },
        {
            "mid": 2,
            "owner_name": "A2",
            "label": "know_info",
            "text": "科技 手机 数码 商业",
        },
        {
            "mid": 3,
            "owner_name": "A3",
            "label": "know_info",
            "text": "相机 测评 科技 数码",
        },
        {
            "mid": 4,
            "owner_name": "B1",
            "label": "music_dance",
            "text": "音乐 翻唱 舞蹈 演唱会",
        },
        {
            "mid": 5,
            "owner_name": "B2",
            "label": "music_dance",
            "text": "乐队 舞台 现场 音乐",
        },
        {
            "mid": 6,
            "owner_name": "B3",
            "label": "music_dance",
            "text": "翻唱 音乐 舞蹈 live",
        },
    ]

    results = evaluate_multiple_models("compare", samples, test_ratio=0.34, seed=7)

    assert set(results.keys()) == {
        "centroid",
        "naive_bayes",
        "naive_bayes_weighted",
        "linear",
    }
    assert results["centroid"]["metrics"]["test_size"] >= 2
    assert results["naive_bayes"]["metrics"]["test_size"] >= 2
    assert results["naive_bayes_weighted"]["metrics"]["test_size"] >= 2
    assert results["linear"]["metrics"]["test_size"] >= 2


def test_owner_domain_linear_classifier_predicts_expected_label():
    train_samples = [
        {"label": "know_info", "text": "影视飓风 科技 数码 相机 测评 芯片"},
        {"label": "music_dance", "text": "洛天依 音乐 翻唱 舞蹈 演唱会"},
        {"label": "know_info", "text": "半佛 科技 商业 数码 手机"},
        {"label": "music_dance", "text": "乐队 音乐 现场 舞台 翻唱"},
    ]

    classifier = OwnerDomainLinearClassifier(min_token_freq=1, epochs=10)
    classifier.fit(train_samples)
    pred_label, scores = classifier.predict("数码 相机 科技 测评")

    assert pred_label == "know_info"
    assert scores["know_info"] > scores["music_dance"]


def test_weighted_naive_bayes_uses_structured_owner_fields():
    train_samples = [
        {
            "label": "know_info",
            "owner_name": "科技观察室",
            "top_tags": ["科技", "数码", "芯片"],
            "sample_titles": ["相机测评", "手机体验"],
            "desc_samples": ["长期做数码和芯片内容"],
        },
        {
            "label": "music_dance",
            "owner_name": "音乐现场台",
            "top_tags": ["音乐", "翻唱", "舞蹈"],
            "sample_titles": ["live现场", "翻唱合集"],
            "desc_samples": ["偏音乐演出和翻唱"],
        },
    ]

    classifier = OwnerDomainNaiveBayesClassifier(
        min_token_freq=1,
        field_weights=DEFAULT_WEIGHTED_FIELD_WEIGHTS,
    )
    classifier.fit(train_samples)
    pred_label, scores = classifier.predict(
        {
            "owner_name": "硬件研究所",
            "top_tags": ["科技", "数码"],
            "sample_titles": ["芯片拆解", "相机体验"],
            "desc_samples": ["持续更新手机和硬件评测"],
        }
    )

    assert pred_label == "know_info"
    assert scores["know_info"] > scores["music_dance"]


def test_tune_weighted_naive_bayes_returns_best_config_and_leaderboard():
    samples = [
        {
            "mid": 1,
            "owner_name": "科技观察室",
            "label": "know_info",
            "top_tags": ["科技", "数码", "芯片"],
            "sample_titles": ["相机测评", "硬件拆解"],
            "desc_samples": ["长期更新科技内容"],
            "text": "科技 数码 芯片 相机测评 硬件拆解",
        },
        {
            "mid": 2,
            "owner_name": "硬件研究社",
            "label": "know_info",
            "top_tags": ["手机", "科技", "评测"],
            "sample_titles": ["手机体验", "芯片分析"],
            "desc_samples": ["偏数码评测"],
            "text": "手机 科技 评测 芯片分析",
        },
        {
            "mid": 3,
            "owner_name": "商业科技局",
            "label": "know_info",
            "top_tags": ["商业", "科技"],
            "sample_titles": ["产品复盘", "行业观察"],
            "desc_samples": ["科技商业分析"],
            "text": "商业 科技 产品复盘 行业观察",
        },
        {
            "mid": 4,
            "owner_name": "音乐现场台",
            "label": "music_dance",
            "top_tags": ["音乐", "翻唱", "live"],
            "sample_titles": ["翻唱合集", "现场舞台"],
            "desc_samples": ["偏音乐演出"],
            "text": "音乐 翻唱 live 舞台",
        },
        {
            "mid": 5,
            "owner_name": "舞蹈热场机",
            "label": "music_dance",
            "top_tags": ["舞蹈", "音乐"],
            "sample_titles": ["编舞练习", "live现场"],
            "desc_samples": ["更新舞蹈和现场"],
            "text": "舞蹈 音乐 编舞 live",
        },
        {
            "mid": 6,
            "owner_name": "乐队录音棚",
            "label": "music_dance",
            "top_tags": ["乐队", "音乐"],
            "sample_titles": ["现场翻唱", "排练日常"],
            "desc_samples": ["以乐队演出为主"],
            "text": "乐队 音乐 现场 翻唱",
        },
    ]

    result = tune_weighted_naive_bayes(
        samples,
        test_ratio=0.34,
        seed=7,
        weight_grid={
            "owner_name": [3.0, 4.0],
            "top_tags": [2.0, 3.0],
            "sample_titles": [1.0],
            "desc_samples": [1.0],
        },
        alpha_grid=[0.5, 1.0],
        top_k=3,
    )

    assert result["metrics"]["best_config"]["alpha"] in {0.5, 1.0}
    assert "field_weights" in result["metrics"]["best_config"]
    assert len(result["metrics"]["leaderboard"]) == 3
    assert result["metrics"]["search_space"]["candidate_count"] == 8
