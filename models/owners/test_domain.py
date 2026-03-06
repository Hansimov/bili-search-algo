from models.owners.domain import (
    OwnerDomainCentroidClassifier,
    OwnerDomainLinearClassifier,
    OwnerDomainNaiveBayesClassifier,
    evaluate_multiple_models,
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

    assert set(results.keys()) == {"centroid", "naive_bayes", "linear"}
    assert results["centroid"]["metrics"]["test_size"] >= 2
    assert results["naive_bayes"]["metrics"]["test_size"] >= 2
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
