from models.vocab_cleanup import (
    is_curated_noise_token,
    is_video_id_token,
    looks_like_keyboard_mash,
    looks_like_random_ascii,
    looks_like_random_mixed_ascii,
    looks_like_stretched_ascii_noise,
    normalize_common_token,
    should_keep_cjk_comma_phrase,
)


def test_normalize_common_token_normalizes_nfkc_and_spaces():
    assert normalize_common_token(" ＡＢＣ　123 ") == "abc 123"


def test_random_ascii_heuristics():
    assert looks_like_random_ascii("uhzhnlzyhnzjkl")
    assert looks_like_random_ascii("asdfgh")
    assert not looks_like_random_ascii("minecraft")
    assert looks_like_random_mixed_ascii("sojy1k4x4as")
    assert not looks_like_random_mixed_ascii("p1harmony")


def test_keyboard_and_stretched_ascii_noise_rules():
    assert looks_like_keyboard_mash("qwerty")
    assert looks_like_keyboard_mash("asdf")
    assert looks_like_stretched_ascii_noise("huaaaaa")
    assert looks_like_stretched_ascii_noise("rebeccasuuuuuu")
    assert not looks_like_stretched_ascii_noise("minecraft")


def test_curated_noise_and_comma_keep_rules():
    assert should_keep_cjk_comma_phrase("原神，启动")
    assert not should_keep_cjk_comma_phrase("生活，记录")
    assert is_video_id_token("av32225766")
    assert is_video_id_token("BV1SM4y1x7AB")
    assert is_curated_noise_token("asdf")
    assert is_curated_noise_token("qwerty")
    assert is_curated_noise_token("huaaaaa")
    assert is_curated_noise_token("请关注我")
    assert is_curated_noise_token("给视频点赞")
    assert is_curated_noise_token("主页收藏有惊喜")
    assert is_curated_noise_token("直播间传送门")
    assert is_curated_noise_token("搬运自")
    assert is_curated_noise_token("av32225766")
    assert not is_curated_noise_token("黑神话：悟空")
    assert not is_curated_noise_token("收藏家")
