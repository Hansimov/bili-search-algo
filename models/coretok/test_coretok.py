from models.coretok.core import (
    build_candidate_plan,
    CoreCorpusStats,
    CoreImpEvaluator,
    CoreTagTokenizer,
    CoreTexTokenizer,
    CoreTokenLexicon,
    count_mixed_units,
    is_valid_stage1_tag,
    suggest_token_budget,
)
from models.coretok.pipeline import CoreTokTrainingPipeline


def test_count_mixed_units_and_stage1_validation_follow_new_rules():
    assert count_mixed_units("abc科技1234") == 5
    assert suggest_token_budget("黑神话") == 1
    assert suggest_token_budget("黑神话悟空") == 2
    assert suggest_token_budget("希区柯克镜头语言") == 3

    assert is_valid_stage1_tag("黑神话") is True
    assert is_valid_stage1_tag("abcdef1234567890abcdef12345") is False
    assert is_valid_stage1_tag("这是一个明显超长的中文标签") is False


def test_core_tag_tokenizer_assigns_budgeted_tokens_without_bigram_noise():
    tokenizer = CoreTagTokenizer()
    encoded = tokenizer.fit(["黑神话", "希区柯克镜头语言"], epochs=1)

    short_ids = tokenizer.encode("黑神话")
    long_ids = tokenizer.encode("希区柯克镜头语言")

    assert encoded
    assert len(short_ids) == 1
    assert 1 <= len(long_ids) <= 3

    long_tokens = tokenizer.decode(long_ids)
    assert "希区柯克镜头语言" in long_tokens or "希区柯克" in long_tokens
    assert all(len(token) >= 2 for token in long_tokens)


def test_core_tex_tokenizer_reuses_seed_tokens_before_creating_new_ones():
    tag_tokenizer = CoreTagTokenizer()
    tag_tokenizer.fit(["黑神话悟空", "相机测评"], epochs=1)

    text_tokenizer = CoreTexTokenizer(lexicon=tag_tokenizer.lexicon)
    known_ids = text_tokenizer.encode("黑神话悟空流程解析", allow_new_tokens=False)
    novel_ids = text_tokenizer.encode("纳塔剧情拆包", allow_new_tokens=True)

    known_tokens = text_tokenizer.decode(known_ids)
    novel_tokens = text_tokenizer.decode(novel_ids)

    assert any("黑神话悟空" in token for token in known_tokens)
    assert novel_tokens
    assert any("纳塔剧情拆包" in token or "纳塔剧情" in token for token in novel_tokens)


def test_core_tex_tokenizer_rejects_non_substring_runtime_reuse():
    lexicon = CoreTokenLexicon()
    lexicon.add_token("leatherface", source="text")

    text_tokenizer = CoreTexTokenizer(lexicon=lexicon)
    token_ids = text_tokenizer.encode("Leather Bench", allow_new_tokens=False)

    assert token_ids == []


def test_candidate_plan_can_be_reused_for_text_encoding():
    tag_tokenizer = CoreTagTokenizer()
    tag_tokenizer.fit(["黑神话悟空", "相机测评"], epochs=1)

    text_tokenizer = CoreTexTokenizer(lexicon=tag_tokenizer.lexicon)
    plan = build_candidate_plan("黑神话悟空流程解析", for_stage1=False)
    planned_ids = text_tokenizer.encode(
        "黑神话悟空流程解析",
        allow_new_tokens=False,
        candidate_plan=plan,
    )
    direct_ids = text_tokenizer.encode("黑神话悟空流程解析", allow_new_tokens=False)

    assert planned_ids == direct_ids


def test_core_tex_tokenizer_can_block_singleton_new_tokens_in_stage2():
    tag_tokenizer = CoreTagTokenizer()
    tag_tokenizer.fit(["黑神话悟空", "相机测评"], epochs=1)

    text_tokenizer = CoreTexTokenizer(lexicon=tag_tokenizer.lexicon)
    text_tokenizer.fit(
        ["只出现一次的新概念", "黑神话悟空流程解析", "黑神话悟空流程解析"],
        epochs=1,
        min_new_token_freq=2,
    )

    assert text_tokenizer.lexicon.get_token_id("只出现一次的新概念") is None
    assert text_tokenizer.last_fit_stats["stage2_blocked_new_candidate_count"] >= 1


def test_large_candidate_match_shortlist_keeps_expected_best_match():
    lexicon = CoreTokenLexicon()
    for index in range(160):
        lexicon.add_token(f"黑神话攻略扩展词条{index}", source="text")
    expected_id = lexicon.add_token("黑神话流程解析", source="text")

    matched_id, matched_score = lexicon.find_best_match("黑神话流程解析实机演示")

    assert matched_id == expected_id
    assert matched_score > 0.5


def test_core_imp_evaluator_prefers_tag_dominant_tokens():
    tokenizer = CoreTagTokenizer()
    tag_sequences = tokenizer.fit(["黑神话悟空", "黑神话流程", "相机测评"], epochs=1)

    text_tokenizer = CoreTexTokenizer(lexicon=tokenizer.lexicon)
    text_sequences = text_tokenizer.fit(
        ["黑神话悟空流程解析", "相机拆解视频", "相机参数记录"], epochs=1
    )

    evaluator = CoreImpEvaluator().fit(tag_sequences, text_sequences)
    black_myth_id = tokenizer.lexicon.get_token_id("黑神话悟空")
    camera_id = tokenizer.lexicon.get_token_id("相机测评")

    assert black_myth_id is not None
    assert camera_id is not None
    assert evaluator.score_token(black_myth_id) > 0
    assert evaluator.score_sequence([black_myth_id, camera_id])


def test_training_pipeline_keeps_shared_lexicon_between_stage1_and_stage2():
    pipeline = CoreTokTrainingPipeline()
    stage1 = pipeline.train_stage1(["黑神话悟空", "相机测评"], epochs=1)
    stage2 = pipeline.train_stage2(["黑神话悟空流程解析", "纳塔剧情拆包"], epochs=1)
    importance = pipeline.train_importance(stage1, stage2)

    assert pipeline.tag_tokenizer is not None
    assert pipeline.text_tokenizer is not None
    assert pipeline.tag_tokenizer.lexicon is pipeline.text_tokenizer.lexicon
    assert importance.score_sequence(stage2[0])


def test_corpus_stats_learn_high_coverage_stop_candidates_without_fixed_term_list():
    stats = CoreCorpusStats(min_docs_for_stop=3, stop_coverage_floor=0.2).fit(
        ["日常", "日常", "日常", "日常记录", "黑神话悟空", "相机测评"],
        for_stage1=True,
    )

    assert stats.total_docs >= 5
    assert stats.is_stop_candidate("日常") is True
    assert stats.is_stop_candidate("黑神话悟空") is False
