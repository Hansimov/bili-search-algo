from pathlib import Path

import polars as pl

from models.word.eng import (
    ChineseWordsExtractor,
    EnglishWordsExtractor,
    REGION_SHARDS,
    build_skip_shard_plans,
    format_shard_progress,
    merge_partial_records,
)


def test_english_extractor_filters_noise_and_keeps_meaningful_tokens():
    extractor = EnglishWordsExtractor()

    tokens = extractor.extract(
        "GTA5 | counting stars | bowlroll.net | 0a | use code | ps5 | zxcvbnm | watch | wte4kt | video"
    )

    assert "gta5" in tokens
    assert "counting stars" in tokens
    assert "ps5" in tokens
    assert "bowlroll.net" not in tokens
    assert "0a" not in tokens
    assert "use code" not in tokens
    assert "zxcvbnm" not in tokens
    assert "watch" not in tokens
    assert "wte4kt" not in tokens
    assert "video" not in tokens

    more_tokens = extractor.extract("cjbc2009 | syc1664 | dgc2208 | spacex")
    assert "cjbc2009" not in more_tokens
    assert "syc1664" not in more_tokens
    assert "dgc2208" not in more_tokens
    assert "spacex" in more_tokens

    phrase_tokens = extractor.extract(
        "in a nutshell | back to december | use this code | one two three four five six | one two three four five six seven"
    )
    assert "in a nutshell" in phrase_tokens
    assert "back to december" in phrase_tokens
    assert "one two three four five six" in phrase_tokens
    assert "one two three four five six seven" not in phrase_tokens
    assert "use this code" not in phrase_tokens

    length_tokens = extractor.extract(
        "meta http-equiv | super ultra amazing soundtrack"
    )
    assert "meta http-equiv" not in length_tokens
    assert "super ultra amazing soundtrack" not in length_tokens

    random_tokens = extractor.extract(
        "uhzhnlzyhnzjkl | caisbggeeaeyaw | mzrlodbinwflza | yzhewzhnjzklz | ylwk23nuoya | ius9jkzjr | sojy1k4x4as"
    )
    assert "uhzhnlzyhnzjkl" not in random_tokens
    assert "caisbggeeaeyaw" not in random_tokens
    assert "mzrlodbinwflza" not in random_tokens
    assert "yzhewzhnjzklz" not in random_tokens
    assert "ylwk23nuoya" not in random_tokens
    assert "ius9jkzjr" not in random_tokens
    assert "sojy1k4x4as" not in random_tokens

    legit_long_tokens = extractor.extract(
        "minecraft | spacex | springboot | soundcloud | nintendoswitch | p1harmony | inst360"
    )
    assert "minecraft" in legit_long_tokens
    assert "spacex" in legit_long_tokens
    assert "springboot" in legit_long_tokens
    assert "soundcloud" in legit_long_tokens
    assert "nintendoswitch" in legit_long_tokens
    assert "p1harmony" in legit_long_tokens
    assert "inst360" not in legit_long_tokens


def test_chinese_extractor_keeps_titles_and_drops_noise():
    extractor = ChineseWordsExtractor()

    tokens = extractor.extract(
        "原神，启动#黑神话：悟空#严阵以待！#Re：从零开始的异世界生活#防脱洗发水什么牌子的好？#a玖"
    )

    assert "原神，启动" in tokens
    assert "黑神话：悟空" in tokens
    assert "严阵以待" in tokens
    assert "re：从零开始的异世界生活" not in tokens
    assert "防脱洗发水什么牌子的好" not in tokens
    assert "a玖" not in tokens
    assert "梦幻手游造梦计划，梦幻西游手游" not in extractor.extract(
        "梦幻手游造梦计划, 梦幻西游手游"
    )
    assert "梦幻手游造梦计划，" not in extractor.extract("梦幻手游造梦计划，")

    more_tokens = extractor.extract(
        "分享#计划#创作灵感#物华弥新创作者激励计划 第三期#黑神话：悟空"
    )
    assert "分享" in more_tokens
    assert "计划" in more_tokens
    assert "创作灵感" in more_tokens
    assert "物华弥新创作者激励计划 第三期" not in more_tokens
    assert "黑神话：悟空" in more_tokens

    noise_tokens = extractor.extract("知识分享官#怎么追女生#值不值得买#黑神话：悟空")
    assert "知识分享官" in noise_tokens
    assert "怎么追女生" not in noise_tokens
    assert "值不值得买" not in noise_tokens
    assert "黑神话：悟空" in noise_tokens

    comma_tokens = extractor.extract(
        "生活记录，记录#打卡挑战，必剪创作#王者荣耀，moba#原神，启动"
    )
    assert "生活记录，记录" not in comma_tokens
    assert "打卡挑战，必剪创作" not in comma_tokens
    assert "王者荣耀，moba" not in comma_tokens
    assert "生活记录" in comma_tokens
    assert "记录" in comma_tokens
    assert "打卡挑战" in comma_tokens
    assert "必剪创作" in comma_tokens
    assert "王者荣耀" in comma_tokens
    assert "原神，启动" in comma_tokens


def test_build_skip_shard_plans_respects_base_skip_and_last_open_end():
    shards = build_skip_shard_plans(total_count=100, workers=4, base_skip_count=10)

    assert shards[0]["skip_count"] == 10
    assert shards[0]["max_count"] == 23
    assert shards[-1]["skip_count"] == 79
    assert shards[-1]["max_count"] == 21
    assert len(shards) == 4


def test_region_shards_follow_sentencepiece_train_split():
    assert REGION_SHARDS == [
        ["cine_movie"],
        ["douga_anime"],
        ["tech_sports"],
        ["music_dance"],
        ["fashion_ent"],
        ["know_info"],
        ["daily_life"],
        ["other_life"],
        ["mobile_game"],
        ["other_game"],
    ]


def test_merge_partial_records_sums_counts(tmp_path: Path):
    partial_a = tmp_path / "part_a.csv"
    partial_b = tmp_path / "part_b.csv"
    output = tmp_path / "merged.csv"

    pl.DataFrame(
        [
            {"word": "gta5", "doc_freq": 3, "term_freq": 5},
            {"word": "ps5", "doc_freq": 1, "term_freq": 1},
        ]
    ).write_csv(partial_a)
    pl.DataFrame(
        [
            {"word": "gta5", "doc_freq": 2, "term_freq": 2},
            {"word": "elden ring", "doc_freq": 4, "term_freq": 6},
        ]
    ).write_csv(partial_b)

    merge_partial_records(
        partial_paths=[partial_a, partial_b],
        dump_path=output,
        sort_key="doc_freq",
        min_freq=2,
    )

    merged = pl.read_csv(output).sort("word")
    assert merged.to_dicts() == [
        {"word": "elden ring", "doc_freq": 4, "term_freq": 6},
        {"word": "gta5", "doc_freq": 5, "term_freq": 7},
    ]


def test_format_shard_progress_summarizes_running_and_done_shards():
    lines = format_shard_progress(
        [
            {
                "shard_idx": 0,
                "state": "running",
                "processed_docs": 50,
                "total_docs": 100,
            },
            {
                "shard_idx": 1,
                "state": "done",
                "processed_docs": 100,
                "total_docs": 100,
            },
        ],
        workers=2,
    )

    assert lines[0] == "> Shard progress: 150/200 docs (75.00%) | done 1/2 | running 1"
    assert lines[1] == "  * shard 00: running 50/100 (50.00%)"
    assert lines[2] == "  * shard 01: done 100/100 (100.00%)"
