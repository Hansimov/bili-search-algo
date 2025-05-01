import argparse
import importlib
import numpy as np

from tclogger import logger, logstr, brk, brp, dict_to_str

from models.fasttext.test import TEST_PAIRS, TEST_KEYWORDS, TEST_SENTENCES
from models.fasttext.run import FasttextModelRunnerClient
from models.fasttext.run import FasttextModelRunnerRemote
from models.vectors.calcs import dot_sim

import models.fasttext.test

importlib.reload(models.fasttext.test)

# === string converters: score, weight, token === #

THS_LOW = 0.25
THS_MID = 0.4


def score2str(num: float, round_digits: int = 2, br: bool = True) -> str:
    num_str = f"{num:.{round_digits}f}"
    if br:
        num_str = brp(num_str)
    if num < THS_LOW:
        str_func = logstr.warn
    elif num < THS_MID:
        str_func = logstr.mesg
    else:
        str_func = logstr.hint
    return str_func(num_str)


def weight2str(num: float, round_digits: int = 2, br: bool = True) -> str:
    num_str = f"{num:.{round_digits}f}"
    if br:
        num_str = brk(num_str)
    return logstr.file(num_str)


def token2str(token: str, score: float = 1.0, weight: float = 1.0) -> str:
    if score < THS_LOW or weight < THS_LOW:
        str_func = logstr.warn
    elif score < THS_MID or weight < THS_MID:
        str_func = logstr.mesg
    else:
        str_func = logstr.okay
    return str_func(token)


def scoreweight2str(
    score: float, weight: float, round_digits: int = 1, br: bool = True
) -> str:
    score_str = score2str(score, round_digits, br=False)
    weight_str = weight2str(weight, round_digits, br=False)
    num_str = f"{score_str}*{weight_str}"
    if br:
        num_str = brp(num_str)
    return logstr.line(num_str)


# === test funcs === #


def test_most_similar_vocab(rc: FasttextModelRunnerRemote):
    logger.note(f"> Testing:")
    for word in TEST_KEYWORDS:
        word = rc.preprocess(word)
        logger.mesg(f"  * [{logstr.file(word)}]:")
        results = rc.most_similar_vocab(positive=word, topn=10)[:6]
        for result in results:
            res_word, res_score = result
            logger.success(f"    * {res_score:>.4f}: {res_word}")
    logger.file(f"* {rc.attr('model_prefix')}")


def test_wv_func(rc: FasttextModelRunnerRemote):
    logger.note(f"> Testing (func):")
    for word in TEST_KEYWORDS:
        word = rc.preprocess(word)
        logger.mesg(f"  * [{logstr.file(word)}]:")
        results = rc.wv_func("most_similar", positive=word, topn=10)[:6]
        for result in results:
            res_word, res_score = result
            logger.success(f"    * {res_score:>.4f}: {res_word}")
    logger.file(f"* {rc.attr('model_prefix')}")


def test_pair_similarities(rc: FasttextModelRunnerRemote):
    logger.note(f"> Testing (similarity):")
    for query, samples in TEST_PAIRS:
        query_tokens = rc.preprocess(query)
        query_token_weights = rc.frequenizer_func("calc_weight_of_tokens", query_tokens)
        query_tokens_str_list = [
            f"{token}{weight2str(weight)}"
            for token, weight in zip(query_tokens, query_token_weights)
        ]
        query_tokens_str = " ".join(query_tokens_str_list)
        logger.note(f"  * [{query_tokens_str}]:")
        sample_tokens_list = [rc.preprocess(sample) for sample in samples]
        results = rc.calc_pairs_scores(query_tokens, sample_tokens_list, level="word")
        for result in results:
            sample_tokens, sample_score, token_scores, token_weights = result
            tokens_str_list = []
            for token, token_weight, token_score in zip(
                sample_tokens, token_weights, token_scores
            ):
                token_str = token2str(token, score=token_score, weight=token_weight)
                score_weight_str = scoreweight2str(
                    score=token_score, weight=token_weight
                )
                token_score_str = f"{token_str}{score_weight_str}"
                tokens_str_list.append(token_score_str)
            tokens_str = " ".join(tokens_str_list)
            logger.line(f"    * {sample_score:>.4f}: [{tokens_str}]")


def test_doc_sims(rc: FasttextModelRunnerRemote):
    query_vec_func = rc.calc_stretch_query_vector
    sample_vec_func = rc.calc_stretch_sample_vector
    logger.note(f"> Testing doc-level similarity:")
    for query, samples in TEST_PAIRS:
        query_vec = query_vec_func(query).astype(np.float16)
        sample_vecs = [sample_vec_func(sample).astype(np.float16) for sample in samples]
        sims = [dot_sim(query_vec, sample_vec) for sample_vec in sample_vecs]
        sample_sims = list(zip(samples, sims))
        sample_sims.sort(key=lambda x: x[1], reverse=True)
        logger.note(f"> [{query}]")
        for sample, sim in sample_sims:
            sim_str = logstr.mesg(f"{sim:.4f}")
            logger.mesg(f"  * {sim_str}: {sample}")
    logger.mesg(
        f"* docvec_dim: {logstr.success(rc.attr('docvec_dim'))}\n"
        f"* vector_dim: {logstr.success(query_vec.shape)}"
    )


def calc_window_token_scores(
    token_vecs: list[np.ndarray], dist: int = None, include_self: bool = False
) -> list[float]:
    token_scores = []
    if dist is None:
        sum_vec = np.sum(token_vecs, axis=0)
        for i, token_vec in enumerate(token_vecs):
            if include_self:
                token_score = dot_sim(token_vec, sum_vec)
            else:
                token_score = dot_sim(token_vec, np.subtract(sum_vec, token_vec))
            token_scores.append(token_score)
    else:
        for i, token_vec in enumerate(token_vecs):
            start = max(0, i - dist)
            end = min(len(token_vecs), i + dist + 1)
            if not include_self and dist > 0:
                window_token_vecs = token_vecs[start:i] + token_vecs[i + 1 : end]
            else:
                window_token_vecs = token_vecs[start:end]
            window_vec = np.sum(window_token_vecs, axis=0)
            token_score = dot_sim(token_vec, window_vec)
            token_scores.append(token_score)
    return token_scores


def test_tokens_importance(rc: FasttextModelRunnerRemote):
    logger.note(f"> Testing token importance:")
    for sentence in TEST_SENTENCES:
        tokens = rc.preprocess(sentence, max_char_len=None)
        token_vecs = [rc.get_vector(token) for token in tokens]
        token_scores = calc_window_token_scores(token_vecs, dist=None)
        tokens_str_list = []
        for token, score in zip(tokens, token_scores):
            token_str = token2str(token, score=score)
            score_str = score2str(score)
            token_score_str = f"{token_str}{score_str}"
            tokens_str_list.append(token_score_str)
        tokens_str = " ".join(tokens_str_list)
        logger.line(f"  * {tokens_str}")


def get_most_unimportant_token(
    tokens: list[str], token_vecs: list[np.ndarray], doc_vec: np.ndarray
) -> tuple[int, float]:
    drop_token_diffs = []
    # only_token_corrs = []
    for i, token_vec in enumerate(token_vecs):
        # drop_vec = np.subtract(doc_vec, token_vec)
        # drop_sim = dot_sim(doc_vec, drop_vec)
        drop_vecs = token_vecs[:i] + token_vecs[i + 1 :]
        drop_sims = [dot_sim(token_vec, drop_vec) for drop_vec in drop_vecs]
        sum_sim = sum(drop_sims)
        drop_diff = len(drop_vecs) - sum_sim
        # only_corr = dot_sim(token_vec, doc_vec)
        drop_token_diffs.append(drop_diff)
        # only_token_corrs.append(only_corr)
    drop_token_diffs_str_list = [f"{diff:.4f}" for diff in drop_token_diffs]
    drop_token_idx = np.argmin(drop_token_diffs)
    drop_diff = drop_token_diffs[drop_token_idx]
    # only_token_corrs_str_list = [f"{corr:.4f}" for corr in only_token_corrs]
    # core_token_idx = np.argmax(only_token_corrs)
    # core_corr = only_token_corrs[core_token_idx]
    info = {
        "tokens": tokens,
        "diffs": drop_token_diffs_str_list,
        # "corrs": only_token_corrs_str_list,
        "drop_token": f"{tokens[drop_token_idx]} ({drop_diff:.4f})",
        # "core_token": f"{tokens[core_token_idx]} ({core_corr:.4f})",
    }
    logger.line(dict_to_str(info, align_list_side="r"))
    return drop_token_idx, drop_diff


def test_drop_token_by_importance(rc: FasttextModelRunnerRemote):
    logger.note(f"> Testing drop token by importance:")
    for sentence in TEST_SENTENCES:
        tokens = rc.preprocess(sentence, max_char_len=None)
        # logger.mesg(f"  * [{' '.join(tokens)}]")
        # token_vecs = [
        #     rc.calc_stretch_sample_vector(token, tokenize=False, shift_offset=idx)
        #     for idx, token in enumerate(tokens)
        # ]
        # doc_vec = rc.calc_stretch_sample_vector(tokens, tokenize=False)
        token_vecs = [rc.get_vector(token) for token in tokens]
        doc_vec = np.sum(token_vecs, axis=0)
        get_most_unimportant_token(tokens, token_vecs, doc_vec)


class RunnerTesterArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def parse_args(self):
        self.args = super().parse_args()


def main(args: argparse.Namespace):
    logger.note("> Loading runner client:")
    client = FasttextModelRunnerClient(model_class="doc")
    rc = client.runner
    logger.okay(f"  * {rc.attr('model_prefix')}")
    # test_most_similar_vocab(rc)
    # test_wv_func(rc)
    # test_pair_similarities(rc)
    # test_doc_sims(rc)
    # test_tokens_importance(rc)
    test_drop_token_by_importance(rc)


if __name__ == "__main__":
    parser = RunnerTesterArgParser()
    args = parser.parse_args()
    main(args)

    # python -m models.fasttext.test_runner
