import argparse
import importlib
import numpy as np

from tclogger import logger, logstr, brk, brp

from models.fasttext.test import TEST_PAIRS, TEST_KEYWORDS
from models.fasttext.run import FasttextModelRunnerClient
from models.fasttext.run import FasttextModelRunnerRemote
from models.vectors.calcs import dot_sim

import models.fasttext.test

importlib.reload(models.fasttext.test)


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
    def score2str(num: float, round_digits: int = 2, br: bool = True) -> str:
        num_str = f"{num:.{round_digits}f}"
        if br:
            num_str = brp(num_str)
        if num < 0.25:
            str_func = logstr.warn
        elif num < 0.5:
            str_func = logstr.mesg
        else:
            str_func = logstr.hint
        return str_func(num_str)

    def weight2str(num: float, round_digits: int = 2, br: bool = True) -> str:
        num_str = f"{num:.{round_digits}f}"
        if br:
            num_str = brk(num_str)
        return logstr.file(num_str)

    def token2str(token: str, score: float, weight: float) -> str:
        if score < 0.25 or weight < 0.25:
            str_func = logstr.warn
        elif score < 0.5 or weight < 0.5:
            str_func = logstr.mesg
        else:
            str_func = logstr.success
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

    logger.note(f"> Testing (similarity):")
    for query, samples in TEST_PAIRS:
        query_tokens = rc.preprocess(query)
        if rc.attr("frequenizer"):
            query_token_weights = rc.frequenizer_func(
                "calc_weight_of_tokens", query_tokens
            )
        else:
            query_token_weights = [1.0] * len(query_tokens)
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


class RunnerTesterArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def parse_args(self):
        self.args = super().parse_args()


def main(args: argparse.Namespace):
    client = FasttextModelRunnerClient(model_class="doc")
    rc = client.runner
    # test_most_similar_vocab(rc)
    # test_wv_func(rc)
    # test_pair_similarities(rc)
    test_doc_sims(rc)


if __name__ == "__main__":
    parser = RunnerTesterArgParser()
    args = parser.parse_args()
    main(args)

    # python -m models.fasttext.test_runner
