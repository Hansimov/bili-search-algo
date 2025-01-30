import numpy as np

from tclogger import brk, brp, logstr, logger
from typing import Union, Literal

from configs.envs import FASTTEXT_MERGED_MODEL_DIMENSION
from models.fasttext.run import FasttextModelRunnerClient
from models.fasttext.preprocess import FasttextModelPreprocessor
from models.vectors.calcs import dot_sim
from models.vectors.forms import trunc, stretch_copy, stretch_shift_add


class FasttextDocVecRunner:
    def __init__(self):
        self.client = FasttextModelRunnerClient()
        self.runner = self.client.runner
        self.vec_dim_scale = 5
        self.vec_dim = FASTTEXT_MERGED_MODEL_DIMENSION * self.vec_dim_scale

    def preprocess(self, doc: str) -> list[str]:
        # return self.runner.preprocess(doc)
        return list(set(self.runner.preprocess(doc)))

    def get_vector(self, word: str) -> np.ndarray:
        return np.array(self.runner.get_vector(word, tolist=True))

    def calc_vector(
        self,
        doc: Union[str, list[str]],
        ignore_duplicates: bool = True,
        weight_func: Literal["mean", "sum", "freq"] = "mean",
        score_func: Literal["ratio", "quantile", "log", "power"] = "power",
        base: Union[int, float] = None,
        min_weight: float = None,
        max_weight: float = None,
        normalize: bool = True,
    ) -> np.ndarray:
        return np.array(
            self.runner.calc_vector(
                doc,
                ignore_duplicates=ignore_duplicates,
                weight_func=weight_func,
                score_func=score_func,
                base=base,
                min_weight=min_weight,
                max_weight=max_weight,
                normalize=normalize,
                tolist=True,
            )
        )

    def calc_query_vector(self, doc: Union[str, list[str]]) -> np.ndarray:
        return self.calc_vector(
            doc,
            ignore_duplicates=False,
            weight_func="freq",
            score_func="log",
            base=10,
            min_weight=0.01,
            max_weight=1,
            normalize=True,
        )

    def calc_weight_of_sample_token(self, token: str) -> float:
        return self.runner.calc_weight_of_token(
            token,
            score_func="log",
            base=10,
            min_weight=0.1,
            max_weight=1,
        )

    def calc_sample_token_vectors(self, doc: Union[str, list[str]]) -> np.ndarray:
        tokens = self.preprocess(doc)
        vectors = np.array([self.get_vector(token) for token in tokens])
        weights = np.array(
            [self.calc_weight_of_sample_token(token) for token in tokens]
        )
        return vectors * weights[:, np.newaxis]

    def calc_stretch_query_vector(self, doc: Union[str, list[str]]) -> np.ndarray:
        query_vector = self.calc_query_vector(doc)
        return stretch_copy(query_vector, scale=self.vec_dim_scale)

    def calc_stretch_sample_vector(self, doc: Union[str, list[str]]) -> np.ndarray:
        token_vectors = self.calc_sample_token_vectors(doc)
        return stretch_shift_add(token_vectors, scale=self.vec_dim_scale)

    def test_doc_sims(self):
        from models.fasttext.test import TEST_PAIRS

        query_vec_func = self.calc_stretch_query_vector
        sample_vec_func = self.calc_stretch_sample_vector
        logger.note(f"> Testing doc-level similarity:")
        for query, samples in TEST_PAIRS:
            query_vec = query_vec_func(query).astype(np.float16)
            sample_vecs = [
                sample_vec_func(sample).astype(np.float16) for sample in samples
            ]
            sims = [dot_sim(query_vec, sample_vec) for sample_vec in sample_vecs]
            sample_sims = list(zip(samples, sims))
            sample_sims.sort(key=lambda x: x[1], reverse=True)
            logger.note(f"> [{query}]")
            for sample, sim in sample_sims:
                sim_str = logstr.mesg(f"{sim:.4f}")
                logger.mesg(f"  * {sim_str}: {sample}")


if __name__ == "__main__":
    runner = FasttextDocVecRunner()
    doc_vec = runner.test_doc_sims()

    # python -m models.fasttext.docvec
