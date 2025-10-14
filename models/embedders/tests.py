from tclogger import logger, logstr
from typing import Union

from models.fasttext.test import TEST_PAIRS
from models.vectors.calcs import dot_sim
from models.vectors.forms import flat_arr


class EmbedderBase:
    def embed(self, sentences: Union[str, list[str]]):
        raise NotImplementedError("Please implement embed() in the subclass.")


def add_prefix(text: str, prefix: str = "") -> str:
    if prefix:
        return f"{prefix} {text}"
    else:
        return text


def test_embedder(
    embedder: EmbedderBase, query_prefix: str = None, passage_prefix: str = None
):
    for pair in TEST_PAIRS:
        query = pair[0]
        samples = pair[1]
        if isinstance(query, list):
            query = " ".join(query)
        logger.note(f"  * [{logstr.file(query)}]: ")
        query = add_prefix(query, prefix=query_prefix)
        query_vector = flat_arr(embedder.embed(query))
        sample_vectors = []
        scores = []
        for sample in samples:
            if isinstance(sample, list):
                sample = " ".join(sample)
            sample = add_prefix(sample, prefix=passage_prefix)
            sample_vector = flat_arr(embedder.embed(sample))
            sample_vectors.append(sample_vector)
            score = dot_sim(query_vector, sample_vector, 4)
            scores.append(score)
        sample_scores = list(zip(samples, scores))
        sample_scores.sort(key=lambda x: x[-1], reverse=True)
        for sample, score in sample_scores:
            logger.success(f"    * {score:>.4f}: {logstr.file(sample)}")
