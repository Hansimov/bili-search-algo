from tclogger import logger, logstr

from models.fasttext.test import TEST_PAIRS
from models.vectors.calcs import dot_sim


def test_embedder(embedder):
    for pair in TEST_PAIRS:
        query = pair[0]
        samples = pair[1]
        if isinstance(query, list):
            query = " ".join(query)
        query_vector = embedder.embed(query)
        sample_vectors = []
        scores = []
        for sample in samples:
            if isinstance(sample, list):
                sample = " ".join(sample)
            sample_vector = embedder.embed(sample)
            sample_vectors.append(sample_vector)
            score = dot_sim(query_vector, sample_vector, 4)
            scores.append(score)
        sample_scores = list(zip(samples, scores))
        sample_scores.sort(key=lambda x: x[-1], reverse=True)

        logger.note(f"  * [{logstr.file(query)}]: ")
        for sample, score in sample_scores:
            logger.success(f"    * {score:>.4f}: {logstr.file(sample)}")
