import numpy as np
import os
import sys

# avoid conflits of local "datasets" module
sys.path = [p for p in sys.path if p != os.getcwd()]

from milvus_model.hybrid import BGEM3EmbeddingFunction
from tclogger import logger, logstr, dict_to_str, brk, brp
from typing import Literal, Union, Generator

from models.vectors.similarity import dot_sim
from models.fasttext.test import TEST_PAIRS


class MilvusEmbedder:
    def __init__(
        self,
        model_name: Literal["BAAI/bge-m3"] = "BAAI/bge-m3",
        device: Literal["cpu", "cuda:0"] = "cuda:0",
        verbose: bool = False,
    ):
        self.model_name = model_name
        self.device = device
        self.verbose = verbose

    def get_model_info(self):
        model_info = {
            "dim_dense": self.model.dim["dense"],
            "dim_sparse": self.model.dim["sparse"],
        }
        return model_info

    def load_model(self):
        if self.verbose:
            logger.note(f"> Loading model: {logstr.file(brk(self.model_name))}")
        if self.model_name == "BAAI/bge-m3":
            self.model = BGEM3EmbeddingFunction(
                model_name=self.model_name,
                device=self.device,
                use_fp16=False if self.device == "cpu" else True,
            )
            model_info = self.get_model_info()
            logger.file(dict_to_str(model_info), indent=2)
        else:
            raise ValueError(f"Invalid model: [{(self.model_name)}]")

    def embed(
        self, sentences: list[str], dense_type: Literal["dense", "sparse"] = "dense"
    ) -> list[np.ndarray]:
        embed_dict = self.model.encode_documents(sentences)
        if dense_type == "dense":
            embeddings = embed_dict["dense"]
        elif dense_type == "sparse":
            embeddings = embed_dict["sparse"]
        else:
            raise ValueError(f"Invalid dense_type: [{dense_type}]")
        return embeddings

    def test(self):
        for pair in TEST_PAIRS:
            query = pair[0]
            samples = pair[1]
            if isinstance(query, list):
                query = " ".join(query)
            query_vector = self.embed([query])[0]
            sample_vectors = []
            scores = []
            for sample in samples:
                if isinstance(sample, list):
                    sample = " ".join(sample)
                sample_vector = self.embed([sample])[0]
                sample_vectors.append(sample_vector)
                score = dot_sim(query_vector, sample_vector, 4)
                scores.append(score)
            sample_scores = list(zip(samples, scores))
            sample_scores.sort(key=lambda x: x[-1], reverse=True)

            logger.note(f"  * [{logstr.file(query)}]: ")
            for sample, score in sample_scores:
                logger.success(f"    * {score:>.4f}: {logstr.file(sample)}")


if __name__ == "__main__":
    embedder = MilvusEmbedder(verbose=True)
    embedder.load_model()
    embedder.test()

    # python -m models.milvus.embed
