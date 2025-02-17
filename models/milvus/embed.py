import numpy as np
import os
import sys

# avoid conflits of local "datasets" module
sys.path = [p for p in sys.path if p != os.getcwd()]

from milvus_model.hybrid import BGEM3EmbeddingFunction
from tclogger import logger, logstr, dict_to_str, brk
from typing import Literal, Union


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

    def embed_list(
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

    def embed(
        self,
        sentences: Union[str, list[str]],
        dense_type: Literal["dense", "sparse"] = "dense",
    ) -> np.ndarray:
        if dense_type == "sparse":
            raise NotImplementedError("Sparse embedding would be supported later")
        if isinstance(sentences, str):
            sentences = [sentences]
        return self.embed_list(sentences, dense_type=dense_type)[0]


if __name__ == "__main__":
    from models.embedders.tests import test_embedder

    embedder = MilvusEmbedder(verbose=True)
    embedder.load_model()
    test_embedder(embedder)

    # python -m models.milvus.embed
