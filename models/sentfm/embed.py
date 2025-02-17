import numpy as np

from sentence_transformers import SentenceTransformer
from tclogger import logger, logstr, dict_to_str, brk, brp
from typing import Literal, Union


class SentfmEmbedder:
    def __init__(
        self,
        model_name: str = "BAAI/bge-base-zh-v1.5",
        verbose: bool = False,
    ):
        self.model_name = model_name
        self.verbose = verbose

    def load_model(self):
        if self.verbose:
            logger.note(f"> Loading model: {logstr.file(brk(self.model_name))}")
        self.model = SentenceTransformer(self.model_name)

    def embed(self, sentences: Union[str, list[str]]) -> np.ndarray:
        return self.model.encode(sentences)


if __name__ == "__main__":
    from models.tests import test_embedder

    embedder = SentfmEmbedder("BAAI/bge-base-zh-v1.5", verbose=True)
    embedder.load_model()
    test_embedder(embedder)

    # python -m models.sentfm.embed
