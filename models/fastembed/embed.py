import numpy as np

from fastembed import TextEmbedding
from tclogger import logger, logstr, dict_to_str, brk, brp
from typing import Literal, Union, Generator


class FastEmbedder:
    def __init__(
        self,
        model_name: Literal[
            "BAAI/bge-small-zh-v1.5", "jinaai/jina-embeddings-v2-base-zh"
        ] = "BAAI/bge-small-zh-v1.5",
        verbose: bool = False,
    ):
        self.model_name = model_name
        self.verbose = verbose

    def get_model_info(self, model_name: str) -> dict:
        for model_info in TextEmbedding.list_supported_models():
            if model_info["model"] == model_name:
                return model_info

    def list_models(self) -> list[dict]:
        logger.note("> List of supported models:")
        suppored_models = TextEmbedding.list_supported_models()
        for model_info in suppored_models:
            # model_name = model_info["model"]
            # logger.mesg(f"  * {model_name}")
            logger.mesg(dict_to_str(model_info), indent=2)
        return suppored_models

    def load_model(self):
        if self.verbose:
            logger.note(f"> Loading model: {logstr.file(brk(self.model_name))}")
            model_info = self.get_model_info(self.model_name)
            logger.mesg(dict_to_str(model_info), indent=2)
        self.model = TextEmbedding(model_name=self.model_name)

    def embed(
        self,
        sentences: Union[str, list[str]],
        result_format: Literal["list", "generator"] = "list",
    ) -> Union[np.ndarray, list[np.ndarray], Generator[np.ndarray, None, None]]:
        embeddings_generator = self.model.embed(sentences)
        if result_format == "list":
            if isinstance(sentences, str):
                return list(embeddings_generator)[0]
            else:
                return list(embeddings_generator)
        else:
            return embeddings_generator


if __name__ == "__main__":
    from models.tests import test_embedder

    embedder = FastEmbedder("BAAI/bge-small-zh-v1.5", verbose=True)
    # embedder.list_models()
    embedder.load_model()
    test_embedder(embedder)

    # python -m models.fastembed.embed
