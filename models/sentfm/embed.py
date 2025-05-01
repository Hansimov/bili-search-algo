import numpy as np

from sentence_transformers import SentenceTransformer
from tclogger import logger, logstr, brk
from typing import Union, Literal

"""
```sh
pip install --upgrade torch torchvision
pip install --upgrade tensorrt
pip install --upgrade 'sentence-transformers[onnx-gpu]'
```
"""


class SentfmEmbedder:
    def __init__(
        self,
        model_name: str = "BAAI/bge-base-zh-v1.5",
        backend: Literal["torch", "onnx"] = "torch",
        model_kwargs: dict = None,
        verbose: bool = False,
    ):
        self.model_name = model_name
        self.backend = backend
        self.model_kwargs = model_kwargs or None
        self.verbose = verbose

    def load_model(self):
        if self.verbose:
            logger.note(f"> Loading model: {logstr.file(brk(self.model_name))}")
        self.model = SentenceTransformer(
            self.model_name, backend=self.backend, model_kwargs=self.model_kwargs
        )

    def embed(self, sentences: Union[str, list[str]]) -> np.ndarray:
        return self.model.encode(sentences)


def test_bge_zh():
    # model_name = "BAAI/bge-base-zh-v1.5"

    # model_name = "BAAI/bge-large-zh-v1.5"
    # embedder = SentfmEmbedder(model_name, verbose=True)

    model_name = "Xenova/bge-large-zh-v1.5"
    backend = "onnx"
    model_kwargs = {"file_name": "model_fp16.onnx"}
    embedder = SentfmEmbedder(
        model_name, verbose=True, backend=backend, model_kwargs=model_kwargs
    )

    embedder.load_model()
    test_embedder(
        embedder,
        # query_prefix="为这个句子生成表示以用于检索相关文章：",
        query_prefix="",
        passage_prefix="",
    )


def test_e5():
    model_name = "intfloat/multilingual-e5-large"
    embedder = SentfmEmbedder(model_name, verbose=True)
    embedder.load_model()
    test_embedder(embedder, query_prefix="query:", passage_prefix="passage:")


if __name__ == "__main__":
    from models.embedders.tests import test_embedder

    # test_e5()
    test_bge_zh()

    # python -m models.sentfm.embed
