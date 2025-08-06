import numpy as np

from sentence_transformers import SentenceTransformer
from tclogger import logger, logstr, PathType, brk, norm_path, strf_path
from typing import Union, Literal

"""
Install dependencies:

```sh
pip install --upgrade torch torchvision
pip install --upgrade tensorrt
pip install --upgrade 'sentence-transformers[onnx-gpu]'
```

Download ONNX model manually if the auto-download fails:
- https://huggingface.co/Xenova/bge-base-zh-v1.5

```sh
export HF_ENDPOINT=https://alpha.hf-mirror.com
export CURRENT_MODEL="Xenova/bge-base-zh-v1.5"
hf auth login
hf download "$CURRENT_MODEL" --include "*.json" "*.txt" "onnx/model_int8.onnx"
# ~/downloads/hfd.sh "$CURRENT_MODEL" --local-dir "/home/<username>/.cache/huggingface/hub/models--Xenova--bge-base-zh-v1.5" --include "*.json" "*.txt" "onnx/model_int8.onnx"
```

"""


class SentfmEmbedder:
    def __init__(
        self,
        model_name: str = None,
        model_path: PathType = None,
        device: Literal["cuda", "cpu"] = "cuda",
        backend: Literal["torch", "onnx"] = "torch",
        model_kwargs: dict = None,
        verbose: bool = False,
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self.backend = backend
        self.model_kwargs = model_kwargs or None
        self.verbose = verbose

    def load_model(self):
        if self.verbose:
            logger.note(f"> Loading model:")
            if self.model_name:
                logger.file(f"  * model_name: {self.model_name}")
            if self.model_path:
                logger.file(f"  * model_path: {self.model_path}")

        self.model = SentenceTransformer(
            self.model_name or self.model_path,
            device=self.device,
            backend=self.backend,
            model_kwargs=self.model_kwargs,
        )

    def embed(self, sentences: Union[str, list[str]]) -> np.ndarray:
        return self.model.encode(sentences)


def test_bge_zh():
    model_name = "BAAI/bge-base-zh-v1.5"
    model_path = None
    model_kwargs = None

    # model_name = "BAAI/bge-large-zh-v1.5"
    # # embedder = SentfmEmbedder(model_name, verbose=True)

    # # model_name = "Xenova/bge-base-zh-v1.5"
    # model_path = strf_path("~/.cache/huggingface/hub/models--Xenova--bge-base-zh-v1.5")
    # model_kwargs = {"file_name": "model.onnx"}

    embedder = SentfmEmbedder(
        model_name=model_name,
        model_path=model_path,
        device="cpu",
        backend="torch",
        model_kwargs=model_kwargs,
        verbose=True,
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
