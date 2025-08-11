import numpy as np
import os
import torch

from transformers import AutoTokenizer, AutoModel
from tclogger import logger, logstr, PathType, StrsType
from typing import Union, Literal

# suppress warnings from TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class HFTransformersEmbedder:
    def __init__(
        self,
        model_name: str = None,
        model_path: PathType = None,
        device: Literal["cuda", "cpu"] = "cuda",
        model_kwargs: dict = None,
        verbose: bool = False,
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self.model_kwargs = model_kwargs or None
        self.verbose = verbose

    def load_model(self):
        if self.verbose:
            logger.note(f"> Loading model:")
            if self.model_name:
                logger.mesg(f"  * model_name  : {logstr.file(self.model_name)}")
            if self.model_path:
                logger.mesg(f"  * model_path  : {logstr.file(self.model_path)}")

        model_name_or_path = self.model_name or self.model_path

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, use_fast=True
        )

        if self.device == "cuda":
            device_map = "auto"
            torch_dtype = torch.float16
        else:
            device_map = None
            torch_dtype = torch.float32

        model_kwargs = self.model_kwargs or {}

        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            **model_kwargs,
        )

        if self.verbose:
            model_params_size = self.model.num_parameters()
            model_params_size_m = round(model_params_size / 1e6, 0)
            model_params_str = f"{model_params_size_m:.0f}M"
            logger.mesg(f"  * model_params: {logstr.file(model_params_str)}")

        self.model.eval()

    def embed(self, sentences: StrsType) -> np.ndarray:
        encoded_input = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            embeddings = model_output[0][:, 0]
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        embeddings_np = embeddings.cpu().numpy()

        if isinstance(sentences, str):
            embeddings_np = embeddings_np[0]
        return embeddings_np


def test_hftfm_bge_zh():
    model_name = "BAAI/bge-base-zh-v1.5"
    model_path = None
    model_kwargs = None

    embedder = HFTransformersEmbedder(
        model_name=model_name,
        model_path=model_path,
        device="cpu",
        model_kwargs=model_kwargs,
        verbose=True,
    )
    embedder.load_model()

    test_embedder(
        embedder,
        query_prefix="为这个句子生成表示以用于检索相关文章：",
        # query_prefix="",
        passage_prefix="",
    )


if __name__ == "__main__":
    from models.embedders.tests import test_embedder

    test_hftfm_bge_zh()

    # python -m models.hftfm.embed
