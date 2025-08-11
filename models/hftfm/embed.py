import numpy as np
import os
import torch

from tclogger import logger, logstr, PathType, StrsType
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from typing import Literal

# suppress warnings from TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class HFTransformersEmbedder:
    def __init__(
        self,
        model_name: str = None,
        model_path: PathType = None,
        device: Literal["cuda", "cpu"] = "cuda",
        use_quantize: bool = False,
        use_prune: bool = False,
        prune_ratio: float = 0.75,
        model_kwargs: dict = None,
        verbose: bool = False,
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self.use_prune = use_prune
        self.prune_ratio = prune_ratio
        self.use_quantize = use_quantize
        self.model_kwargs = model_kwargs or None
        self.verbose = verbose

    def set_device(self):
        if self.device == "cuda":
            device_map = "auto"
            torch_dtype = torch.float16
        else:
            device_map = None
            torch_dtype = torch.float16

        self.device_kwargs = {
            "device_map": device_map,
            "torch_dtype": torch_dtype,
        }

    def set_quantize(self):
        if self.use_quantize:
            self.quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="fp4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            self.low_cpu_mem_usage = True
        else:
            self.quantization_config = None
            self.low_cpu_mem_usage = False

        self.quantize_kwargs = {
            "quantization_config": self.quantization_config,
            "low_cpu_mem_usage": self.low_cpu_mem_usage,
        }

    def set_prune(self):
        if (
            self.use_prune
            and hasattr(self.model, "encoder")
            and hasattr(self.model.encoder, "layer")
        ):
            old_layers = len(self.model.encoder.layer)
            if self.prune_ratio:
                new_layers = max(1, int(old_layers * self.prune_ratio))
            else:
                new_layers = old_layers
            if new_layers < old_layers:
                self.model.encoder.layer = self.model.encoder.layer[-new_layers:]

            if self.verbose:
                if new_layers < old_layers:
                    layers_str = f"{old_layers} -> {new_layers}"
                else:
                    layers_str = f"{new_layers}"
                logger.mesg(f"  * layers : {logstr.file(layers_str)}")

    def log_model_loading(self):
        if self.verbose:
            logger.note(f"> Loading model:")
            if self.model_name:
                logger.mesg(f"  * model_name  : {logstr.file(self.model_name)}")
            if self.model_path:
                logger.mesg(f"  * model_path  : {logstr.file(self.model_path)}")

    def log_model_loaded(self):
        if self.verbose:
            model_params_size = self.model.num_parameters()
            model_params_size_m = round(model_params_size / 1e6, 0)
            model_params_str = f"{model_params_size_m:.0f}M"
            logger.mesg(f"  * model_params: {logstr.file(model_params_str)}")

    def load_model(self):
        self.log_model_loading()

        model_name_or_path = self.model_name or self.model_path
        model_kwargs = self.model_kwargs or {}
        self.set_device()
        self.set_quantize()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, use_fast=True
        )

        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            **self.device_kwargs,
            **self.quantize_kwargs,
            **model_kwargs,
        )
        self.set_prune()
        self.model.eval()

        self.log_model_loaded()

    def embed(self, sentences: StrsType) -> np.ndarray:
        encoded_input = self.tokenizer(
            sentences,
            padding=True,
            truncation=False,
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
        use_quantize=False,
        use_prune=True,
        prune_ratio=0.5,
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
