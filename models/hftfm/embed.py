import numpy as np
import os
import torch

from tclogger import logger, logstr, PathType, StrsType, Runtimer
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from typing import Literal
from torch.ao.quantization import quantize_dynamic

# suppress warnings from TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def resize_linear_layer(
    linear_layer: torch.nn.Linear,
    new_in_features: int = None,
    new_out_features: int = None,
):
    old_weight = linear_layer.weight.data
    old_bias = linear_layer.bias.data if linear_layer.bias is not None else None

    if new_in_features is None:
        new_in_features = linear_layer.in_features

    new_linear_layer = torch.nn.Linear(
        new_in_features,
        new_out_features,
        bias=linear_layer.bias is not None,
        device=linear_layer.weight.device,
        dtype=linear_layer.weight.dtype,
    )

    weight_out_size = min(new_out_features, old_weight.size(0))
    weight_in_size = min(new_in_features, old_weight.size(1))

    new_linear_layer.weight.data = old_weight[:weight_out_size, :weight_in_size].clone()

    if old_bias is not None:
        new_linear_layer.bias.data = old_bias[:weight_out_size].clone()

    new_linear_layer = new_linear_layer.to(linear_layer.weight.device)

    return new_linear_layer


class HFTransformersEmbedder:
    def __init__(
        self,
        model_name: str = None,
        model_path: PathType = None,
        device: Literal["cuda", "cpu"] = "cuda",
        use_quantize: bool = False,
        use_layer_prune: bool = False,
        layer_prune_ratio: float = 0.75,
        use_attention_prune: bool = False,
        attention_prune_ratio: float = 0.75,
        model_kwargs: dict = None,
        verbose: bool = False,
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self.use_quantize = use_quantize
        self.use_layer_prune = use_layer_prune
        self.layer_prune_ratio = layer_prune_ratio
        self.use_attention_prune = use_attention_prune
        self.attention_prune_ratio = attention_prune_ratio
        self.model_kwargs = model_kwargs or None
        self.verbose = verbose

    def set_device(self):
        if self.device == "cuda":
            device_map = "auto"
            torch_dtype = torch.float16
        else:
            device_map = None
            torch_dtype = torch.float32

        self.device_kwargs = {
            "device_map": device_map,
            "torch_dtype": torch_dtype,
        }

    def set_gpu_quantize(self):
        if self.use_quantize and self.device == "cuda":
            logger.mesg(f"  * set gpu quantize ...")
            self.quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
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

    def set_cpu_quantize(self):
        if self.use_quantize and self.device == "cpu":
            logger.mesg(f"  * set cpu quantize ...")
            self.model = self.model.to(torch.float32)
            self.model = quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8, inplace=False
            )

    def has_layer(self) -> bool:
        return hasattr(self.model, "encoder") and hasattr(self.model.encoder, "layer")

    def set_layer_prune(self):
        if not (self.use_layer_prune and self.has_layer()):
            return
        if not self.layer_prune_ratio:
            return
        old_layers = len(self.model.encoder.layer)
        new_layers = max(1, int(old_layers * self.layer_prune_ratio))
        if new_layers < old_layers:
            self.model.encoder.layer = self.model.encoder.layer[-new_layers:]

        if self.verbose:
            if new_layers < old_layers:
                layers_str = f"{old_layers} -> {new_layers}"
            else:
                layers_str = f"{new_layers}"
            logger.mesg(f"  * layers: {logstr.file(layers_str)}")

    def has_attention(self, layer: torch.nn.Module) -> bool:
        return hasattr(layer, "attention") and hasattr(layer.attention, "self")

    def set_attention_prune(self):
        if not (self.use_attention_prune and self.has_layer()):
            return
        if not self.attention_prune_ratio:
            return
        if self.device == "cuda":
            logger.warn("  * skip attention prune for cuda device")
            return

        for layer in self.model.encoder.layer:
            if not self.has_attention(layer):
                continue

            old_heads = layer.attention.self.num_attention_heads
            new_heads = max(1, int(old_heads * self.attention_prune_ratio))

            if new_heads < old_heads:
                head_size = layer.attention.self.attention_head_size
                new_all_head_size = new_heads * head_size
                hidden_size = layer.attention.self.query.in_features

                # adjust attention layers to new number of heads
                layer.attention.self.num_attention_heads = new_heads
                layer.attention.self.all_head_size = new_all_head_size

                # adjust Q, K, V layers to new all_head_size (output features)
                # but keep the same input features (hidden_size)
                new_features_params = {
                    "new_in_features": hidden_size,
                    "new_out_features": new_all_head_size,
                }
                layer.attention.self.query = resize_linear_layer(
                    layer.attention.self.query, **new_features_params
                )
                layer.attention.self.key = resize_linear_layer(
                    layer.attention.self.key, **new_features_params
                )
                layer.attention.self.value = resize_linear_layer(
                    layer.attention.self.value, **new_features_params
                )

                # adjust dense layer input dim to new_all_head_size,
                # keeps output dim as old_hidden_size
                old_hidden_size = layer.attention.output.dense.out_features
                layer.attention.output.dense = resize_linear_layer(
                    layer.attention.output.dense,
                    new_in_features=new_all_head_size,
                    new_out_features=old_hidden_size,
                )

                if self.verbose:
                    heads_str = f"{old_heads} -> {new_heads}"
                    logger.mesg(f"    * attention heads: {logstr.file(heads_str)}")

    def log_model_loading(self):
        if self.verbose:
            logger.note(f"> Loading model:")
            if self.model_name:
                logger.mesg(f"  * model_name: {logstr.file(self.model_name)}")
            if self.model_path:
                logger.mesg(f"  * model_path: {logstr.file(self.model_path)}")

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
        self.set_gpu_quantize()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, use_fast=True
        )
        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            **self.device_kwargs,
            **self.quantize_kwargs,
            **model_kwargs,
        )
        self.set_layer_prune()
        self.set_attention_prune()
        self.set_cpu_quantize()

        self.model.eval()
        self.log_model_loaded()

    def embed(self, sentences: StrsType) -> np.ndarray:
        encoded_input = self.tokenizer(
            sentences,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        with torch.inference_mode():
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
        use_quantize=True,
        use_layer_prune=True,
        layer_prune_ratio=0.5,
        use_attention_prune=True,
        attention_prune_ratio=0.5,
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

    with Runtimer():
        test_hftfm_bge_zh()

    # python -m models.hftfm.embed
