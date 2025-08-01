import models.sentencepiece.sentencepiece_model_pb2 as spm_pb2

from pathlib import Path
from tclogger import logger
from typing import Union


class SentencePieceModelProtor:
    def __init__(self, model_path: Union[str, Path] = None, verbose: bool = False):
        self.model_path = model_path
        self.verbose = verbose

    def load_model(self, model_path: Union[str, Path] = None) -> spm_pb2.ModelProto:
        logger.note(f"> Load sentencepiece model from:", verbose=self.verbose)
        model_path = model_path or self.model_path
        self.model = spm_pb2.ModelProto()
        with open(model_path, "rb") as f:
            self.model.ParseFromString(f.read())
        logger.file(f"  * {model_path}", verbose=self.verbose)
        return self.model

    def save_model(
        self, model: spm_pb2.ModelProto = None, model_path: Union[str, Path] = None
    ):
        logger.note(f"> Save sentencepiece model to:", verbose=self.verbose)
        model = model or self.model
        model_path = model_path or self.model_path
        with open(model_path, "wb") as f:
            f.write(model.SerializeToString())
        logger.file(f"  * {model_path}", verbose=self.verbose)

    def save_vocab(
        self, model: spm_pb2.ModelProto = None, vocab_path: Union[str, Path] = None
    ):
        logger.note(f"> Save sentencepiece vocab to:", verbose=self.verbose)
        model = model or self.model
        vocab_path = vocab_path or Path(self.model_path).with_suffix(".vocab")
        vocab_str = ""
        for piece in model.pieces:
            vocab_str += f"{piece.piece}\t{piece.score}\n"
        with open(vocab_path, "w", encoding="utf-8") as wf:
            wf.write(vocab_str)
        logger.file(f"  * {vocab_path}", verbose=self.verbose)
