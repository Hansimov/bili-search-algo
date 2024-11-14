import re

from pathlib import Path
from tclogger import logger, logstr, brk
from typing import Union

import models.sentencepiece.sentencepiece_model_pb2 as spm_pb2


class SentencePieceModelProtor:
    def __init__(self, model_path: Union[str, Path], verbose: bool = False):
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


class SentencePieceModelVocabEditor:
    RE_DIGITS_PURE = r"^\d+$"
    PT_DIGIT_PURE = re.compile(RE_DIGITS_PURE)

    def __init__(self, model_path: Union[str, Path], verbose: bool = False):
        self.model_path = model_path
        self.protor = SentencePieceModelProtor(model_path)
        self.verbose = verbose

    def load_model(self):
        self.model = self.protor.load_model()

    def save_model(self):
        self.protor.save_model(model=self.model)

    def remove_digits(self) -> spm_pb2.ModelProto:
        logger.note("> Remove pure digit tokens from vocab:", verbose=self.verbose)
        old_vocab_size = len(self.model.pieces)
        new_pieces = [
            piece
            for piece in self.model.pieces
            if not self.PT_DIGIT_PURE.match(piece.piece)
        ]
        self.model.ClearField("pieces")
        self.model.pieces.extend(new_pieces)
        new_vocab_size = len(self.model.pieces)
        if self.verbose:
            removed_count = old_vocab_size - new_vocab_size
            logger.mesg(f"  * Old vocab size: {logstr.warn(brk(old_vocab_size))}")
            logger.mesg(f"  * New vocab size: {logstr.success(brk(new_vocab_size))}")
            logger.mesg(f"  - Tokens removed: {logstr.file(brk(removed_count))}")

    def edit(self):
        self.load_model()
        self.remove_digits()
        self.save_model()


if __name__ == "__main__":
    model_path = Path(__file__).parents[2] / "sp_100m_100k_no.model"
    editor = SentencePieceModelVocabEditor(model_path, verbose=True)
    editor.load_model()
    editor.remove_digits()
    editor.save_model()

    # python -m models.sentencepiece.edit
