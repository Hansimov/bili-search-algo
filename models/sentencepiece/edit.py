import re
import models.sentencepiece.sentencepiece_model_pb2 as spm_pb2

from pathlib import Path
from tclogger import logger, logstr, brk
from typing import Union

from models.sentencepiece.proto import SentencePieceModelProtor

RE_DIGITS_PURE = r"^\d+$"
RE_DIGITS_CJK = r"^\d+[^\da-z]+$"
CH_MASK = r"▂"

PT_DIGIT_PURE = re.compile(RE_DIGITS_PURE)
PT_DIGIT_CJK = re.compile(RE_DIGITS_CJK)


class SentencePieceModelVocabEditor:
    def __init__(self, model_path: Union[str, Path], verbose: bool = False):
        self.model_path = model_path
        self.protor = SentencePieceModelProtor(model_path)
        self.verbose = verbose

    def load_model(self):
        self.model = self.protor.load_model()

    def save_model(self):
        self.protor.save_model(model=self.model)

    def should_keep_concated_piece(self, piece) -> bool:
        piece_str = piece.piece
        if piece_str.startswith(CH_MASK) or piece_str.endswith(CH_MASK):
            return False
        if piece_str.startswith("-") or piece_str.endswith("-"):
            return False
        return True

    def remove_bad_pieces(self) -> spm_pb2.ModelProto:
        logger.note("> Remove bad pieces:", verbose=self.verbose)
        old_vocab_size = len(self.model.pieces)
        new_pieces = []
        for piece in self.model.pieces:
            if (
                not PT_DIGIT_PURE.match(piece.piece)
                and not PT_DIGIT_CJK.match(piece.piece)
                and self.should_keep_concated_piece(piece)
            ):
                new_pieces.append(piece)
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
        self.remove_bad_pieces()
        self.save_model()


if __name__ == "__main__":
    model_path = Path(__file__).parents[2] / "sp_100m_100k_no.model"
    editor = SentencePieceModelVocabEditor(model_path, verbose=True)
    editor.load_model()
    editor.remove_bad_pieces()
    editor.save_model()

    # python -m models.sentencepiece.edit
