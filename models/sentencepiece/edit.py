import re
import models.sentencepiece.sentencepiece_model_pb2 as spm_pb2

from pathlib import Path
from tclogger import logger, logstr, brk
from typing import Union

from configs.envs import SENTENCEPIECE_CKPT_ROOT
from models.sentencepiece.proto import SentencePieceModelProtor

RE_DIGITS_PURE = r"^\d+$"
RE_DIGITS_CJK = r"^\d+[^\da-z]+$"
RE_EXCLUDE_STRS = r"%|[\.\-]{2,}"
CH_MASK = r"▂"
STRIPED_STRS = [".", CH_MASK, "-"]

PT_DIGIT_PURE = re.compile(RE_DIGITS_PURE)
PT_DIGIT_CJK = re.compile(RE_DIGITS_CJK)
PT_EXCLUDE_STR = re.compile(RE_EXCLUDE_STRS)


class SentencePieceModelVocabEditor:
    def __init__(self, model_path: Union[str, Path], verbose: bool = False):
        self.model_path = model_path
        self.protor = SentencePieceModelProtor(model_path)
        self.verbose = verbose

    def load_model(self):
        self.model = self.protor.load_model()

    def save_model(self):
        self.protor.save_model(model=self.model)
        self.protor.save_vocab(
            model=self.model, vocab_path=self.model_path.with_suffix(".vocabx")
        )

    def should_keep_concated_piece(self, piece) -> bool:
        piece_str = piece.piece
        if len(piece_str) < 2:
            return True
        for s in STRIPED_STRS:
            if piece_str.startswith(s) or piece_str.endswith(s):
                return False
        if PT_EXCLUDE_STR.search(piece_str):
            return False
        return True

    def trunc_pieces(self, pieces):
        """drop tail pieces whose str size is 1, which are low-quality pieces initialized in training."""
        trunc_idx = len(pieces)
        for idx in range(len(pieces) - 1, 0, -1):
            if len(pieces[idx].piece) > 1:
                break
            trunc_idx = idx
        return pieces[:trunc_idx]

    def remove_bad_pieces(self) -> spm_pb2.ModelProto:
        logger.note("> Remove bad pieces:", verbose=self.verbose)
        old_vocab_size = len(self.model.pieces)
        new_pieces = []
        trunced_pieces = self.trunc_pieces(self.model.pieces)
        for piece in trunced_pieces:
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
            logger.mesg(f"  * Old vocab size: {logstr.file(brk(old_vocab_size))}")
            logger.mesg(f"  - Removed vocabs: {logstr.warn(brk(removed_count))}")
            logger.mesg(f"  * New vocab size: {logstr.okay(brk(new_vocab_size))}")

    def edit(self):
        self.load_model()
        self.remove_bad_pieces()
        self.save_model()


if __name__ == "__main__":
    model_path = SENTENCEPIECE_CKPT_ROOT / "sp_wiki_8m_400k.model"
    editor = SentencePieceModelVocabEditor(model_path, verbose=True)
    editor.edit()

    # python -m models.sentencepiece.edit
