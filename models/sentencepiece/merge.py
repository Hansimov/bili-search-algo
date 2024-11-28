import models.sentencepiece.sentencepiece_model_pb2 as spm_pb2

from pathlib import Path
from tclogger import logger, logstr, brk, chars_len
from typing import Union

from models.sentencepiece.proto import SentencePieceModelProtor


class SentencePieceModelMerger:
    def __init__(
        self,
        model_paths: list[Union[str, Path]],
        output_path: Union[str, Path],
        max_vocab_size: int = None,
        verbose: bool = False,
    ):
        self.model_paths = model_paths
        self.output_path = output_path
        self.max_vocab_size = max_vocab_size
        self.protors: list[SentencePieceModelProtor] = []
        self.verbose = verbose

    def load_models(self):
        logger.note("> Load sentencepiece models:", verbose=self.verbose)
        for path in self.model_paths:
            protor = SentencePieceModelProtor(path, verbose=False)
            protor.load_model()
            self.protors.append(protor)
            logger.file(f"  * {path}", verbose=self.verbose)

    def merge_vocabs(self):
        logger.note("> Merge sentencepiece models:", verbose=self.verbose)
        merged_pieces = {}
        vocab_count_total = 0

        # merge vocab pieces from all models
        for protor in self.protors:
            model = protor.model
            vocab_size_str = logstr.mesg(brk(len(model.pieces)))
            vocab_count_total += len(model.pieces)
            logger.file(f"  * {protor.model_path}", verbose=self.verbose)
            logger.file(f"    * vocab size: {vocab_size_str}", verbose=self.verbose)
            for piece in model.pieces:
                if piece.piece not in merged_pieces:
                    merged_pieces[piece.piece] = piece
                else:
                    merged_pieces[piece.piece].score = max(
                        merged_pieces[piece.piece].score, piece.score
                    )

        # log merged vocab size and duplicated vocab count
        self.merged_vocab_size = len(merged_pieces)
        merged_vocab_size_str = logstr.mesg(brk(self.merged_vocab_size))
        duplicated_vocab_count = vocab_count_total - self.merged_vocab_size
        duplicated_vocab_count_str = logstr.mesg(brk(duplicated_vocab_count))
        logger.warn(f"  - duplicated vocab count: {duplicated_vocab_count_str}")
        logger.success(f"  + merged vocab size: {merged_vocab_size_str}")

        # sort pieces by score in desc order
        logger.note("> Sort merged pieces by score ...", verbose=self.verbose)
        self.merged_pieces_list = list(merged_pieces.values())
        self.merged_pieces_list.sort(key=lambda x: x.score, reverse=True)

        if self.max_vocab_size and self.merged_vocab_size > self.max_vocab_size:
            max_voaab_size_str = logstr.mesg(brk(self.max_vocab_size))
            logger.note(f"> Keep top vocabs: {max_voaab_size_str}")
            self.merged_pieces_list = self.merged_pieces_list[: self.max_vocab_size]

    def dump_merged_model(self) -> spm_pb2.ModelProto:
        merged_model = spm_pb2.ModelProto()
        merged_model.pieces.extend(self.merged_pieces_list)

        if self.protors:
            model = self.protors[0].model
            merged_model.trainer_spec.CopyFrom(model.trainer_spec)
            merged_model.normalizer_spec.CopyFrom(model.normalizer_spec)

        protor = SentencePieceModelProtor(self.output_path, verbose=self.verbose)
        protor.save_model(model=merged_model, model_path=self.output_path)

        return merged_model

    def export_vocab(self):
        output_path = Path(self.output_path).with_suffix(".vocab")
        logger.note(f"> Export vocab to:")
        logger.file(f"  * {output_path}")
        with open(output_path, "w", encoding="utf-8") as f:
            for piece in self.merged_pieces_list:
                piece_chars_len = chars_len(piece.piece)
                piece_str_len = max(0, 12 - piece_chars_len)
                piece_str = f"{piece.piece:<{piece_str_len}}\t{piece.score:.5f}\n"
                f.write(piece_str)

    def merge(self):
        self.load_models()
        self.merge_vocabs()
        self.dump_merged_model()
        self.export_vocab()


if __name__ == "__main__":
    input_model_paths = [
        "sp_wiki_all_400k_0.9995.model",
        "sp_480m_400k_0.9995_0.9.model",
    ]
    output_model_path = "sp_400k_merged.model"

    merger = SentencePieceModelMerger(
        input_model_paths, output_model_path, verbose=True
    )
    merger.merge()

    # python -m models.sentencepiece.merge
    # python -m models.sentencepiece.train -mp sp_400k_merged -t