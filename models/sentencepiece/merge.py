import argparse
import re
import sys

import models.sentencepiece.sentencepiece_model_pb2 as spm_pb2

from pathlib import Path
from tclogger import logger, logstr, brk, chars_len
from typing import Union

from data_utils.videos.convert import CH_CJK
from models.sentencepiece.proto import SentencePieceModelProtor
from configs.envs import SENTENCEPIECE_CKPT_ROOT
from models.sentencepiece.filter import REGION_MONGO_FILTERS

RE_START_WITH_DE = rf"^的"
CH_VALID_DE = r"确士卢"
RE_VALID_DE = rf"^的[{CH_VALID_DE}]?$"

RE_START_WITH_LE = rf"^了"
CH_VALID_LE = r"断解得结却然如若无事悟了不"
WD_VALID_LE = "不得|不起(的.+)?"
RE_VALID_LE = rf"^了([{CH_VALID_LE}]|{WD_VALID_LE})?$"

RE_START_WITH_HE = rf"^和"
RE_START_WITH_ZAI = rf"^在"
RE_START_WITH_SHI = rf"^是"
RE_START_WITH_YU = rf"^与"
RE_START_WITH_JI = rf"^及"
RE_START_WITH_BING = rf"^并"

SPECIAL_CHAR_SCORES = {"游": -8.0, "影": -8.0, "学": -8.0, "▂": 0}

RE_CH_CJK = rf"[{CH_CJK}]"
PT_CH_CJK = re.compile(RE_CH_CJK)


def calc_cjk_char_len(token: str) -> int:
    return sum(1 for char in token if PT_CH_CJK.match(char))


class SentencePieceModelMerger:
    def __init__(
        self,
        model_paths: list[Union[str, Path]],
        output_path: Union[str, Path],
        max_vocab_size: int = None,
        max_cjk_char_len: int = 8,
        verbose: bool = False,
    ):
        self.model_paths = model_paths
        self.output_path = output_path
        self.max_vocab_size = max_vocab_size
        self.max_cjk_char_len = max_cjk_char_len
        self.protors: list[SentencePieceModelProtor] = []
        self.verbose = verbose

    def load_models(self):
        logger.note("> Load sentencepiece models:", verbose=self.verbose)
        for idx, path in enumerate(self.model_paths):
            protor = SentencePieceModelProtor(path, verbose=False)
            protor.load_model()
            self.protors.append(protor)
            idx_str = logstr.mesg(brk(str(idx + 1)))
            logger.file(f"  * {idx_str} {path}", verbose=self.verbose)

    def is_prune(self, token: str) -> bool:
        if re.match(RE_START_WITH_DE, token):
            return not re.fullmatch(RE_VALID_DE, token)
        if re.match(RE_START_WITH_LE, token):
            return not re.fullmatch(RE_VALID_LE, token)
        if calc_cjk_char_len(token) > self.max_cjk_char_len:
            return True

    def merge_vocabs(self):
        logger.note("> Merge sentencepiece models:", verbose=self.verbose)
        merged_pieces = {}
        vocab_count_total = 0
        pruned_count = 0
        duplicated_count = 0

        # merge vocab pieces from all models
        for protor in self.protors:
            model = protor.model
            vocab_size_str = logstr.mesg(brk(len(model.pieces)))
            vocab_count_total += len(model.pieces)
            logger.file(f"  * {protor.model_path}", verbose=self.verbose)
            logger.file(f"    * vocab size: {vocab_size_str}", verbose=self.verbose)
            for piece in model.pieces:
                piece_str = piece.piece
                if self.is_prune(piece_str):
                    pruned_count += 1
                    continue
                if piece_str not in merged_pieces:
                    merged_pieces[piece_str] = piece
                else:
                    duplicated_count += 1
                    merged_pieces[piece_str].score = max(
                        merged_pieces[piece_str].score, piece.score
                    )

        # log merged vocab size and duplicated vocab count
        self.merged_vocab_size = len(merged_pieces)
        merged_vocab_size_str = logstr.mesg(brk(self.merged_vocab_size))
        duplicated_count_str = logstr.mesg(brk(duplicated_count))
        pruned_count_str = logstr.mesg(brk(pruned_count))
        logger.warn(f"  - pruned vocab count: {pruned_count_str}")
        logger.warn(f"  - duplicated vocab count: {duplicated_count_str}")
        logger.success(f"  + merged vocab size: {merged_vocab_size_str}")

        # smooth score for special chars
        for key, score in SPECIAL_CHAR_SCORES.items():
            if key in merged_pieces:
                merged_pieces[key].score = score
            else:
                piece = spm_pb2.ModelProto.SentencePiece()
                piece.piece = key
                piece.score = score
                merged_pieces[key] = piece

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


class MergerArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("-i", "--input-prefix", type=str, default="sp_507m_")
        self.add_argument("-o", "--output-prefix", type=str, default="sp_merged")
        self.add_argument("-vs", "--vocab-size", type=int, default=None)

    def parse_args(self):
        self.args, self.unknown_args = self.parse_known_args(sys.argv[1:])
        return self.args


if __name__ == "__main__":
    arg_parser = MergerArgParser()
    args = arg_parser.parse_args()
    # input_model_prefixes = [
    #     "sp_wiki_all_400k_0.9995",
    #     "sp_480m_400k_0.9995_0.9.model",
    # ]
    input_model_prefixes = ["sp_wiki_8m_400k"]
    input_model_prefixes.extend(
        [f"{args.input_prefix}{suffix}" for suffix in REGION_MONGO_FILTERS.keys()]
    )
    input_model_paths = [
        SENTENCEPIECE_CKPT_ROOT / f"{prefix}.model" for prefix in input_model_prefixes
    ]
    input_model_paths = [path for path in input_model_paths if path.exists()]

    output_model_path = SENTENCEPIECE_CKPT_ROOT / f"{args.output_prefix}.model"

    merger = SentencePieceModelMerger(
        input_model_paths,
        output_model_path,
        max_vocab_size=args.vocab_size,
        verbose=True,
    )
    merger.merge()

    # Backup old model
    # cd ~/repos/bili-search-algo/models/sentencepiece/checkpoints
    # cp sp_merged.model sp_merged_518m.model && cp sp_merged.vocab sp_merged_518m.vocab

    # Merge with prefix
    # python -m models.sentencepiece.merge -vs 1000000 -i sp_518m_ -o sp_merged
    # python -m models.sentencepiece.merge -vs 1000000 -i sp_575m_ -o sp_merged
    # cp ~/repos/bili-search-algo/models/sentencepiece/checkpoints/sp_merged.model ~/repos/btok/src/btok/sp.model

    # Test
    # python -m models.sentencepiece.train -m sp_merged_518m -t
    # python -m models.sentencepiece.train -m sp_merged -t
