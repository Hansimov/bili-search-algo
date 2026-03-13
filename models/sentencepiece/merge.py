import argparse
import json
import re
import sys

import models.sentencepiece.sentencepiece_model_pb2 as spm_pb2

from dataclasses import dataclass, field
from pathlib import Path
from tclogger import logger, logstr, brk, chars_len
from typing import Union

from models.sentencepiece.proto import SentencePieceModelProtor
from models.sentencepiece.vocab_filters import build_token_profile
from configs.envs import SENTENCEPIECE_CKPT_ROOT
from data_utils.videos.filter import REGION_MONGO_FILTERS

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

SPECIAL_CHAR_SCORES = {"游": 100.0, "影": 100.0, "学": 100.0, "▂": 80.0}


@dataclass
class ModelSource:
    name: str
    path: Path
    vocab_size: int
    trunc_count: int
    is_wiki: bool


@dataclass
class PieceAggregate:
    token: str
    rank_sum: float = 0.0
    norm_score_sum: float = 0.0
    best_rank: float = 0.0
    occurrences: int = 0
    sources: set[str] = field(default_factory=set)
    video_sources: set[str] = field(default_factory=set)
    wiki_sources: set[str] = field(default_factory=set)

    def add(
        self,
        source: ModelSource,
        rank_score: float,
        normalized_score: float,
    ):
        self.rank_sum += rank_score
        self.norm_score_sum += normalized_score
        self.best_rank = max(self.best_rank, rank_score)
        self.occurrences += 1
        self.sources.add(source.name)
        if source.is_wiki:
            self.wiki_sources.add(source.name)
        else:
            self.video_sources.add(source.name)

    @property
    def source_count(self) -> int:
        return len(self.sources)

    @property
    def video_support(self) -> int:
        return len(self.video_sources)

    @property
    def wiki_support(self) -> int:
        return len(self.wiki_sources)

    @property
    def avg_rank(self) -> float:
        return self.rank_sum / self.occurrences if self.occurrences else 0.0

    @property
    def avg_norm_score(self) -> float:
        return self.norm_score_sum / self.occurrences if self.occurrences else 0.0


class SentencePieceModelMerger:
    def __init__(
        self,
        model_paths: list[Union[str, Path]],
        output_path: Union[str, Path],
        max_vocab_size: int = None,
        trunc_ratio: float = 0.9,
        max_cjk_char_len: int = 8,
        min_ascii_video_support: int = 1,
        min_ascii_source_support: int = 2,
        min_ascii_len: int = 4,
        export_stats: bool = False,
        verbose: bool = False,
    ):
        self.model_paths = model_paths
        self.output_path = output_path
        self.max_vocab_size = max_vocab_size
        self.trunc_ratio = trunc_ratio
        self.max_cjk_char_len = max_cjk_char_len
        self.min_ascii_video_support = min_ascii_video_support
        self.min_ascii_source_support = min_ascii_source_support
        self.min_ascii_len = min_ascii_len
        self.export_stats = export_stats
        self.protors: list[SentencePieceModelProtor] = []
        self.sources: list[ModelSource] = []
        self.verbose = verbose

    def load_models(self):
        logger.note("> Load sentencepiece models:", verbose=self.verbose)
        for idx, path in enumerate(self.model_paths):
            protor = SentencePieceModelProtor(path, verbose=False)
            protor.load_model()
            self.protors.append(protor)
            vocab_size = len(protor.model.pieces)
            trunc_count = max(1, int(vocab_size * self.trunc_ratio))
            self.sources.append(
                ModelSource(
                    name=Path(path).stem,
                    path=Path(path),
                    vocab_size=vocab_size,
                    trunc_count=trunc_count,
                    is_wiki="wiki" in Path(path).stem,
                )
            )
            idx_str = logstr.mesg(brk(str(idx + 1)))
            logger.file(f"  * {idx_str} {path}", verbose=self.verbose)

    def is_prune(self, token: str) -> bool:
        if re.match(RE_START_WITH_DE, token):
            return not re.fullmatch(RE_VALID_DE, token)
        if re.match(RE_START_WITH_LE, token):
            return not re.fullmatch(RE_VALID_LE, token)
        profile = build_token_profile(token)
        if profile.cjk_char_len > self.max_cjk_char_len:
            return True
        if profile.is_pure_digits or profile.malformed:
            return True
        return False

    def should_drop_aggregate(self, aggregate: PieceAggregate) -> bool:
        token = aggregate.token
        profile = build_token_profile(token)

        if self.is_prune(token):
            return True

        if profile.is_ascii_token:
            if aggregate.video_support == 0:
                return True
            if len(token) < self.min_ascii_len and (
                aggregate.video_support < self.min_ascii_video_support + 1
            ):
                return True
            if aggregate.source_count < self.min_ascii_source_support and (
                aggregate.best_rank < 0.995 or aggregate.video_support == 0
            ):
                return True
            if profile.has_connector and aggregate.video_support < 2:
                return True

        return False

    def calc_piece_score(self, aggregate: PieceAggregate) -> float:
        profile = build_token_profile(aggregate.token)
        video_models_count = max(1, sum(not source.is_wiki for source in self.sources))
        model_count = max(1, len(self.sources))
        video_coverage = aggregate.video_support / video_models_count
        source_coverage = aggregate.source_count / model_count

        quality_bonus = 0.0
        if profile.has_cjk:
            quality_bonus += 0.9
        elif profile.is_ascii_token:
            quality_bonus -= 0.35
            if len(profile.token) >= 6:
                quality_bonus += 0.1
            if profile.is_ascii_alpha and aggregate.video_support >= 3:
                quality_bonus += 0.15

        quality_bonus += min(aggregate.video_support, 4) * 0.2
        quality_bonus += min(aggregate.wiki_support, 1) * 0.03

        return (
            aggregate.avg_rank * 5.0
            + aggregate.avg_norm_score * 2.0
            + video_coverage * 2.5
            + source_coverage * 1.5
            + quality_bonus
        )

    def export_stats_file(self):
        if not self.export_stats:
            return
        stats_path = Path(self.output_path).with_suffix(".stats.jsonl")
        logger.note("> Export merge stats to:")
        logger.file(f"  * {stats_path}")
        with open(stats_path, "w", encoding="utf-8") as wf:
            for item in self.merged_stats:
                wf.write(json.dumps(item, ensure_ascii=False) + "\n")

    def merge_vocabs(self):
        logger.note("> Merge sentencepiece models:", verbose=self.verbose)
        aggregates: dict[str, PieceAggregate] = {}
        vocab_count_total = 0
        pruned_count = 0
        dropped_count = 0

        # merge vocab pieces from all models
        for protor, source in zip(self.protors, self.sources):
            model = protor.model
            vocab_size = len(model.pieces)
            vocab_size_str = logstr.mesg(brk(len(model.pieces)))
            vocab_count_total += len(model.pieces)
            logger.file(f"  * {protor.model_path}", verbose=self.verbose)
            logger.file(f"    * vocab size: {vocab_size_str}", verbose=self.verbose)
            trunc_count = source.trunc_count
            truncated_pieces = model.pieces[:trunc_count]
            min_score = min(piece.score for piece in truncated_pieces)
            max_score = max(piece.score for piece in truncated_pieces)
            score_span = max(max_score - min_score, 1e-6)
            for idx, piece in enumerate(truncated_pieces):
                piece_str = piece.piece
                if self.is_prune(piece_str):
                    pruned_count += 1
                    continue
                rank_score = 1.0 - (idx / max(1, trunc_count - 1))
                normalized_score = (piece.score - min_score) / score_span
                aggregate = aggregates.setdefault(piece_str, PieceAggregate(piece_str))
                aggregate.add(
                    source, rank_score=rank_score, normalized_score=normalized_score
                )

        # log merged vocab size and duplicated vocab count
        self.merged_vocab_size = len(aggregates)
        merged_vocab_size_str = logstr.mesg(brk(self.merged_vocab_size))
        pruned_count_str = logstr.mesg(brk(pruned_count))
        logger.warn(f"  - pruned vocab count: {pruned_count_str}")
        logger.success(f"  + merged vocab size: {merged_vocab_size_str}")

        merged_pieces = []
        self.merged_stats = []
        for token, aggregate in aggregates.items():
            if self.should_drop_aggregate(aggregate):
                dropped_count += 1
                continue
            piece = spm_pb2.ModelProto.SentencePiece()
            piece.piece = token
            piece.score = self.calc_piece_score(aggregate)
            merged_pieces.append(piece)
            profile = build_token_profile(token)
            self.merged_stats.append(
                {
                    "token": token,
                    "score": piece.score,
                    "avg_rank": aggregate.avg_rank,
                    "avg_norm_score": aggregate.avg_norm_score,
                    "source_count": aggregate.source_count,
                    "video_support": aggregate.video_support,
                    "wiki_support": aggregate.wiki_support,
                    "has_cjk": profile.has_cjk,
                    "is_ascii_token": profile.is_ascii_token,
                }
            )

        logger.warn(
            f"  - dropped aggregated vocab count: {logstr.warn(brk(dropped_count))}"
        )

        for key, score in SPECIAL_CHAR_SCORES.items():
            if any(piece.piece == key for piece in merged_pieces):
                for piece in merged_pieces:
                    if piece.piece == key:
                        piece.score = max(piece.score, score)
                        break
            else:
                piece = spm_pb2.ModelProto.SentencePiece()
                piece.piece = key
                piece.score = score
                merged_pieces.append(piece)
                self.merged_stats.append(
                    {
                        "token": key,
                        "score": score,
                        "avg_rank": 1.0,
                        "avg_norm_score": 1.0,
                        "source_count": 0,
                        "video_support": 0,
                        "wiki_support": 0,
                        "has_cjk": True,
                        "is_ascii_token": False,
                    }
                )

        logger.note("> Sort merged pieces by score ...", verbose=self.verbose)
        self.merged_pieces_list = sorted(
            merged_pieces,
            key=lambda x: (x.score, chars_len(x.piece), x.piece),
            reverse=True,
        )
        self.merged_stats.sort(key=lambda x: x["score"], reverse=True)

        if self.max_vocab_size and self.merged_vocab_size > self.max_vocab_size:
            max_voaab_size_str = logstr.mesg(brk(self.max_vocab_size))
            logger.note(f"> Keep top vocabs: {max_voaab_size_str}")
            self.merged_pieces_list = self.merged_pieces_list[: self.max_vocab_size]
            self.merged_stats = self.merged_stats[: self.max_vocab_size]

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
        self.export_stats_file()


class MergerArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("-i", "--input-prefix", type=str, default="sp_908m_")
        self.add_argument("-o", "--output-prefix", type=str, default="sp_merged")
        self.add_argument("-vs", "--vocab-size", type=int, default=None)
        self.add_argument("-tr", "--trunc-ratio", type=float, default=0.9)
        self.add_argument("-mc", "--max-cjk-char-len", type=int, default=8)
        self.add_argument("-av", "--min-ascii-video-support", type=int, default=1)
        self.add_argument("-as", "--min-ascii-source-support", type=int, default=2)
        self.add_argument("-al", "--min-ascii-len", type=int, default=4)
        self.add_argument("-es", "--export-stats", action="store_true")

    def parse_args(self):
        self.args, self.unknown_args = self.parse_known_args(sys.argv[1:])
        return self.args


def main(args: argparse.Namespace):
    input_model_prefixes = ["sp_wiki_8m_400k"]
    input_model_prefixes.extend(
        [
            f"{args.input_prefix}{suffix}"
            for suffix in REGION_MONGO_FILTERS.keys()
            if suffix not in ["test"]
        ]
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
        trunc_ratio=args.trunc_ratio,
        max_cjk_char_len=args.max_cjk_char_len,
        min_ascii_video_support=args.min_ascii_video_support,
        min_ascii_source_support=args.min_ascii_source_support,
        min_ascii_len=args.min_ascii_len,
        export_stats=args.export_stats,
        verbose=True,
    )
    merger.merge()


if __name__ == "__main__":
    arg_parser = MergerArgParser()
    args = arg_parser.parse_args()
    main(args)

    # ANCHOR[id=sp-merge]

    # Backup old model
    # cd ~/repos/bili-search-algo/models/sentencepiece/checkpoints
    # cp sp_merged.model sp_merged_750m.model && cp sp_merged.vocab sp_merged_750m.vocab

    # Merge with prefix "sp_908m_"
    # cd ~/repos/bili-search-algo
    # python -m models.sentencepiece.merge -vs 1000000 -i sp_908m_ -o sp_merged -es

    # Convert .vocab to .txt
    # See: # LINK: models/sentencepiece/convert.py#sp-convert

    # Copy to btok (sp_merged.* -> sp.*)
    # for f in ~/repos/bili-search-algo/models/sentencepiece/checkpoints/sp_merged.*; do cp "$f" ~/repos/btok/src/btok/sp.${f##*.}; done
    # cp ~/repos/btok/src/btok/sp.txt ~/elasticsearch-docker-9.2.4-pro/plugins/es01/es_tok/vocabs.txt

    # Test
    # python ~/repos/btok/tests.py
    # vi ~/repos/btok/src/btok/sp.txt
