import argparse
import re
import polars as pl

from pathlib import Path
from tclogger import PathType, PathsType, logger, logstr, brk

from configs.envs import SP_MERGED_MODEL_PATH
from models.word.eng import get_dump_path

CH_MASK = "▂"
RE_ONE_WORD = r"^[a-zA-Z]+$"
PT_ONE_WORD = re.compile(RE_ONE_WORD)
RE_ENG_WORD = r"^[a-zA-Z\s▂\-\_]+$"
PT_ENG_WORD = re.compile(RE_ENG_WORD)


class SentencePieceConverter:
    def __init__(self, model_path: PathType = SP_MERGED_MODEL_PATH):
        self.model_path = Path(model_path)
        self.vocab_path = self.model_path.with_suffix(".vocab")
        self.txt_path = self.model_path.with_suffix(".txt")

    def load_vocabs(self):
        logger.note(f"> Loading vocab:", end=" ")
        with open(self.vocab_path, "r", encoding="utf-8") as f:
            vocab_score_lines = f.readlines()
        self.vocab_scores = [
            line.strip().split() for line in vocab_score_lines if line.strip()
        ]
        logger.mesg(f"{brk(len(self.vocab_scores))}")
        logger.file(f"  * {self.vocab_path}")

    def to_txt(
        self,
        ignore_score: bool = True,
        sep: str = ",",
        de_mask: bool = True,
        ignore_eng_word: bool = True,
        ignore_one_char: bool = True,
        ignore_one_word: bool = True,
    ) -> Path:
        self.load_vocabs()
        logger.note("> Convert vocab to txt:", end=" ")
        lines = []
        for vocab_score in self.vocab_scores:
            if len(vocab_score) == 2:
                vocab, score = vocab_score
            else:
                vocab = vocab_score[0]
                score = ""
            if ignore_eng_word and PT_ENG_WORD.match(vocab):
                continue
            if ignore_one_char and len(vocab) <= 1:
                continue
            if de_mask and len(vocab) > 1:
                vocab = vocab.replace(CH_MASK, " ")
            if ignore_one_word and PT_ONE_WORD.match(vocab):
                continue
            if ignore_score:
                line = vocab
            else:
                line = f"{vocab}{sep}{score}"
            lines.append(line)
        with open(self.txt_path, "w", encoding="utf-8") as wf:
            wf.write("\n".join(lines) + "\n")
        logger.mesg(f"{brk(len(lines))}")
        logger.okay(f"  * {self.txt_path}")


class WordRecordsConverter:
    def __init__(self, csv_path: PathType):
        self.csv_path = Path(csv_path)
        self.txt_path = self.csv_path.with_suffix(".txt")

    def set_csv_path(self, csv_path: PathType):
        self.csv_path = Path(csv_path)
        self.txt_path = self.csv_path.with_suffix(".txt")

    def load_csv(self):
        logger.note(f"> Loading csv:", end=" ")
        df = pl.read_csv(self.csv_path)
        logger.mesg(f"{brk(len(df))}")
        logger.file(f"  * {self.csv_path}")
        return df

    def save_txt(self, words: list[str]):
        with open(self.txt_path, "w", encoding="utf-8") as wf:
            wf.write("\n".join(words) + "\n")
        logger.mesg(f"{brk(len(words))}")
        logger.okay(f"  * {self.txt_path}")

    def to_txt(self):
        df = self.load_csv()
        logger.note("> Save to txt:", end=" ")
        # only keep first column, and remove header
        words = df.select(df.columns[0]).to_series().to_list()
        self.save_txt(words)


class VocabsMerger:
    def load_vocabs(self, path: PathType) -> list[str]:
        logger.note(f"> Loading vocabs:", end=" ")
        path = Path(path)
        vocabs = []
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            vocab = line.strip()
            if vocab:
                vocabs.append(vocab)
        logger.mesg(f"{brk(len(vocabs))}")
        logger.file(f"  * {path}")
        return vocabs

    def save_vocabs(self, vocabs: list[str], save_path: PathType = None):
        save_path = Path(save_path or Path(__file__).parent / "vocabs.txt")
        logger.note(f"> Saving vocabs:", end=" ")
        with open(save_path, "w", encoding="utf-8") as wf:
            wf.write("\n".join(vocabs) + "\n")
        logger.mesg(f"{brk(len(vocabs))}")
        logger.okay(f"  * {save_path}")

    def merge(self, paths: PathsType, save_path: PathType = None):
        merged_vocabs = []
        for path in paths:
            vocabs = self.load_vocabs(path)
            merged_vocabs.extend(vocabs)
        total_count = len(merged_vocabs)
        merged_vocabs = list(set(merged_vocabs))
        nodup_count = len(merged_vocabs)
        logger.mesg(
            f"* Merged vocabs: "
            f"{logstr.okay(nodup_count)}/{logstr.mesg(total_count)}"
        )
        self.save_vocabs(merged_vocabs, save_path)


class ConverterMergerArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("-s", "--sentencepiece", action="store_true")
        self.add_argument("-r", "--record", action="store_true")
        self.add_argument("-m", "--merge", action="store_true")
        self.add_argument("-o", "--save-path", type=str)
        self.args, _ = self.parse_known_args()


def main():
    arg_parser = ConverterMergerArgParser()
    args = arg_parser.args

    txt_paths = []

    if args.sentencepiece:
        sp_converter = SentencePieceConverter()
        sp_converter.to_txt()
        txt_paths.append(sp_converter.txt_path)

    if args.record:
        doc_count = 770000000
        en_csv_path = get_dump_path(doc_count, lang="en")
        word_converter = WordRecordsConverter(csv_path=en_csv_path)
        word_converter.to_txt()
        txt_paths.append(word_converter.txt_path)

        zh_csv_path = get_dump_path(doc_count, lang="zh")
        word_converter.set_csv_path(zh_csv_path)
        word_converter.to_txt()
        txt_paths.append(word_converter.txt_path)

    if args.merge:
        if not txt_paths:
            logger.warn("No paths to merge. Please specify args: `-s` and `-r`.")
            return
        merger = VocabsMerger()
        merger.merge(txt_paths, save_path=args.save_path)


if __name__ == "__main__":
    main()

    # python -m models.sentencepiece.convert -s -r -m
