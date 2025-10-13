import re
import polars as pl

from pathlib import Path
from typing import Union
from tclogger import PathType, logger, brk

from configs.envs import SP_MERGED_MODEL_PATH
from models.word.eng import get_dump_path

CH_MASK = "▂"
RE_ONE_WORD = r"^[a-zA-Z]+$"
PT_ONE_WORD = re.compile(RE_ONE_WORD)
RE_ENG_WORD = r"^[a-zA-Z\s▂\-\_]+$"
PT_ENG_WORD = re.compile(RE_ENG_WORD)


class SentencePieceConverter:
    def __init__(self, model_path: Union[str, Path] = SP_MERGED_MODEL_PATH):
        self.model_path = Path(model_path)
        self.load_vocabs()

    def load_vocabs(self):
        logger.note(f"> Loading vocab:", end=" ")
        self.vocab_path = self.model_path.with_suffix(".vocab")
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
    ):
        logger.note("> Convert vocab to txt:", end=" ")
        txt_path = self.model_path.with_suffix(".txt")
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
        with open(txt_path, "w", encoding="utf-8") as wf:
            wf.write("\n".join(lines) + "\n")
        logger.mesg(f"{brk(len(lines))}")
        logger.okay(f"  * {txt_path}")


class WordRecordsConverter:
    def __init__(self, csv_path: PathType):
        self.csv_path = Path(csv_path)

    def set_csv_path(self, csv_path: PathType):
        self.csv_path = Path(csv_path)

    def load_csv(self):
        logger.note(f"> Loading csv:", end=" ")
        df = pl.read_csv(self.csv_path)
        logger.mesg(f"{brk(len(df))}")
        logger.file(f"  * {self.csv_path}")
        return df

    def to_txt(self):
        df = self.load_csv()
        logger.note("> Save to txt:", end=" ")
        # only keep first column, and remove header
        words = df.select(df.columns[0]).to_series().to_list()
        txt_path = self.csv_path.with_suffix(".txt")
        with open(txt_path, "w", encoding="utf-8") as wf:
            wf.write("\n".join(words) + "\n")
        logger.mesg(f"{brk(len(words))}")
        logger.okay(f"  * {txt_path}")


def main():
    # sp_converter = SentencePieceConverter()
    # sp_converter.to_txt()

    doc_count = 770000000
    en_csv_path = get_dump_path(doc_count, lang="en")
    word_converter = WordRecordsConverter(csv_path=en_csv_path)
    word_converter.to_txt()

    zh_csv_path = get_dump_path(doc_count, lang="zh")
    word_converter.set_csv_path(zh_csv_path)
    word_converter.to_txt()


if __name__ == "__main__":
    main()

    # python -m models.sentencepiece.convert
