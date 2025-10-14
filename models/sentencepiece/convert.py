import argparse
import re
import polars as pl

from pathlib import Path
from tclogger import PathType, PathsType, logger, logstr, brk, chars_len

from configs.envs import SP_MERGED_MODEL_PATH
from models.word.eng import get_dump_path

CH_MASK = "▂"
RE_ONE_WORD = r"^[a-zA-Z]+$"
PT_ONE_WORD = re.compile(RE_ONE_WORD)
RE_ENG_WORD = r"^[a-zA-Z\s▂\-\_]+$"
PT_ENG_WORD = re.compile(RE_ENG_WORD)

# RE_SPECIALS = r"[()（）\[\]【】{}<>《》「」‘’“”'\"`~!！@#$￥%^&*+=;；:：,，。?？\\|\/]"
RE_SPECIALS = r"[()\[\]{}<>'\"`~!@#$%^&*+=;,?|\\\/]"
PT_SPECIALS = re.compile(RE_SPECIALS)


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

    def filter_vocabs(
        self,
        ignore_score: bool = True,
        sep: str = ",",
        de_mask: bool = True,
        ignore_eng_word: bool = True,
        ignore_one_char: bool = True,
        ignore_one_word: bool = True,
    ) -> list[str]:
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
        return lines

    def save_txt(self, lines: list[str]):
        logger.note("> Save to txt:", end=" ")
        with open(self.txt_path, "w", encoding="utf-8") as wf:
            wf.write("\n".join(lines) + "\n")
        logger.mesg(f"{brk(len(lines))}")
        logger.okay(f"  * {self.txt_path}")

    def to_txt(self):
        self.load_vocabs()
        lines = self.filter_vocabs()
        self.save_txt(lines)


class WordRecordsConverter:
    def __init__(self, min_doc_freq: int = 20, max_char_len: int = 32):
        self.min_doc_freq = min_doc_freq
        self.max_char_len = max_char_len

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

    def filter_words(
        self,
        words: list[str],
        ignore_one_char: bool = True,
        ignore_specials: bool = True,
    ) -> list[str]:
        filtered_words = []
        for word in words:
            if ignore_one_char and len(word) <= 1:
                continue
            # word contain any special characters
            if ignore_specials and PT_SPECIALS.search(word):
                continue
            if self.max_char_len and chars_len(word) > self.max_char_len:
                continue
            filtered_words.append(word)
        return filtered_words

    def to_txt(self):
        df = self.load_csv()
        logger.note("> Save to txt:", end=" ")
        # filter by doc_freq
        if "doc_freq" in df.columns:
            df = df.filter(pl.col("doc_freq") >= self.min_doc_freq)
        # only keep first column ("word"), and remove header
        words = df.select(df.columns[0]).to_series().to_list()
        words = self.filter_words(words)
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
        logger.note(f"> Sorting vocabs:")
        # sort by len(vocab) asc, then alphabetically
        merged_vocabs.sort(key=lambda x: (chars_len(x), x))
        self.save_vocabs(merged_vocabs, save_path)


class ConverterMergerArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("-s", "--sentencepiece", action="store_true")
        self.add_argument("-r", "--record", action="store_true")
        self.add_argument("-m", "--merge", action="store_true")
        self.add_argument("-n", "--min-doc-freq", type=int, default=20)
        self.add_argument("-l", "--max-char-len", type=int, default=32)
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
        word_converter = WordRecordsConverter(
            min_doc_freq=args.min_doc_freq, max_char_len=args.max_char_len
        )
        word_converter.set_csv_path(en_csv_path)
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

    # python -m models.sentencepiece.convert -s -r -m -n 20 -l 32
