import re

from pathlib import Path
from typing import Union
from tclogger import logger, brk

from configs.envs import SP_MERGED_MODEL_PATH

CH_MASK = "▂"
RE_ONE_WORD = r"^[a-zA-Z]+$"
PT_ONE_WORD = re.compile(RE_ONE_WORD)


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
        ignore_one_char: bool = True,
        ignore_one_word: bool = True,
    ):
        logger.note("> Convert vocab to txt:")
        txt_path = self.model_path.with_suffix(".txt")
        lines = []
        for vocab_score in self.vocab_scores:
            if len(vocab_score) == 2:
                vocab, score = vocab_score
            else:
                vocab = vocab_score[0]
                score = ""
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
        logger.okay(f"  * {txt_path}")


if __name__ == "__main__":
    converter = SentencePieceConverter()
    converter.to_txt()

    # python -m models.sentencepiece.convert
