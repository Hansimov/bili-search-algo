from pathlib import Path
from typing import Union
from tclogger import logger, logstr, brk

from configs.envs import SP_MERGED_MODEL_PATH

CH_MASK = "▂"


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

    def to_txt(self, keep_score: bool = False, sep: str = ",", de_mask: bool = True):
        logger.note("> Convert vocab to txt:")
        txt_path = self.model_path.with_suffix(".txt")
        lines = []
        for vocab_score in self.vocab_scores:
            if len(vocab_score) == 2:
                vocab, score = vocab_score
            else:
                vocab = vocab_score[0]
                score = ""
            if de_mask and len(vocab) > 1:
                vocab = vocab.replace(CH_MASK, " ")
            if keep_score:
                line = f"{vocab}{sep}{score}"
            else:
                line = vocab
            lines.append(line)
        with open(txt_path, "w", encoding="utf-8") as wf:
            wf.write("\n".join(lines) + "\n")
        logger.okay(f"  * {txt_path}")


if __name__ == "__main__":
    converter = SentencePieceConverter()
    converter.to_txt()

    # python -m models.sentencepiece.convert
